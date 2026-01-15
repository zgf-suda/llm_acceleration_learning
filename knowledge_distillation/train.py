from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator,Qwen3VLForConditionalGeneration,AutoProcessor,get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from utils import compute_fkl, compute_rkl, compute_skewed_fkl, compute_skewed_rkl
from torch.utils.data import IterableDataset, Dataset
import json
from PIL import Image
import os
class KGTrainer(Trainer):
    
    def __init__(
        self,
        model = None,
        teacher_model = None,
        if_use_entropy = False,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None, 
        compute_metrics = None, 
        callbacks = None,
        preprocess_logits_for_metrics = None
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        # outputs = model(
        #     input_ids=inputs["input_ids"],
        #     labels=inputs["labels"],
        #     pixel_values=inputs["pixel_values"],
        #     image_grid_thw=inputs["image_grid_thw"]
        #     )

        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        loss = outputs.loss  #学生模型的输出
        logits = outputs.logits  #[batch_size, seq_len, vocab_size]
        teacher_logits = teacher_outputs.logits
        
        # 如果教师模型和学生模型输出形状不匹配，对学生模型进行padding或对教师模型进行截断
        if logits.shape[-1] != teacher_logits.shape[-1]:
            # gap = teacher_logits.shape[-1] - logits.shape[-1]
            # if gap > 0:
            #     pad_logits = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
            #     logits = torch.cat([logits, pad_logits], dim=-1)
            
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        labels = inputs['labels']
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=tokenizer.pad_token_id, temp=2.0)
        
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        #print("cur_loss:",loss_total)
        return (loss_total, outputs) if return_outputs else loss_total
        

# def find_assistant_tokens(tokenizer, target):
#     result = []
#     start_index =0
#     end_index = 0
#     while start_index <= len(target)-1:
#         if target[start_index]!=tokenizer('assistant')['input_ids'][0]:
#             start_index+=1
#             end_index+=1
#         else:
#             end_index+=1
#             if target[end_index]==tokenizer('<|im_end|>')['input_ids'][0]:
#                 result.append((start_index+1,end_index+1))
#                 start_index=end_index+1
#     return result

def find_assistant_tokens(tokenizer, input_ids: list[int]):
    """
    返回每条 assistant 输出文本的 token start/end 索引
    保证 <image> token 不算作 assistant 的文本
    """
    result = []
    i = 0
    n = len(input_ids)
    assistant_id = tokenizer("assistant")["input_ids"][0]
    im_end_id = tokenizer("<|im_end|>")["input_ids"][0]
    image_id = tokenizer("<|image_pad|>")["input_ids"][0]
    # print("assistant_id", assistant_id)
    # print("im_end_id", im_end_id)
    # print("image_id", image_id)
    while i < n:
        if input_ids[i] == assistant_id:
            # 找到 assistant 输出开始
            start = i + 1  # assistant token 后第一个 token
            end = start
            while end < n and input_ids[end] != im_end_id:
                if input_ids[end] == image_id:
                    # 跳过 <image> token
                    end += 1
                else:
                    end += 1
            result.append((start, end))
            i = end + 1
        else:
            i += 1
    return result


class SFTDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)   
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = str(sample['image'])
            conversations = sample['conversations']
            messages = [{"role":"system", "content":'You are a helpful assistant.'}]
            for conversation in conversations:
                if conversation['from'] == 'human':
                    messages.append({"role":"user", "content":conversation['value']})
                else:
                    messages.append({"role":"assistant", "content":conversation['value']})
            #[{"role":"system","content":""},{},{}]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            ).replace('<image>', '<|image_pad|>')  #qwen3叫image_pad，只需要 1 个 <|image_pad|>，图像embedding自动填入这里

            image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
            encoding = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True
            )
            input_ids = encoding["input_ids"].squeeze(0) #所有的token id 包括了图像图像占位符。前向推理时再把图像像素值一起丢给模型处理
            attention_mask = encoding["attention_mask"].squeeze(0)
            pixel_values = encoding["pixel_values"].squeeze(0)
            image_grid_thw = encoding["image_grid_thw"].squeeze(0)

            indexs = find_assistant_tokens(tokenizer, input_ids.tolist())
            labels = len(input_ids) * [tokenizer.pad_token_id]
            for index in indexs:
                #print("index:",index)
                labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
            input_ids = input_ids[:-1]
            labels = labels[1:] 
            #print("attention_mask",attention_mask)  #只有是pad处为0
            #print("tokenizer.pad_token_id",tokenizer.pad_token_id)  endoftext
            #print("image_name:",image_name)
            #151643:endoftext ;151644:im_start ; 151645:im_end ; 151655:image_pad
            #print("input_ids:",(input_ids))
            #print("labels:",labels)
        except:
            import traceback
            print(traceback.format_exc())
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values,
            "image_grid_thw": image_grid_thw,
            "attention_mask":attention_mask
        }   

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: list[dict[str, any]]) -> dict[str, any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        attention_mask = []
        image_grid_thw = []
        for feature in features:
            input_ids.append(torch.cat([feature['input_ids'] , torch.full(((max_len - len(feature['input_ids'])),),(self.tokenizer.pad_token_id),dtype=torch.long)],dim=0))
            labels.append(torch.cat([torch.tensor(feature['labels'],dtype=torch.long) ,torch.full((max_len-len(feature['labels']),),(self.tokenizer.pad_token_id),dtype=torch.long)],dim=0))
            pixel_values.append(feature['pixel_values'])
            attention_mask.append(torch.tensor(([1] * len(feature["input_ids"]) + [0] * (max_len - len(feature['input_ids']))),dtype=torch.long))
            image_grid_thw.append(feature["image_grid_thw"])
        #print("image_grid:",image_grid_thw)
        return {'input_ids': torch.stack(input_ids),
                'labels': torch.stack(labels),
                "attention_mask": torch.stack(attention_mask,dim=0),
                'pixel_values': torch.stack(pixel_values, dim=0),
                "image_grid_thw": torch.stack(image_grid_thw,dim=0),
                }

if __name__ == '__main__':
    teacher_model_path = "/home/user/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/qwen3-vl-4b"
    student_model_path = "/home/user/qwen3-vl-2b"
    images_path = '/home/user/KD_qwen2.5-vl/images'
    data_path = '/home/user/KD_qwen2.5-vl/sft_train.json'

    torch.cuda.empty_cache()#清理显存
    # 学生模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(student_model_path)
    #print(model)
    #只训练LLM，视觉编码器和投影不训练
    for p in model.language_model.parameters():
        p.requires_grad = True 

    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM) #自回归任务
    #使用lora方法训练
    model = get_peft_model(model, lora_config)
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    
    # 教师模型，在给定数据上通过lora微调
    teacher_model = Qwen3VLForConditionalGeneration.from_pretrained(teacher_model_path)
    # 是否加载lora模型
    teacher_model.cuda()
    teacher_model.eval()
    
    args = TrainingArguments(output_dir='./results', 
                            num_train_epochs=10, 
                            do_train=True, 
                            per_device_train_batch_size=1,
                            gradient_accumulation_steps=8,
                            logging_steps=10,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=10,
                            bf16=True,
                            learning_rate=0.0005,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)

    processor = AutoProcessor.from_pretrained(student_model_path)
    #print("processor:",processor)
    dataset = SFTDataset(images_path, data_path, tokenizer, processor)
   
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        if_use_entropy = True,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=MyDataCollator(tokenizer),
                       )
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()
    
      
    