import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径
target_model_name = "/mnt/nova_ssd/zgf_doc/llm_acceleration_learning/speculative_sample/models--Qwen--Qwen2.5-3B-Instruct/snapshots/qwen2.5-3b"

# 系统提示模板
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. 你是一个乐于助人的助手。"
TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

# 输入文本
inputs = ["写个100字的诗歌"]
inputs = [generate_prompt(text) for text in inputs]

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(target_model_name)
print("begin loading model")

# 加载模型
target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    load_in_8bit=False
)
target_model.eval()
print(f"Load {target_model_name} finish")

torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

for text in inputs:
    input_ids = tokenizer.encode(text, return_tensors='pt').to(torch_device)

    start_time = time.time()

    # 使用 generate 生成文本
    with torch.no_grad():
        generated_ids = target_model.generate(
            input_ids,
            max_new_tokens=200,  # 控制生成长度
            do_sample=True,      # 开启采样
            top_p=0.9,           # nucleus sampling
            temperature=0.7,     # 温度控制
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码生成的 token
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("生成结果:\n", generated_text)

    end_time = time.time()
    print("cost_time:", end_time - start_time)