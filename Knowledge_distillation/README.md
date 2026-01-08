#知识蒸馏学习代码
环境配置：
    机器NVIDIA L40，显存占用大概43322MiB
    教师模型：qwen3-vl-4b
    学生模型: qweh3-vl-2b
    学生模型用Lora微调了10个epoch，目前只测试了前向KL，反向KL暂未测
训练情况：
    bs为1，gradient_accumulation_steps为8，一次迭代3秒左右
    损失稳定在33左右不下降了
精度情况：
    测试集1000张
    教师模型：
    蒸馏后的2b模型：

