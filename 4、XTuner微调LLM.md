# 模型微调

## Finetune

### 为什么微调？

使通用大模型在某些领域中表现更好

### 两种Fine tune范式

1. 增量预训练

   使用场景：让base模型学习到新知识

   训练数据：文章、代码、书籍等

2. 指令跟随微调

   使用场景：让模型学会对话模板，根据人类指令进行对话

   训练数据：**高质量**的对话、问答数据

![image-20240417150912599](https://s2.loli.net/2024/04/17/mkXivsBSdHTojcM.png)

### 数据处理

![image-20240422145352879](https://s2.loli.net/2024/04/22/KzgeGARN4nJSaMr.png)

`标准格式数据`:训练框架能够识别的数据

![image-20240422145742226](https://s2.loli.net/2024/04/22/O1BuydFa2x4UbzL.png)

`对话模板`:让LLM能够区分出System、User、Assistant

![image-20240422150028117](https://s2.loli.net/2024/04/22/McyTN1qY3RwVhBD.png)

数据tokeization并添加label

![image-20240422150747065](https://s2.loli.net/2024/04/22/SfVtsgeiwXuFlnJ.png)

预训练时只对output部分计算loss

![image-20240422150909051](https://s2.loli.net/2024/04/22/zPYugnNkAf2h3rl.png)

![image-20240422150918229](https://s2.loli.net/2024/04/22/Rysri57qEVdZJh3.png)

### 常用的微调算法

Lora和QLora

![image-20240422151247709](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240422151247709.png)

## 多模态模型

![image-20240422152019220](https://s2.loli.net/2024/04/22/nCJKklDmNrpYxQz.png)

### LLaVA模型

使用文本描述+图像作为训练数据，配合文本单模态LLM训练

![image-20240422152249121](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240422152249121.png)

- #### 模型转换、整合、测试、部署

  - `模型转换`:模型转换的本质其实就是将原本使用 Pytorch 训练出来的模型权重文件转换为目前通用的 Huggingface 格式文件(adapter_model.bin文件，可以简单理解：LoRA 模型文件 = Adapter)
  - `模型整合`：对于 LoRA 或者 QLoRA 微调出来的模型其实并不是一个完整的模型，而是一个额外的层（adapter），训练完的这个层最终还是要与原模型进行组合才能被正常的使用。对于全量微调的模型其实是不需要进行整合这一步的，因为全量微调修改的是原模型的权重而非微调一个新的 adapter ，因此是不需要进行模型整合的。

![image-20240422164728961](https://s2.loli.net/2024/04/22/NKknO8c32TqIERy.png)

## 基础作业

- 训练自己的小助手认知

  数据准备+模型准备$\Rightarrow$配置文件$\Rightarrow$使用DeepSpeed加速训练(可选)$\Rightarrow$模型转换、整合、测试、部署

  ![image-20240422162439836](https://s2.loli.net/2024/04/22/pf4svKYarIoNEub.png)

  web版
  
  ![image-20240422201852943](https://s2.loli.net/2024/04/22/9QDVjTGHw8ClEY7.png)

## 进阶作业

- 将自我认知的模型上传到 OpenXLab，并将应用部署到 OpenXLab

  参考文档OpenXLab 部署 [OpenXLab 部署 InternLM2 实践指南](https://github.com/InternLM/Tutorial/tree/camp2/tools/openxlab-deploy)

  ![image-20240428203300696](https://s2.loli.net/2024/04/28/QKDXctBp1bZ3A8I.png)

- 复现多模态微调

  修改配置文件`llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py` 

  | 变量                          | 值                                                           |
  | ----------------------------- | ------------------------------------------------------------ |
  | `pretrained_pth`              | `/root/share/new_models/xtuner/iter_2181.pth`                |
  | `llm_name_or_path`            | `/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b` |
  | `visual_encoder_name_or_path` | `/root/share/new_models/openai/clip-vit-large-patch14-336`   |
  | `data_root`                   | `/root/xtuner0117/data/`                                     |
  | `data_path`                   | `data_root + 'repeated_data.json'`                           |
  | `image_folder`                | `data_root`                                                  |
  | `batch_size`                  | `1`                                                          |
  | `evaluation_inputs`           | `['Please describe this picture','What is the equipment in the image?']` |

 	开始Finetune

```bash
cd /root/tutorial/xtuner/llava/
xtuner train /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py --deepspeed deepspeed_zero2
```

![image-20240428205536695](https://s2.loli.net/2024/04/28/9zWYdM4n6TFeO2X.png)

​	对比Finetune前后的性能差异

```bash
#Finetune前

# pth转huggingface
xtuner convert pth_to_hf \
  llava_internlm2_chat_1_8b_clip_vit_large_p14_336_e1_gpu8_pretrain \
  /root/share/new_models/xtuner/iter_2181.pth \
  /root/tutorial/xtuner/llava/llava_data/iter_2181_hf

# 启动！
xtuner chat /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
  --visual-encoder /root/share/new_models/openai/clip-vit-large-patch14-336 \
  --llava /root/tutorial/xtuner/llava/llava_data/iter_2181_hf \
  --prompt-template internlm2_chat \
  --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
  
  
# Finetune后  

# pth转huggingface
xtuner convert pth_to_hf \
  /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py \
  /root/tutorial/xtuner/llava/work_dirs/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy/iter_1200.pth \
  /root/tutorial/xtuner/llava/llava_data/iter_1200_hf

# 启动！
xtuner chat /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
  --visual-encoder /root/share/new_models/openai/clip-vit-large-patch14-336 \
  --llava /root/tutorial/xtuner/llava/llava_data/iter_1200_hf \
  --prompt-template internlm2_chat \
  --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
```

Q1: Describe this image.
Q2: What is the equipment in the image?

Finetune前：只会打标题

![image-20240428210734644](https://s2.loli.net/2024/04/28/wsl1FRQWCzGgqM2.png)

Finetune后：会回答问题了

![image-20240428211406483](https://s2.loli.net/2024/04/28/etdygbh6Knmwlfp.png)
