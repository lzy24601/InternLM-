## 笔记

### 部署基本环境

```bash
#python和pytorch版本
conda create -n demo python==3.10 -y
conda activate demo
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

#其他包
pip install huggingface-hub==0.17.3
pip install transformers==4.34 
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
```

### 下载**`InternLM2-Chat-1.8B` 模型**

```python
import os
#从魔搭下载模型
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')

```

执行命令，下载模型参数文件：

```bash
python /root/demo/download_mini.py
```

### 使用模型对话

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=====Welcome to InternLM chatbot, type 'exit' to exit.=====")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

```

### **运行 Chat-八戒 Demo**

```python
streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
#端口映射
#从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 39305 
#一直卡着可以换成ssh -C -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 39305 
```

![image-20240409143903726](https://s2.loli.net/2024/04/09/pFHadiO5KTCZDGs.png)

### 通过 InternLM2-Chat-7B 运行 Lagent智能体 Demo

Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。它的整个框架图如下:

![image-20240410121051167](https://s2.loli.net/2024/04/10/PWScqEzRwmLM7N2.png)

Lagent 的特性总结如下：

- 流式输出：提供 stream_chat 接口作流式输出，本地就能演示酷炫的流式 Demo。
- 接口统一，设计全面升级，提升拓展性，包括：
  - Model : 不论是 OpenAI API, Transformers 还是推理加速框架 LMDeploy 一网打尽，模型切换可以游刃有余；
  - Action: 简单的继承和装饰，即可打造自己个人的工具集，不论 InternLM 还是 GPT 均可适配；
  - Agent：与 Model 的输入接口保持一致，模型到智能体的蜕变只需一步，便捷各种 agent 的探索实现；
- 文档全面升级，API 文档全覆盖。

![image-20240410111357834](https://s2.loli.net/2024/04/10/beaYD86EcjsPxKf.png)

### 部署 浦语·灵笔2

`浦语·灵笔2` 是基于 `书生·浦语2` 大语言模型研发的突破性的图文多模态大模型，具有非凡的图文写作和图像理解能力，在多种应用场景表现出色，总结起来其具有：

- 自由指令输入的图文写作能力： `浦语·灵笔2` 可以理解自由形式的图文指令输入，包括大纲、文章细节要求、参考图片等，为用户打造图文并貌的专属文章。生成的文章文采斐然，图文相得益彰，提供沉浸式的阅读体验。
- 准确的图文问题解答能力：`浦语·灵笔2` 具有海量图文知识，可以准确的回复各种图文问答难题，在识别、感知、细节描述、视觉推理等能力上表现惊人。
- 杰出的综合能力： `浦语·灵笔2-7B` 基于 `书生·浦语2-7B` 模型，在13项多模态评测中大幅领先同量级多模态模型，在其中6项评测中超过 `GPT-4V` 和 `Gemini Pro`。

### Hugging Face模型下载

```python
# pip install -U huggingface_hub 先安装huggingface_hub包
import os 
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'使用镜像站
# os.environ得在import huggingface库相关语句之前执行。
from huggingface_hub import hf_hub_download  # Load model directly 

hf_hub_download(repo_id="internlm/internlm2-7b", filename="config.json", local_dir="D:\\LLM", resume_download=True)
```

## 作业

### **基础作业** 

- 使用 `InternLM2-Chat-1.8B` 模型生成 300 字的小故事

![image-20240409142413006](https://s2.loli.net/2024/04/09/lwRa97KPConYHM1.png)

### 进阶作业

- 熟悉 `huggingface` 下载功能，使用 `huggingface_hub` python 包，下载 `InternLM2-Chat-7B` 的 `config.json` 文件到本地

  ![image-20240410140734517](https://s2.loli.net/2024/04/10/UhSWYxnPB5ibIkJ.png)

  ![image-20240410135504155](https://s2.loli.net/2024/04/10/gCqmSplVF68GfUi.png)

  ![image-20240410135542765](https://s2.loli.net/2024/04/10/5ytC78LZBf4MYXR.png)

- 完成 `浦语·灵笔2` 的 `图文创作` 及 `视觉问答` 部署

  - 图文创作

    > 根据以下标题：“围棋：古老智慧的棋局”，创作长文章，字数不少于500字。
    > 请结合以下文本素材：围棋，又称“围棋”或“圍棋”，是一种源于中国的策略棋类游戏，以其深奥的规则和丰富的战术而闻名于世。在围棋中，黑白两方玩家交替在棋盘上落子，目标是通过占领更多的领地和捕获对方的棋子来赢得比赛。
    >
    > 围棋的棋盘上有着19*19个交叉点，这些交叉点形成了361个棋位，玩家可以在其中选择落子。初始时，棋盘是空的，玩家需要使用黑色和白色的棋子在棋盘上布局，以构建自己的势力和领地。
    >
    > 基本的围棋只需要棋盘和黑白两色的棋子，但随着游戏的发展，也出现了高级的围棋变体，如象棋围棋和五子棋等。在这些变体中，不仅有着丰富多彩的棋局，还有着不同的规则和战术。
    >
    > 在围棋中，以围棋特有的布局和棋局为主要素材，加以黑白两色的棋子的变化，营造出不同的战术和策略。每一步棋的走向，都可能影响到整个棋局的发展，因此，围棋被誉为是一种“棋艺”和“智慧”的象征。
    >
    > 随着围棋的流行，它已经不仅仅是一种游戏，更是一种文化，一种生活方式。在围棋的世界里，人们追求的不仅仅是胜利，更是对智慧和策略的追求，同时也是对文化传承和精神交流的传承。

    ![image-20240410114725783](https://s2.loli.net/2024/04/10/6Nqd2xyeXwJjHKV.png)

    ![image-20240410114753041](https://s2.loli.net/2024/04/10/WsPBTAMxkQUd4XZ.png)

    ![image-20240410114826320](https://s2.loli.net/2024/04/10/FCUSkzL6nIbjZod.png)

  - 视觉问答

    ![image-20240410115841313](https://s2.loli.net/2024/04/10/PUz6dlQDCEyu2p1.png)


- 完成 `Lagent` 工具调用 `数据分析` Demo 部署

![image-20240410111541356](https://s2.loli.net/2024/04/10/9IHSc4Cts6FOorQ.png)

