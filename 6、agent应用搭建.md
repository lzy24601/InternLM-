

## agent

> LLM的局限性:
>
> **幻觉:**模型可能会生成虚假信息，与现实严重不符或脱节。
>
>  **时效性:**模型训练数据过时无法反映最新趋势和信息。
>
> **可靠性**：面对复杂任务时，可能频发错误输出现象

#### 什么是agent？

![image-20240425185443837](https://s2.loli.net/2024/04/25/y6MTsPWz1395eO8.png)

#### agent的组成

1. **感知（Perception）**：智能体能够通过感知器（如传感器、摄像头、或接收数据输入的接口）来观察和接收环境中的信息。这允许智能体获取关于其所处环境的实时数据。
2. **决策（Decision Making）**：基于收集到的信息，智能体需要处理和评估数据，做出决策。这通常涉及一定的推理或学习过程，以确定如何有效地响应或改变环境以达到其目标。
3. **行动（Action）**：智能体能够通过行为器（如机械臂、输出信号、或发送指令等）对环境进行干预。行动是基于智能体的决策过程，并旨在推动智能体向其目标前进。

#### LLM中的智能体

> **agent范式**：

1. Auto-GPT

   ![image-20240425185806493](https://s2.loli.net/2024/04/25/cY9vG5JNgaZCxRb.png)

2. ReWoo

   ![image-20240425185926204](https://s2.loli.net/2024/04/25/VHRW5p7P4wQc9hS.png)

3. ReAct

   ![image-20240425185950271](https://s2.loli.net/2024/04/25/kqm5sXMvgRu7HoY.png)

### Lagent框架

> Lagent 是一个轻量级开源智能体框架，旨在让用户可以高效地构建基于大语言模型的智能体。同时它也提供了一些典型工具以增强大语言模型的能力。
>
> Lagent 目前已经支持了包括 AutoGPT、ReAct 等在内的多个经典智能体范式，也支持了如下工具：
>
> - Arxiv 搜索
> - Bing 地图
> - Google 学术搜索
> - Google 搜索
> - 交互式 IPython 解释器
> - IPython 解释器
> - PPT
> - Python 解释器

Lagent架构如下：

![image-20240425190658997](https://s2.loli.net/2024/04/25/gxpQVin6zjsL7aY.png)

### AgentLego包

> AgentLego 是一个提供了多种开源工具 API 的多模态工具包，旨在像是乐高积木一样，让用户可以快速简便地拓展自定义工具，从而组装出自己的智能体。通过 AgentLego 算法库，不仅可以直接使用多种工具，也可以利用这些工具，在相关智能体框架（如 Lagent，Transformers Agent 等）的帮助下，快速构建可以增强大语言模型能力的智能体。

AgentLego 目前提供了如下工具：

![image-20240425193401342](https://s2.loli.net/2024/04/25/9TzSZANhlWEnQwI.png)

Lagent 是一个智能体框架，而 AgentLego 与大模型智能体并不直接相关，而是作为工具包，在相关智能体的功能支持模块发挥作用。

- Lagent有三个任务：调用工具，工具功能支持，工具输出

- agentlego实现的是工具功能支持的自定义任务。

两者之间的关系可以用下图来表示：

![image-20240425190525733](https://s2.loli.net/2024/04/25/2gZ1CldSyTmVIiG.png)

## 基础作业



1. 完成 Lagent Web Demo 使用，文档可见 [Lagent Web Demo](https://github.com/InternLM/Tutorial/blob/camp2/agent/lagent.md#1-lagent-web-demo)

   使用LMDeploy启动api_server并启动Lagent Web Demo

   ```bash
   # 两个端口映射
   ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 你的 ssh 端口号
   ```

   ![image-20240425195837635](https://s2.loli.net/2024/04/25/BZqAXezUj5FyQdR.png)

2. 完成 AgentLego 直接使用部分，文档可见 [直接使用 AgentLego](https://github.com/InternLM/Tutorial/blob/camp2/agent/agentlego.md#1-直接使用-agentlego)。

   **环境搭建**

   ```bash
   # 下载demo文件
   cd /root/agent
   wget http://download.openmmlab.com/agentlego/road.jpg
   
   # AgentLego 所实现的目标检测工具是基于 mmdet (MMDetection) 算法库中的 RTMDet-Large 模型
   # 因此我们首先安装 mim，然后通过 mim 工具来安装 mmdet。
   conda activate agent
   pip install openmim==0.3.9
   mim install mmdet==3.3.0
   ```

   **使用目标检测工具**

   ```python
   import re
   
   import cv2
   from agentlego.apis import load_tool
   
   # load tool
   tool = load_tool('ObjectDetection', device='cuda')
   
   # apply tool
   visualization = tool('/root/agent/road.jpg')
   print(visualization)
   
   # visualize
   image = cv2.imread('/root/agent/road.jpg')
   
   preds = visualization.split('\n')
   pattern = r'(\w+) \((\d+), (\d+), (\d+), (\d+)\), score (\d+)'
   
   for pred in preds:
       name, x1, y1, x2, y2, score = re.match(pattern, pred).groups()
       x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), int(score)
       cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
       cv2.putText(image, f'{name} {score}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
   
   cv2.imwrite('/root/agent/road_detection_direct.jpg', image)
   ```

   ![image-20240425204821430](https://s2.loli.net/2024/04/25/GQtxipLCflWbU3O.png)

## 进阶作业



1. 完成 AgentLego WebUI 使用，文档可见 [AgentLego WebUI](https://github.com/InternLM/Tutorial/blob/camp2/agent/agentlego.md#2-作为智能体工具使用)。

   ![image-20240425210548451](https://s2.loli.net/2024/04/25/woHxdIuaNKTWf2y.png)

2. 使用 Lagent 或 AgentLego 实现自定义工具并完成调用，文档可见：
   
   - [用 Lagent 自定义工具](https://github.com/InternLM/Tutorial/blob/camp2/agent/lagent.md#2-用-lagent-自定义工具)
   
   用 Lagent 自定义工具主要分为以下几步：
   
   1. 继承 BaseAction 类
   2. 实现简单工具的 run 方法；或者实现工具包内每个子工具的功能
   3. 简单工具的 run 方法可选被 tool_api 装饰；工具包内每个子工具的功能都需要被 tool_api 装饰
   
   ![image-20240425202322337](https://s2.loli.net/2024/04/25/4stxl7i3V9NGrMz.png)
   
   - [用 AgentLego 自定义工具](https://github.com/InternLM/Tutorial/blob/camp2/agent/agentlego.md#3-用-agentlego-自定义工具)
   
     自定义工具主要分为以下几步：
   
     1. 继承 BaseTool 类
     2. 修改 default_desc 属性（工具功能描述）
     3. 如有需要，重载 setup 方法（重型模块延迟加载）
     4. 重载 apply 方法（工具功能实现）
   
     其中第一二四步是必须的步骤。下面将实现一个调用 MagicMaker 的 API 以实现图像生成的工具。
   
     [MagicMaker](https://magicmaker.openxlab.org.cn/home) 是汇聚了优秀 AI 算法成果的免费 AI 视觉素材生成与创作平台。主要提供图像生成、图像编辑和视频生成三大核心功能，全面满足用户在各种应用场景下的视觉素材创作需求。
   
     ![image-20240425212136915](https://s2.loli.net/2024/04/25/qPHpSTf37ohMZ2L.png)

## 大作业选题



### 算法方向



1. 在 Lagent 或 AgentLego 中实现 RAG 工具，实现智能体与知识库的交互。
2. 基于 Lagent 或 AgentLego 实现工具的多轮调用，完成复杂任务。如：智能体调用翻译工具，再调用搜索工具，最后调用生成工具，完成一个完整的任务。

### 应用方向



1. 基于 Lagent 或 AgentLego 实现一个客服智能体，帮助用户解决问题。
2. 基于 Lagent 或 AgentLego 实现一个智能体，实现艺术创作，如生成图片、视频、音乐等。