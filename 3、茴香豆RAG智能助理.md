## RAG

RAG 能够让基础模型实现非参数知识更新，无需训练就可以掌握新领域的知识。

RAG 技术的优势就是非参数化的模型调优

![image-20240410185438387](https://s2.loli.net/2024/04/10/GvsWPLH13Bo6yCf.png)

将用户输入作为索引，在外部知识库中搜寻相关内容，结合大模型的能力生成回答

![image-20240410185732366](https://s2.loli.net/2024/04/10/BJ5Vb4CxDA3fl1a.png)

#### 向量数据库

![image-20240410185848699](https://s2.loli.net/2024/04/10/QxOZr8XSFW31CPp.png)

![image-20240410190017719](https://s2.loli.net/2024/04/10/UFRylg5LGbO8f2E.png)

![image-20240410190517727](https://s2.loli.net/2024/04/10/78foEGRWFLvZiyM.png)



## 基础作业

### 在茴香豆 Web 版中创建自己领域的知识问答助手

![image-20240416210825518](https://s2.loli.net/2024/04/16/vRrEbue89Bj4tCz.png)

![image-20240416211140840](https://s2.loli.net/2024/04/16/2JWwxX4pBhI15sT.png)

![image-20240416210852781](https://s2.loli.net/2024/04/16/3St9dYvnzGjefqm.png)

![image-20240416210919804](https://s2.loli.net/2024/04/16/wECyLNuVAbMeXzR.png)

![image-20240416211203654](https://s2.loli.net/2024/04/16/yqzwehNGWQVmgF2.png)

![image-20240416211215663](https://s2.loli.net/2024/04/16/76eySUiRzIVBK4D.png)

![image-20240416211323403](https://s2.loli.net/2024/04/16/nojKZlHNT16qtXf.png)

### 在 InternLM Studio 上部署茴香豆技术助手

- 创建知识库

- 将问题添加到接受(good_questions.json)\拒答(bad_questions.json)问题列表

- 创建RAG检索过程中使用的向量数据库

  ```python
  # 分别向量化知识语料、接受问题和拒绝问题中后保存到 workdir
  python3 -m huixiangdou.service.feature_store --sample ./test_queries.json
  ```

- 填入问题后运行

  ```python
  # 填入问题
  sed -i '74s/.*/    queries = ["huixiangdou 是什么？", "茴香豆怎么部署到微信群", "今天天气怎么样？"]/' /root/huixiangdou/huixiangdou/main.py
  
  # 运行茴香豆
  cd /root/huixiangdou/
  python3 -m huixiangdou.main --standalone
  ```

  ![image-20240417115706425](https://s2.loli.net/2024/04/17/2bMuxtGTKUdIj8i.png)

## 进阶作业

### A.【应用方向】 结合自己擅长的领域知识（游戏、法律、电子等）、专业背景，搭建个人工作助手或者垂直领域问答助手，参考茴香豆官方文档，部署到下列任一平台。

- 飞书、微信
- 可以使用 茴香豆 Web 版 或 InternLM Studio 云端服务器部署
- 涵盖部署全过程的作业报告和个人助手问答截图

#### 在[茴香豆Web](https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web)中创建知识库并上传相关文档

![image-20240417100249186](https://s2.loli.net/2024/04/17/FMxtlYPqgmRSV1i.png)

![image-20240417101044568](https://s2.loli.net/2024/04/17/8o4pLDAykOF6ZK3.png)

#### 根据⁡⁡‍⁤⁢⁣⁡‍⁤‍⁢⁤‌⁣⁡‬‍⁡‬‍⁣﻿⁤⁣⁣﻿⁡﻿⁣‬‌﻿﻿‬⁢⁡‌⁡‍‌‌‬[茴香豆零编程接入飞书教程](https://aicarrier.feishu.cn/docx/H1AddcFCioR1DaxJklWcLxTDnEc)创建飞书机器人，应用凭证、加密策略、事件配置、权限配置，配置完成后将机器人添加到飞书群聊中，将群名后缀加上对应的suffix

![image-20240417101721087](https://s2.loli.net/2024/04/17/IW8RQvEkNLGFpMS.png)

