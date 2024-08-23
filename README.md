# 品目分类器

## 一、配置环境

Python 版本`3.12.4`

1.创建虚拟环境

`python -m venv venv`

2.激活虚拟环境

Linux 命令: `source venv/bin/activate`

Windows 命令: `venv\bin\activate.bat`

3.安装依赖

`pip install -r requirements.txt`

## 二、使用 
1.启动 server

`python classifier_server.py`

2.预测关键词

`http://127.0.0.1:5000/predict?keyword=联想笔记本电脑`