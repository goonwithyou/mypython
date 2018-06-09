# 1 introduction
---
该目录主要是介绍python中网络相关的一些知识(network),以及相关模块功能的介绍

# 2 contents
---
1. network_basic 主要记录了相关内容学习的一些笔记，主要是一些术语概念，能够对网络编程的一些基础内容有所了解
2. 相关功能模块，放在`mypython/modules/`文件下
    - socket: 基础网络接口，用于创建常用的TCP,UDP协议服务
    - select: IO多路复用的相关实现，能够简单实现多并发的效果。
    - selectors: 对select模块的功能进行了包装，也是IO多路复用最常使用的模块。