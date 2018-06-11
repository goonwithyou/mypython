# 1 introduction
---
该目录主要是介绍python中网络相关的一些知识(network),以及相关模块功能的介绍

# 2 contents
---
1. network_basic_1:主要记录了相关内容学习的一些笔记，主要是一些术语概念，能够对网络编程的一些基础内容有所了解
    - 1 OSI七层模型，五层，四层
    - 2 协议
    - 3 IP
    - 4 相关网络术语概念
    - 5 传输层服务
    - 6 socket
    - 7 other
    - 8 http
    - 9 IO
    - 10 local socket
    - 11 多任务
    
2. network_basic_2
    - 多进程
2. 相关功能模块，放在`mypython/modules/`文件下
    - socket: 基础网络接口，用于创建常用的TCP,UDP协议服务
    - select: IO多路复用的相关实现，能够简单实现多并发的效果。
    - selectors: 对select模块的功能进行了包装，也是IO多路复用最常使用的模块。