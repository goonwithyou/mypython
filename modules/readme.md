# 1 introduction
---
该文件夹主要存放python的一些标准库功能介绍，以及一些常用的第三方库，每个模块都有一个文件夹，内部有相应的代码实现。

# 2 contents
---
1. 基础模块
    - re: 正则表达式操作
    - collections:对dict,list,set,tuple的一些快捷操作，如链式字典，计数器
1. 函数式编程
    - itertools: 创建迭代器以及相关迭代操作
1. network相关模块(建议使用\*nix系统)
    - socket: 创建网络服务接口
    - select: IO多路复用基础模块
    - selectors: IO多路复用高级模块（常用）(ing)
    - collections: 集合的相关应用（ing）
    - multiprocessing(ing)
    - threading thread模块的内容和multiprocessing模块的使用方法基本类似，可以先查看multiprocessing模块的内容，这里的threading模块没有详细介绍。(ing)
    - signal : 常用于进程间的信号传递，
    
2. server
    - httpserver : (ing)
    - socketserver : 用于快捷创建tcp,udp,unix,多线程，多进程的server.
3. 文件和文件夹
    - pathlib : 文件、文件夹操作
3. 协程
    - greenlet:(ing)
    - gevent:(ing)