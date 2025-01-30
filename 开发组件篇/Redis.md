# Redis
## 目录
* 数据类型篇
* 持久化篇
* 功能篇
* 高可用篇
* 缓存篇
* 面试篇


## 1 数据类型篇
Redis提供了丰富的数据类型，常见的有5种: <font color=blue>String、Hash、List、Set、ZSet</font>
随着 Redis 版本的更新，后面又支持了四种数据类型：<font color=blue>BitMap、HyperLoglog、GEO、Stream</font>
### String
String是最基本的key-value结构，key是唯一标识，value是具体的值，**value 最多可以容纳的数据长度是 512M。**
String 类型的底层的数据结构实现主要是 <font color=blue>int 和 SDS（简单动态字符串）</font>。
应用场景: \
其中包含4种：
缓存对象、常规计数、分布式锁、共享Session信息 \
>直接缓存整个对象的JSON
>采用将key进行分离为：user：ID：属性，采用MSET存储，用MGET获取各属性值，命令例子: SET user:1 '{"name'":"xiaolin"，"age":18}
> MSET user:1:name xiaolin user: 1: ago 18 user:2: name xiaomei user:2:age 20

因为Redis处理命令是单线程的，所以执行命令的过程是原子的，因此String类型适合计数场景，如计算访问次数、点赞、评论、库存数量等
SET 命令有个 NX 参数可以实现「key不存在才插入」，可以用它来实现分布式锁： \
如果 key 不存在，则显示插入成功，可以用来表示加锁成功； \
如果 key 存在，则会显示插入失败，可以用来表示加锁失败。 \
一般而言，还会对分布式锁加上过期时间





### List
List 列表是简单的字符串列表，按照插入顺序排序，可以从头部或尾部向 List 列表添加元素。\
列表的最大长度为 2^32 - 1，也即每个列表支持超过 40 亿个元素。
List 类型的底层数据结构:\
由<font color=blue>双向链表或压缩列表</font>实现的
但是在Redis3.2版本之后，List数据类型底层数据结构就只由quicklist实现了，替代了双向链表和压缩列表 \
**应用场景**：\
消息队列 \
应用与消息队列时的问题：\
1）如何满足消息的保序需求？
2）如何处理重复的消息？
3）如何保证消息的可靠性？






### Hash


### Set


### ZSet


### BitMap


### HyperLoglog


### GEO


### Stream






## 2 持久化篇




## 3 功能篇






## 4 高可用篇





## 5 缓存篇










## 6 面试突击篇
这部分内容主要是针对Redis的突击内容，大致包含了所需要的面试知识，比较全面，如需细节方面的了解需要从各个章节进行详细阅读。 \
这部分主要有8个部分： \
认识Redis、Redis数据结构、Redis线程模型、Redis持久化、Redis集群、Redis过期删除与内存淘汰、Redis缓存设计、Redis实战
### 6.1 认识Redis
1. 概念： \
Redis是一种基于内存的数据库，对数据的读写操作都是在内存中完成的，因此读写速度非常快，常用于<font color=blue>缓存、消息队列、分布式锁等场景</font>。 \
Redis提供了多种数据类型来支持不同的业务场景，比如基础5种数据类型(String(字符串)、Hash(哈希)、List(列表)、Set(集合)、ZSet(有序集合))、 Bitmaps(位图)、HyperLoglog(基数统计)、GEO(地理信息)、Stream(流)，并且对数据类型的操作都是原子性的，因为<font color=blue>执行命令由单线程负责</font>的，不存在并发竞争的问题。除此之外，Redis还支持事务、持久化、Lua脚本、多种集群方案(主从复制模式、哨兵模式、切片机群模式)、发布/订阅模式、内存淘汰机制、过期删除机制等等。

2. Redis与Memcached的区别 \
共同点: 
>都是基于内存的数据库，一般都用来当做缓存使用。 \
>都有过期策略。 \
>两者的性能都非常高。 

区别：
>* Redis支持的数据据类型更丰富(String、Hash、List、Set、ZSet)，而 Memcached 只支持最简单的 key-value 数据类型；
>* Redis支持数据的持久化、可以将内存种的数据保持在磁盘中，重启的时候可以再次加载进行使用，而Memcached没有持久化功能，数据全部存在内存之中，Memcached 重启或者挂掉后，数据就没了；
>* Redis原生支持集群模，，Memcached 没有原生的集群模式，需要依靠客户端来实现往集群中分片写入数据；
>* Redis支持发布订阅模型、Lua脚本、事务等功能，而Memcached不支持



3. 为什么Redis作为MySQL的缓存？\
主要是因为Redis具备【高性能】和【高并发】两种特性。
>* 用Redis将数据进行缓存，是直接操作内存，所以访问速度相当快。
>* 单台设备的Redis的QPS（Query Per Second，每秒钟处理完请求的次数）是MySQL的10倍，Redis单机的QPS能轻松破10W，而MySQL单机的QPS很难破1w。





