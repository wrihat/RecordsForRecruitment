# Object-C 学习笔记
## 1. 基本特性
    @符号的含义
    import
    @string
    NSLog
    BOOL
    nil值
    m、mm、cpp的文件扩展含义： 
        cpp是c语言扩展了c++，所以支持c++和c语言
        m是c语言扩展了object-c，所以支持object-c和c语言
        mm是object-c扩展了c++，所以支持object-c和c++和c语言
    @class 类前置声明


## 2. 面向对象特性
    {}花括号表示创建对象时的模板
    方法声明：
        + (void) setColor: (Color)color atIndex: (int)index;
        - (Color) getColor;
        最前面的+/-号的含义：+表示静态方法，-表示实例方法，只有这两种。
    方法调用：
        myColor setColor： red atIndex: 2\
        实例名 方法名：实参1 参数2描述：实参2


    get方法:
        object-c中，get方法一般将参数作为指针来返回值，即对于get方法一般传入指针来获取需要的值。对于自己设计的get方法一般也要遵循这个规则
    @interface接口
    @implement实现
    实例化
    访问当前对象自己： 使用self即可，访问父类对象使用super
    继承：object-c只能单一继承，C++才有多继承
    重写NSLog对象的输出信息：

```object-c 
- (NSString *) description
{
  return (@"I am a new NSLog output");
}
```


## 3. XCode
## 4. 框架
    Foundation 框架， 主要是NSXXX对象，比如NSString、NSArray、NSEnumerator、NSNumber等100多个类
    CoreFoundation 框架， CoreFoundation是纯C语言编写的，函数或变量以CF开头。
                Foundation框架是以CoreFoundation框架为基础创建的，几乎都有CoreFoundation的NS版本。  
    CoreGraphics框架， 用于处理集合图形，以CG开头的类。
    AppKit框架， AppKit是基于OS X平台（Mac），比如NSColor
    UIKit框架，UIKit是基于IOS平台（iPhone），比如UIColor


## 5. 一些有用的数据结构而不是对象
## 6. 字符串
    Objective-C支持C语言字符串方面的约定，也就是说单个字符被单引号包括，字符串被双引号包括。
    大多数框架把字符串传递给NSString对象。NSString类提供了字符串的类包装，包括对保存任意长度字符串的内建内存管理机制，也支持Unicode、printf风格的格式化工具。


    NSString和java中的String一样，是不可变的，只能重新构造。
    正如java中有StringBuffer一样，如果你需要可变字符串，则使用NSString的子类NSMutableString。
    NSString *height;
    height = [NSString stringWithFormat:@"Your height is %d feet, %d inches", 5, 11];

## 8. NSValue
## 9. NSNull
## 内存管理
## 异常
## 对象



    Objective-C的类规格说明包含了两个部分：定义和实现。类声明总是由@interface编译选项开始，
```Objective-C
@interface MyClass:NSObject: NSObject {
    int count;
    id  data;
    NSString* name;
}
- (id)initWithString:(NSString*) aName;
+ (MyClass*)createMyClassWithString:(NSString*)aName;
#end

NSString* anotherString = [NSString stringWithCString:"A C stirng" encoding:NSASCIIStringEncoding];

```






## 属性


### 消息传递
    Objective-C最大的特色是承自Smalltalk的消息传递模型, 此机制与C++主流风格差异甚大。Objective-C里, 与其说对象互相调用方法, 不如说对象之间互相传递消息更为精确。
    此二种风格的主要差异在于调用方法/消息传递这个动作。C++里类别与方法的关系严格清楚，一个方法必定属于一个类别，而且在编译时就已经紧密绑定，不可能调用一个不存在类别里的方法。
    在Objective-C里，类别与消息的关系比较松散，调用方法视为对对象发送消息，所有方法都被视为对消息的回应。所有消息处理直到运行时才会动态决定，并由类别自行决定如何处理收到的消息。
    也就是说，一个类别不保证一定会回应收到的消息。Objective-C天生即具备鸭子类型之绑定能力，因为运行期间才处理消息，允许发送未知消息给对象--》可以发送消息给整个对象集合而不需要一一检查对象的类型。




