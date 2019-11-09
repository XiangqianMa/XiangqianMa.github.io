---
title: Python中的函数
date: 2019-10-19 21:09:20
summary: 总结Python函数的相关知识点
categories:
- 编程基础
- 编程语言
tags:
- 编程基础
- Python
mathjax: true
img: /images/1547662972python2.png
---

# Python中的函数

![](Python中的函数/58154038eb26ff83d72f993821002b0f.jpg)

该博客主要记录Python中函数的一些高级用法，例如函数变量作用域、闭包、固定函数的部分参数等。该博客的大部分内容来自《Python Cookbook》。

## 函数变量作用域

Python函数中变量的作用域和其它语言类似。如果变量是在函数内部定义的，就称为局部变量，只在函数内部有效。一旦函数执行完毕，局部变量就会被回收，无法访问。

而全局变量是定义在整个文件层次上的，如果局部变量和全局或者外部变量同名，局部变量会覆盖全局或者外部变量。但这里要注意的一点是我们不能直接在函数中改变全局变量的值，如下述代码：

```python
MIN_VALUE = 1
MAX_VALUE = 10
def validation_check(value):
    ...
    MIN_VALUE += 1
    ...
validation_check(5)
```

会引发以下错误：

```python
UnboundLocalError: local variable 'MIN_VALUE' referenced before assignment
```

因为，Python的解释器会默认函数内部的变量为局部变量，但是局部变量尚未被声明，因而无法执行相关操作。所有，如果我们一定要在函数内部改变全局变量的值，必须加上`global`声明：

```python
# 变量作用域
MIN_VALUE = 1
def func():
    global MIN_VALUE 
    MIN_VALUE = 3
    print("Inner value: %d" % MIN_VALUE)

func()
print('Outer value: %d' % MIN_VALUE)

# 结果
Inner value: 3
Outer value: 3
```

如果我们在函数中重新声明一个同名的局部变量，那么在函数内部局部变量会覆盖全局变量，无论对内部局部变量进行何种操作都不会影响到外部的全局变量：

```python
# %%
# 变量作用域
MIN_VALUE = 1
def func():
    MIN_VALUE = 3
    print("Inner value: %d" % MIN_VALUE)

func()
print('Outer value: %d' % MIN_VALUE)

# 结果
Inner value: 3
Outer value: 1
```

类似，对于嵌套函数来说，内部函数是无法改变外部函数定义的变量的，如果要修改就需要加上`nonlocal`关键字：

```python
def outer():
    x = "local"
    def inner():
        nonlocal x # nonlocal 关键字表示这里的 x 就是外部函数 outer 定义的变量 x
        x = 'nonlocal'
        print("inner:", x)
    inner()
    print("outer:", x)
outer()
# 输出
inner: nonlocal
outer: nonlocal
```

如果没有`nonlocal`关键字，内部的变量会覆盖外部变量：

```python
def outer():
    x = "local"
    def inner():
        x = 'nonlocal' # 这里的 x 是 inner 这个函数的局部变量
        print("inner:", x)
    inner()
    print("outer:", x)
outer()
# 输出
inner: nonlocal
outer: local
```

总的来说，如果我们想要在函数作用域里面修改函数外部的变量的值，就必须使用相应的关键字进行提前声明。

## 闭包

该部分参考[Python之禅-一步一步教你认识Python闭包](https://foofish.net/python-closure.html)。

闭包的作用：使得局部变量在函数外被访问成为可能。闭包返回内部的嵌套函数。

> 在计算机科学中，闭包（Closure）是词法闭包（Lexical Closure）的简称，是引用了自由变量的函数。这个被引用的自由变量将和这个函数一同存在，即使已经离开了创造它的环境也不例外。所以，有另一种说法认为闭包是由函数和与其相关的引用环境组合而成的实体。

以下述代码为例：

```python
def print_msg():
    # print_msg 是外围函数
    msg = "zen of python"
    def printer():
        # printer 是嵌套函数
        print(msg)
    return printer

another = print_msg()
# 输出 zen of python
another()
```

这里的`another`就是一个闭包，闭包本质上是一个函数，由两部分组成：`printer`函数和变量`msg`。闭包的作用是使得变量始终被保存在内存中。

> 闭包，顾名思义，就是一个封闭的包裹，里面包裹着自由变量，就像在类里面定义的属性值一样，自由变量的可见范围随同包裹，哪里可以访问到这个包裹，哪里就可以访问到这个自由变量。

除此之外，如果一个函数被反复调用，且在这个函数的开始会调用一些类型检查、参数初始化的语句时，就可以使用闭包来实现。

## 使用函数替代只有单个方法的类

有时，我们需要在执行函数的过程中保存其中的一些状态变量。一种最简单的方法是定义一个类，使用类的属性保存变量，但这种做法未免过于冗余。为了简单起见，我们可以使用闭包技术将其转换为一个函数。

```python
# 使用类保存状态
from urllib.request import urlopen

class UrlTemplate:
    def __init__(self, template):
        self.template = template
    
    def open(self, **kwargs):
        return urlopen(self.template.format_map(kwargs))


yahoo = UrlTemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo.open(names='IBM,AAPL,FB', fields='sllclv'):
    print(line.decode('utf-8'))

    
# 使用闭包
def urltemplate(template):
    def opener(**kwargs):
        return urlopen(template.format_map(kwargs))
    return opener


yahoo = urltemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo(names='IBM,AAPL,FB', fields='sllclv'):
    print(line.decode('utf-8'))
```

相比于使用只有单个方法的类，使用闭包会更加简洁优雅。闭包的核心就是它可以记住定义闭包时的环境。无论何时，当在编写代码时遇到需要附加额外的状态给函数时，请考虑闭包。

## 在回调函数中携带额外的状态

在实际项目中，我们会编写许多需要回调函数的代码，有时我们需要在回调函数中保存额外的状态。与上一小节类似，我们可以考虑使用类来实现这一功能。

```python
# 使用类保存额外的状态
class ResultHandler:
    def __init__(self):
        self.sequence = 0
        
    def handler(self, result):
        self.sequence += 1
        print('[{}] Got: {}'.format(self.sequence, result))

        
def apply_async(func, args, *, callback):
    result = func(*args)
    # 调用回调函数
    callback(result)


def add(x, y):
    return x + y


r = ResultHandler()
apply_async(add, (2, 3), callback=r.handler)
apply_async(add, ('hello', 'world'), callback=r.handler)
```

运行结果如下：

```shell
[1] Got: 5
[2] Got: helloworld
```

同样，我们也可以使用闭包来捕获状态：

```python
# 使用闭包
def make_handler():
    sequence = 0
    def handler(result):
        # nonlocal声明用于表明变量sequence是在回调函数中修改的
        nonlocal sequence
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))
    
    return handler


handler = make_handler()
apply_async(add, (2, 3), callback=handler)
apply_async(add, ('hello', 'world'), callback=handler)
```

除此之外，还可以使用**协程**来携带状态：

```python
# 使用协程的方式在回调函数中携带状态
def apply_async(func, args, *, callback):
    result = func(*args)
    # 调用回调函数
    callback(result)

    
def add(x, y):
    return x + y


def make_handler():
    sequence = 0
    while(True):
        result = yield
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))
 

handler = make_handler()
next(handler)
apply_async(add, (2, 3), callback=handler.send)
apply_async(add, ('hello', 'world'), callback=handler.send)
```

最后，也可以使用额外的参数在回调函数中携带状态：

```python
from functools import partial


def apply_async(func, args, *, callback):
    result = func(*args)
    # 调用回调函数
    callback(result)

    
def add(x, y):
    return x + y


class SequenceNo:
    def __init__(self):
        self.sequence = 0

        
def handler(result, seq):
    seq.sequence += 1
    print('[{}] Got: {}'.format(seq.sequence, result))


seq = SequenceNo()
apply_async(add, (2, 3), callback=partial(handler, seq=seq))
apply_async(add, ('hello', 'world'), callback=partial(handler, seq=seq))
```

在上述代码中，因为要传入额外的参数来保存状态，因而回调函数会多出一个参数。同时，在调用回调函数时，只能传入一个参数，因而需要使用`partial`函数来解决这一问题。

## 访问定义在闭包内的变量

希望通过函数来扩展闭包，使得在闭包内层定义的变量可以被访问或修改。一般来说，在闭包内层定义的变量对于外界来说是完全隔离的，但是可以通过编写存取函数，并将它们作为函数属性附加到闭包上来提供对内层变量的访问机制。

```python
def sample():
    n = 0
    # Closure function
    def func():
        print('n=', n)

    # Accessor methods for n
    def get_n():
        return n

    def set_n(value):
        nonlocal n
        n = value

    # Attach as function attributes
    func.get_n = get_n
    func.set_n = set_n
    return func
```

`nolocal`关键字使得我们可以编写函数来修改内部变量的值；函数属性允许我们用一种简单的方式将方法绑定到闭包函数上。

## 参考

* Python-Cookbook