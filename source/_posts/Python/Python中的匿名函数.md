---
title: Python中的匿名函数
date: 2019-10-24 12:22:20
summary: lambda函数
categories:
- 编程基础
- 编程语言
tags:
- 编程基础
- Python
mathjax: true
img: /images/1547662972python2.png
---

# 匿名函数

## 什么是匿名函数

匿名函数的格式如下：

```python
lambda argument1, argument2,... argumentN : expression
```

匿名函数的关键字为`lambda`，用法如下：

```python
square = lambda x: x**2
square(3)

9
```

其对应的常规函数形式为：

```python
def square(x):
    return x**2
square(3)
 
9
```

匿名函数和常规函数一样，返回的都是函数对象（function object）,不同之处有以下几点：

* `lambda`是一个表达式，而不是一个语句

  所谓表达式就是用一系列“公式”去表达一个东西，如x+2、x**2等；所谓语句就是完成了某些功能，如果赋值语句完成赋值，比较语句完成比较等。

  `lambda`表达式可以用在一些常规函数不能使用的地方，如：列表内部：

  ```python
  [(lambda x: x*x)(x) for x in range(10)]
  # 输出
  [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
  
  ```

  被用作某些函数的参数，而常规函数不能：

  ```python
  l = [(1, 20), (3, 0), (9, 10), (2, -1)]
  l.sort(key=lambda x: x[1]) # 按列表中元祖的第二个元素排序
  print(l)
  # 输出
  [(2, -1), (3, 0), (9, 10), (1, 20)]
  ```

  常规函数必须通过其函数名被调用，因而必须首先被定义。但是`lambda`是一个表达式，返回的函数对象不需要名字。

* `lambda`表达式是只有一行的简单表达式，并不能扩展成一个多行的代码块。

  Python发明`lambda`表达式的目的就是让它处理一些简单的任务，较为复杂的任务则由常规函数处理。

## 为什么使用匿名函数

在一些情况下，使用匿名函数`lambda`可以帮助我们降低代码的复杂度，提高代码的可读性。

我们使用函数的目的有以下几点：

* 减少代码的重复性
* 模块化代码

但当我们只需要一个函数，这个函数非常简短，只需要一行；同时在程序中只调用一次，此时，我们就不需要给它一个定义和名字。

例如，需要对一个列表中的所有元素求平方和，且该程序只需要运行一次，用匿名函数可以写成：

```python
squared = map(lambda x: x**2, [1, 2, 3, 4, 5])
```

用常规函数需要写成：

```python
def square(x):
    return x**2

squared = map(square, [1, 2, 3, 4, 5])
```

很明显，匿名函数更为简洁。

## Python函数式编程

所谓函数式编程，是指代码中每一块都是不可变的，都由纯函数组成。纯函数是指函数本身相互独立、互不影响，对于相同的输入，总会有相同的输出，没有任何副作用。

例如，将一个列表中的元素值都变成原来的两倍，如下：

```python
def multiply_2(l):
    for index in range(0, len(l)):
        l[index] *= 2
    return l
```

上述代码不是一个纯函数，因为输入列表的值发生了改变，多次调用该函数将会得到不同的值。修改成如下形式：

```python
def multiply_2_pure(l):
    new_list = []
    for item in l:
        new_list.append(item * 2)
    return new_list
```

新建列表并返回，这就是一个纯函数。

函数式编程的优点是其纯函数和不可变的特性使得程序更加健壮，易于调试和测试；缺点是限制多、程序编写难度高。Python不是一门函数式编程语言，但是提供了相关特性，主要有以下函数`map()`，`filter()`，`reduce()`，这几个函数通常和匿名函数一起使用。

* `map()`

  `map(function, iterable)`对`iterable`中的每个元素都调用`function`函数，最后返回一个新的可遍历的集合。`map()`函数直接由C语言写成，运行时不需要Python解释器间接调用，性能高。

* `filter(function, iterable)`

  `filter()`函数对`iterable`中的每个元素都使用`function`进行判断，返回`True`或`False`，将`True`对应的元素组成一个新的可遍历的集合。

  ```python
  l = [1, 2, 3, 4, 5]
  new_list = filter(lambda x: x % 2 == 0, l) # [2, 4]
  ```

* `reduce(function, iterable)`

  该函数通常用来对一个集合做一些累积操作。`function()`函数同样是一个函数对象，它有两个函数，表示对`iterable`中的每个元素以及上一次调用后的结构运用`function`进行计算，最后返回的是一个单独的值。

  例如，计算某个列表元素的乘积：

  ```python
  l = [1, 2, 3, 4, 5]
  product = reduce(lambda x, y: x * y, l) # 1*2*3*4*5 = 120
  ```

通常来说，当我们需要对集合中的元素进行一些操作时，如果操作非常简单，则有限考虑`map()`、`filter()`、`reduce()`这类或者列表表达式的形式。那么如何在这两种方式中进行选择？

* 当数据量非常多时，例如机器学习的应用，倾向于使用函数式编程的表示，效率更高；
* 当数据量不多时，为了使得程序更加的Pythonic，可以使用列表表达式。

当操作比较复杂时，使用`for`循环。

## 练习

对一个字典根据其值从高到低进行排列。

```python
# 匿名函数
d = {'mike': 10, 'lucy': 2, 'ben': 30}
e = sorted(d.items(), key=lambda x: x[1], reverse=True)
e

Out[15]: [('ben', 30), ('mike', 10), ('lucy', 2)]
```

## 参考

* [Python核心技术与实战-景霄](https://time.geekbang.org/column/article/98411)