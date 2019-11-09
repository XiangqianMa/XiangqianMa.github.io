---
title: Python面向对象基础
date: 2019-10-24 21:09:20
summary: Python面向对象基础
categories:
- 编程基础
- 编程语言
tags:
- 编程基础
- Python
mathjax: true
img: /images/1547662972python2.png
---

# Python面向对象

这篇博客主要介绍Python中面向对象的一些比较高级的用法。

## 面向对象编程的四要素

> 类、属性、函数、对象

那么，类就是一群具有相同属性和函数的对象的集合。

## 抽象类和抽象函数

假设有以下代码，有一个父类和两个子类。

```python
class Entity():
    def __init__(self, object_type):
        print('parent class init called')
        self.object_type = object_type
    
    def get_context_length(self):
        raise Exception('get_context_length not implemented')
    
    def print_title(self):
        print(self.title)

class Document(Entity):
    def __init__(self, title, author, context):
        print('Document class init called')
        Entity.__init__(self, 'document')
        self.title = title
        self.author = author
        self.__context = context
    
    def get_context_length(self):
        return len(self.__context)
    
class Video(Entity):
    def __init__(self, title, author, video_length):
        print('Video class init called')
        Entity.__init__(self, 'video')
        self.title = title
        self.author = author
        self.__video_length = video_length
    
    def get_context_length(self):
        return self.__video_length

harry_potter_book = Document('Harry Potter(Book)', 'J. K. Rowling', '... Forever Do not believe any thing is capable of thinking independently ...')
harry_potter_movie = Video('Harry Potter(Movie)', 'J. K. Rowling', 120)

print(harry_potter_book.object_type)
print(harry_potter_movie.object_type)

harry_potter_book.print_title()
harry_potter_movie.print_title()

print(harry_potter_book.get_context_length())
print(harry_potter_movie.get_context_length())

########## 输出 ##########

Document class init called
parent class init called
Video class init called
parent class init called
document
video
Harry Potter(Book)
Harry Potter(Movie)
77
120
```

在上述代码中的`Entity`类本身没有什么作用，只是提供了一些`Document`和`Video`的基本元素。因而，我们不需要生成`Entity`的对象。那么，如何防止生成`Entity`的对象呢？

这里需要引入**抽象类**和**抽象函数**的概念：

> 所谓抽象类是一种特殊的类，该类的作用就是作为父类存在的，一旦对其进行对象化就会产生错误。同样，抽象函数定义在抽象类中，子类必须重写该函数才能被使用。抽象函数使用装饰器`@abstractmethod`来表示。

如下代码：

```python
from abc import ABCMeta, abstractmethod

class Entity(metaclass=ABCMeta):
    @abstractmethod
    def get_title(self):
        pass

    @abstractmethod
    def set_title(self, title):
        pass

class Document(Entity):
    def get_title(self):
        return self.title
    
    def set_title(self, title):
        self.title = title

document = Document()
document.set_title('Harry Potter')
print(document.get_title())

entity = Entity()

########## 输出 ##########

Harry Potter

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-7-266b2aa47bad> in <module>()
     21 print(document.get_title())
     22 
---> 23 entity = Entity()
     24 entity.set_title('Test')

TypeError: Can't instantiate abstract class Entity with abstract methods get_title, set_title
```

在上述代码中，我们直接声明了抽象类`Entity`的对象，引发了类型错误。我们必须使用子类对其进行继承才能正常使用。

抽象类的作用就是定义接口，是一种自上而下的设计方法。只需要使用少量的代码对需要做的事情进行描述，定义好接口，然后分发给不同的开发人员进行开发和对接。