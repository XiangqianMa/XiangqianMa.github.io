---
title: Python的模块化编程
date: 2019-10-19 21:09:20
summary: 如何对Python程序进行模块化管理。
categories:
- 编程基础
- 编程语言
tags:
- 编程基础
- Python
mathjax: true
img: /images/1547662972python2.png
---

# Python模块化编程

## 包与模块的导入

### 包对应的from语句和import语句

import语句和包一起使用时，有些不方便，因为你必须经常在程序中重新输入路径。因此，让包使用from语句，来避免每次读取时都得重新输入路径，并且当目录树结构发生改变时，只需要在程序中更新一次路径即可。

实际中需要包导入的场合，就是解决当多个同名程序文件安装在同一个机器上时，所引发的模糊性。假设目录结构如下：

```shell
system1\

	utilities.py

	main.py

	other.py
```

与`system1`位于同一父目录下存在另一目录：

```shell
system2\

    utilities.py

	main.py

	other.py
```

Python总是先搜索主目录，也就是包含顶层文件的目录，例如，在`system1/main.py`中使用导入语句`import utilities`，导入会先搜索`system1`，只有在跨目录进行导入时才需要模块搜索路径的设置。

但是在第三个目录下导入另两个目录中的同名文件时就会发生错误，例如在第三个目录的程序文件中使用`import utilities`，便会产生模糊性，解释器会先搜索位于搜索路径左侧的`utilities.py`，这样做很容易出错，当然，我们可以使用`sys.path`对路径进行修改，但很容易出错。

**使用包便可以解决模块查找的模糊性**，不要在单独的目录内把文件安装成单纯的文件列表，而是将它们打包，在共同的根目录下，以子目录的方式进行组织，如下所示：

```shell
root\
	system1\
		__init__.py
		utilities.py
		main.py
		other.py
	system2\
		__init__.py
		utilities.py
		main.py
		other.py
    system3\
    	__init__py
    	myfile.py
```

在`myfile.py`中可以使用import语句进行导入：

```python
import system1.utilities
impott system2.utilities

system1.utilities.function('spam')
system2.utilities.function('eggs')
```

**注意**：当读取两个或两个以上路径内的同名属性时，必须使用 `import`，不能使用`from`。

### 包相对导入

上述导入方式都是针对从包的外部导入包文件而言的，在包自身的内部，包文件的导入可以使用和外部导入相同的路径语法。但是，也存在特殊的包内搜索规则来简化导入语句，也就是说，**包内的导入可能相对于包，而不是列出包导入路径**。

注意，包的相对导入机制与版本有关。Python2.6首先在导入上隐式地搜索包目录，而Python3.0需要显示地使用相对导入语法，这种变化使得相同的包的导入更为明显，从而增强代码的可读性。

#### Python3.0中的变化

引入了两个变化：

* 修改了模块导入搜索路径语义，默认跳过包自己的目录。导入只是检查搜索路径的其他组件，叫做“**绝对导入**”。
* 
* 扩展了`from`语句，允许显式地要求导入只搜索包的目录，叫做“**相对导入**”.

在python3.0和python2.6中，from语句可以使用前面的点号`.`来指定，导入位于同一包的模块，即相对导入，而不是位于模块导入路径上某处的模块，即绝对导入。

* 点号表示导入应该相对于外围的包-这样的导入将只在包的内部搜索，并且不会搜索位于导入搜索路径(sys.path)上某处的同名模块。即包模块覆盖了外部的模块。
* 在python2.6中，包的代码中的常规导入默认为先相对再绝对。而在Python3.0中在一个包中导入默认是绝对的。

使用两个点表示在文件所在的包的父目录的相对导入，如：

```python
from .. import spam
```

表示从与spam所在包的父目录开始进行相对导入，假设目录结构如下。

```
A\
|_ _ __init__.py
|_ _ B\
|	|_ _ myfile.py
|	|_ _ D\
|		 |_ _ X
|_ _ E\
	 |_ _X
	
```

> **注意：A目录下有`__init__.py`文件，也就是说A是一个包，如果没有`__init__.py`** ，则会出现以下错误：

```shell
ImportError: attempted relative import with no known parent package
```

该错误说明，解释器在从当前文件向上查找包时，超出了包的范围。

在该目录结构下，`myfile.py`有如下导入方式：

```python
from . import D # 导入A.B.D
from .. import E # 导入 A.E

from .D import X # 导入A.B.D.X
from ..E import X # 导入A.E.X
```

### 使用相对导入要注意以下几点

* 相对导入只适用于包内导入。
* 相对导入只适用于from语句。

### 可选导入

如果你希望优先使用某个模块或包，但是同时也想在没有这个模块或包的情况下有备选，你就可以使用可选导入这种方式。这样做可以导入支持某个软件的多种版本或者实现性能提升。以[github2包](http://pythonhosted.org/github2/_modules/github2/request.html)中的代码为例：

```python
try:
    # For Python 3
    from http.client import responses
except ImportError:  # For Python 2.5-2.7
    try:
        from httplib import responses  # NOQA
    except ImportError:  # For Python 2.4
        from BaseHTTPServer import BaseHTTPRequestHandler as _BHRH
        responses = dict([(k, v[0]) for k, v in _BHRH.responses.items()])
```

`lxml`包也有使用可选导入方式：

```python
try:
    from urlparse import urljoin
    from urllib2 import urlopen
except ImportError:
    # Python 3
    from urllib.parse import urljoin
    from urllib.request import urlopen
```

正如以上示例所示，**可选导入的使用很常见，是一个值得掌握的技巧**。

### 局部导入

当你在局部作用域中导入模块时，你执行的就是局部导入。如果你在Python脚本文件的顶部导入一个模块，那么你就是在将该模块导入至全局作用域，这意味着之后的任何函数或方法都可能访问该模块。例如：

```python
import sys  # global scope

def square_root(a):
    # This import is into the square_root functions local scope
    import math
    return math.sqrt(a)

def my_pow(base_num, power):
    return math.pow(base_num, power)

if __name__ == '__main__':
    print(square_root(49))
    print(my_pow(2, 3))
```

这里，我们将`sys`模块导入至全局作用域，但我们并没有使用这个模块。然后，在`square_root`函数中，我们将`math`模块导入至该函数的局部作用域，这意味着`math`模块只能在`square_root`函数内部使用。如果我们试图在`my_pow`函数中使用`math`，会引发`NameError`。

使用局部作用域的好处之一，是你使用的模块可能需要很长时间才能导入，如果是这样的话，将其放在某个不经常调用的函数中或许更加合理，而不是直接在全局作用域中导入。

但是，**根据约定，所有的导入语句都应该位于模块的顶部**。

### 导入注意事项

在导入模块方面，有几个程序员常犯的错误。这里我们介绍两个。

- 循环导入（circular imports）
- 覆盖导入（Shadowed imports，暂时翻译为覆盖导入）

先来看看循环导入。

### 循环导入

如果你创建两个模块，二者相互导入对方，那么就会出现循环导入。例如：

```python
a.py
import b

def a_test():
    print("in a_test")
    b.b_test()

a_test()
```

然后在同个文件夹中创建另一个模块，将其命名为`b.py`。

```Python
import a

def b_test():
    print('In test_b"')
    a.a_test()

b_test()
```

如果你运行任意一个模块，都会引发`AttributeError`。这是因为这两个模块都在试图导入对方。简单来说，模块`a`想要导入模块`b`，但是因为模块`b`也在试图导入模块`a`（这时正在执行），模块`a`将无法完成模块`b`的导入。一般来说，**修改方法是重构代码，避免发生这种情况**。

#### 覆盖导入

当你创建的模块与标准库中的模块同名时，如果你导入这个模块，就会出现覆盖导入。举个例子，创建一个名叫`math.py`的文件，在其中写入如下代码：

```python
import math

def square_root(number):
    return math.sqrt(number)

square_root(72)
```

试着运行这个文件，你会得到以下回溯信息（traceback）：

```shell
Traceback (most recent call last):
  File "math.py", line 1, in <module>
    import math
  File "/Users/michael/Desktop/math.py", line 6, in <module>
    square_root(72)
  File "/Users/michael/Desktop/math.py", line 4, in square_root
    return math.sqrt(number)
AttributeError: module 'math' has no attribute 'sqrt'
```

你运行这个文件的时候，Python解释器首先在当前运行脚本所处的的文件夹中查找名叫`math`的模块。在这个例子中，解释器找到了我们正在执行的模块，试图导入它。但是我们的模块中并没有叫`sqrt`的函数或属性，所以就抛出了`AttributeError`。

## 项目模块化

在Linux系统中，每一个文件都有一个绝对路径，以`\`开头，来表示从根目录到叶子结点的路径，这种方法叫做绝对路径。另外，对于任意两个文件，都存在从一个文件到另一个文件的路径，如：`../../Downloads/example.json`，该路径称为相对路径。

在大型工程中应该尽可能使用绝对位置，而非相对位置，对于一个独立的项目，所有的模块的追寻方式都最好从项目的根目录开始，称为相对的绝对路径。

例如，有一个项目的结构如下：

```shell

.
├── proto
│   ├── mat.py
├── utils
│   └── mat_mul.py
└── src
    └── main.py
```

各个文件中的代码如下：

```python

# proto/mat.py

class Matrix(object):
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self.m = len(data[0])
```

```python

# proto/mat.py

class Matrix(object):
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self.m = len(data[0])
```

```python

# utils/mat_mul.py

from proto.mat import Matrix

def mat_mul(matrix_1: Matrix, matrix_2: Matrix):
    assert matrix_1.m == matrix_2.n
    n, m, s = matrix_1.n, matrix_1.m, matrix_2.m
    result = [[0 for _ in range(n)] for _ in range(s)]
    for i in range(n):
        for j in range(s):
            for k in range(m):
                result[i][k] += matrix_1.data[i][j] * matrix_2.data[j][k]

    return Matrix(result)
```

```python

# src/main.py

from proto.mat import Matrix
from utils.mat_mul import mat_mul


a = Matrix([[1, 2], [3, 4]])
b = Matrix([[5, 6], [7, 8]])

print(mat_mul(a, b).data)

########## 输出 ##########

[[19, 22], [43, 50]]
```

观察上述代码，在`utils/mat_mul.py`文件中，导入`Matrix`的方式是从工程的目录开始导入`from proto.mat import Matrix`，而不是使用`..`从上一级目录导入。

在`Pycharm`中，上述代码可以被成功运行，但是如果在命令行中，无论是进入`src`文件夹输入`python main.py`还是退回上一级目录，输入`python src/main.py`，都会出现找不到包`proto`的错误。

实际上，正如上文中所示，Python解释器在导入模块时，会在一个特定的列表中查找模块，如下：

```python

import sys  

print(sys.path)

########## 输出 ##########

['', '/usr/lib/python36.zip', '/usr/lib/python3.6', '/usr/lib/python3.6/lib-dynload', '/usr/local/lib/python3.6/dist-packages', '/usr/lib/python3/dist-packages']
```

对于Pycharm来说，当它运行程序时，会首先将上述列表的第一项设置为项目的根目录。因而，无论如何运行`main.py`，导入模块时都会首先从项目的根目录中寻找对应的包和模块。

为了在命令函中也能达到无论如何运行`main.py`也能正确找到包和模块的目的，有以下两种方法：

* 直接对上述列表的第一个位置进行修改。但这样就在代码中写入了绝对路径，不推荐。

* 修改`PYTHONHOME`。Python存在一个虚拟运行环境，提倡每一个项目最好都有一个对立的运行环境来保持包和模块的纯洁性。

  可以直接在Virtual Environment中的activate文件中加入：

  ```shell
  export PYTHONPATH="/home/ubuntu/workspace/your_projects"
  ```

## if __name__ == '__main__'

C++、Java等语言需要显式提供入口函数`main()`，但Python不用。那么，`if __name__ == __main__`的作用是什么？

有项目结构如下：

```shell
.
├── utils.py
├── utils_with_main.py
├── main.py
└── main_2.py
```

```python
# utils.py

def get_sum(a, b):
    return a + b

print('testing')
print('{} + {} = {}'.format(1, 2, get_sum(1, 2)))
```

```python
# utils_with_main.py

def get_sum(a, b):
    return a + b

if __name__ == '__main__':
    print('testing')
    print('{} + {} = {}'.format(1, 2, get_sum(1, 2)))
```

```python
# main.py

from utils import get_sum

print('get_sum: ', get_sum(1, 2))

########## 输出 ##########

testing
1 + 2 = 3
get_sum: 3
```

```python
# main_2.py

from utils_with_main import get_sum

print('get_sum: ', get_sum(1, 2))

########## 输出 ##########

get_sum_2: 3
```

`import`在导入模块时会自动将文件中的暴露代码执行一遍，对于模块的测试代码，如果我们不想在导入模块的时候执行这些代码，就要将这些代码放在`if __name__ == __main__`之下。