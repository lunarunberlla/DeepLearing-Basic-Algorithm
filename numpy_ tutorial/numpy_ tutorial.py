
import numpy as np

# #### 2.建立一个一维数组 a 初始化为[4,5,6], (1)输出a 的类型（type）(2)输出a的各维度的大小（shape）(3)输出 a的第一个元素（值为4）

# In[4]:

# 初始化数组
a = np.array([4, 5, 6])

# 输出a的类型
print(type(a))

# 输出a的各维度的大小
print(a.shape)

# 输出a的第一个元素
print(a[0])

# #### 3.建立一个二维数组 b,初始化为 [ [4, 5, 6],[1, 2, 3]] (1)输出各维度的大小（shape）(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）

# In[5]:

# 初始化二维数组
b = np.array([[4, 5, 6], [1, 2, 3]])

# 输出b的各维度的大小
print(b.shape)

# 输出b(0,0)，b(0,1),b(1,1) 这三个元素
print(b[0, 0], b[0, 1], b[1, 1])

# #### 4.  (1)建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）(2)建立一个全1矩阵b,大小为4x5;  (3)建立一个单位矩阵c ,大小为4x4; (4)生成一个随机数矩阵d,大小为 3x2.

# In[6]:

# (1) 创建全0矩阵 a，大小为3x3，类型为整型
a = np.zeros((3, 3), dtype=int)
print("矩阵 a:\n", a)

# (2) 创建全1矩阵 b，大小为4x5
b = np.ones((4, 5))
print("矩阵 b:\n", b)

# (3) 创建单位矩阵 c，大小为4x4
c = np.eye(4)
print("矩阵 c:\n", c)

# (4) 创建随机数矩阵 d，大小为3x2
d = np.random.random((3, 2))
print("矩阵 d:\n", d)

# #### 5. 建立一个数组 a,(值为[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] ) ,(1)打印a; (2)输出  下标为(2,3),(0,0) 这两个数组元素的值

# In[7]:

# 创建数组 a
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 打印数组 a
print("数组 a:\n", a)

# 输出指定位置元素的值
print("位置 (2, 3) 的元素值:", a[2, 3])
print("位置 (0, 0) 的元素值:", a[0, 0])


# #### 6.把上一题的 a数组的 0到1行 2到3列，放到b里面去，（此处不需要从新建立a,直接调用即可）(1),输出b;(2) 输出b 的（0,0）这个元素的值

# In[8]:

# 创建数组 a
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 提取子数组赋值给数组 b
b = a[0:2, 2:4]

# 输出数组 b
print("数组 b:\n", b)

# 输出数组 b 中 (0, 0) 元素的值
print("数组 b 中 (0, 0) 元素的值:", b[0, 0])


#  #### 7. 把第5题中数组a的最后两行所有元素放到 c中，（提示： a[1:2, :]）(1)输出 c ; (2) 输出 c 中第一行的最后一个元素（提示，使用 -1                 表示最后一个元素）

# In[9]:

# 创建数组 a
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 提取最后两行的所有元素赋值给数组 c
c = a[1:3, :]

# 输出数组 c
print("数组 c:\n", c)

# 输出数组 c 中第一行的最后一个元素
print("数组 c 中第一行的最后一个元素:", c[0, -1])

# #### 8.建立数组a,初始化a为[[1, 2], [3, 4], [5, 6]]，输出 （0,0）（1,1）（2,0）这三个元素（提示： 使用 print(a[[0, 1, 2], [0, 1, 0]]) ）

# In[11]:

import numpy as np

# 初始化数组 a
a = np.array([[1, 2], [3, 4], [5, 6]])

# 输出指定位置的元素
indices = [[0, 1, 2], [0, 1, 0]]
selected_elements = a[indices]
print("输出指定位置的元素:", selected_elements)


# #### 9.建立矩阵a ,初始化为[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，输出(0,0),(1,2),(2,0),(3,1) (提示使用 b = np.array([0, 2, 0, 1])                     print(a[np.arange(4), b]))

# In[12]:

import numpy as np

# 初始化矩阵 a
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# 输出指定位置的元素
b = np.array([0, 2, 0, 1])
selected_elements = a[np.arange(4), b]
print("输出指定位置的元素:", selected_elements)


# #### 10.对9 中输出的那四个元素，每个都加上10，然后重新输出矩阵a.(提示： a[np.arange(4), b] += 10 ）

# In[13]:

# 初始化矩阵 a
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# 将指定位置的元素加上 10
b = np.array([0, 2, 0, 1])
a[np.arange(4), b] += 10

# 输出更新后的矩阵 a
print("更新后的矩阵 a:\n", a)


# ### array 的数学运算

# #### 11.  执行 x = np.array([1, 2])，然后输出 x 的数据类型

# In[14]:

x = np.array([1, 2])
print("x 的数据类型:", x.dtype)


# #### 12.执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型

# In[15]:

x = np.array([1.0, 2.0])
print("x 的数据类型:", x.dtype)


# #### 13.执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)

# In[16]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

result1 = x + y
result2 = np.add(x, y)

print("x + y:\n", result1)
print("np.add(x, y):\n", result2)


# #### 14. 利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)

# In[17]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

result1 = x - y
result2 = np.subtract(x, y)

print("x - y:\n", result1)
print("np.subtract(x, y):\n", result2)

# #### 15. 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 还有  np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。

# In[18]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

result1 = x * y
result2 = np.multiply(x, y)
result3 = np.dot(x, y)

print("x * y:\n", result1)
print("np.multiply(x, y):\n", result2)
print("np.dot(x, y):\n", result3)
###矩阵乘法运算要求两个矩阵的形状匹配，即前一个矩阵的列数等于后一个矩阵的行数。

###如果使用非方阵的矩阵，请确保满足矩阵乘法的形状要求，否则会引发维度不匹配的错误。

# 16. 利用13题目中的x,y,输出 x / y .(提示 ： 使用函数 np.divide())

# In[19]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

result = np.divide(x, y)

print("x / y:\n", result)

# #### 17. 利用13题目中的x,输出 x的 开方。(提示： 使用函数 np.sqrt() )

# In[20]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)

result = np.sqrt(x)

print("数组 x 的开方:\n", result)

# #### 18.利用13题目中的x,y ,执行 print(x.dot(y)) 和 print(np.dot(x,y))

# In[21]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

result1 = x.dot(y)
result2 = np.dot(x, y)

print("x.dot(y):\n", result1)
print("np.dot(x, y):\n", result2)

# ##### 19.利用13题目中的 x,进行求和。提示：输出三种求和 (1)print(np.sum(x)):   (2)print(np.sum(x，axis =0 ));   (3)print(np.sum(x,axis = 1))

# In[22]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)

sum_x = np.sum(x)

print("数组 x 的求和结果:", sum_x)

# #### 20.利用13题目中的 x,进行求平均数（提示：输出三种平均数(1)print(np.mean(x)) (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）

# In[23]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)

mean_x = np.mean(x)

print("数组 x 的平均值:", mean_x)

# #### 21.利用13题目中的x，对x 进行矩阵转置，然后输出转置后的结果，（提示： x.T 表示对 x 的转置）

# In[24]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)

transpose_x = np.transpose(x)
# 或者使用 transpose_x = x.T

print("转置后的数组 x:\n", transpose_x)

# #### 22.利用13题目中的x,求e的指数（提示： 函数 np.exp()）

# In[25]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)

exp_x = np.exp(x)

print("e 的指数结果:\n", exp_x)

# #### 23.利用13题目中的 x,求值最大的下标（提示(1)print(np.argmax(x)) ,(2) print(np.argmax(x, axis =0))(3)print(np.argmax(x),axis =1))

# In[26]:

x = np.array([[1, 2], [3, 4]], dtype=np.float64)

max_index = np.argmax(x)
max_index_axis0 = np.argmax(x, axis=0)
max_index_axis1 = np.argmax(x, axis=1)

print("整个数组中最大值的下标:", max_index)
print("沿 axis=0 方向最大值的下标:", max_index_axis0)
print("沿 axis=1 方向最大值的下标:", max_index_axis1)

# #### 24,画图，y=x*x 其中 x = np.arange(0, 100, 0.1) （提示这里用到  matplotlib.pyplot 库）
# In[27]:

import matplotlib.pyplot as plt

x = np.arange(0, 100, 0.1)
y = x**2

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = x^2')
plt.show()

# #### 25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)(提示：这里用到 np.sin() np.cos() 函数和 matplotlib.pyplot 库)

# In[28]:
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin, label='sin(x)')
plt.plot(x, y_cos, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of sin(x) and cos(x)')
plt.legend()
plt.show()
# In[ ]:




