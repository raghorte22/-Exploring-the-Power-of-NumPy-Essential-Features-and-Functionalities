#!/usr/bin/env python
# coding: utf-8

# ### NUPPY ARRAYS

# In[1]:


#NumPy is a popular Python library for numerical computing that provides support for arrays, matrices, and mathematical functions. 


# In[2]:


#creating numpy/n-d arrays 

import numpy as np


# In[3]:


arr1= np.array([1,2,3,4,5])
arr1


# In[4]:


type(arr1)


# In[5]:


arr2=np.array([[1,2,3,],[2,3,4]])
arr2


# In[6]:


arr3 =np.zeros((2,3))
arr3


# In[7]:


arr4 = np.ones((3,3))
arr4


# In[8]:


arr5 =np.identity(5)
arr5


# In[9]:


arr6 = np.arange(10)
arr6


# In[10]:


arr7 = np.arange(5,16)
arr7


# In[11]:


type(arr7)


# In[12]:


arr7.shape


# In[13]:


arr8 = np.linspace(10,20,10)
arr8


# In[14]:


arr9 = arr8.copy()
arr9


# In[15]:


arr10 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]]) #3 dimensional matrix
arr10


# In[16]:


arr10.shape


# In[17]:


arr10.ndim


# In[18]:


arr2


# In[19]:


arr2.ndim


# In[20]:


arr1


# In[21]:


arr1.ndim


# In[22]:


#size
arr1.size #numeber of item


# In[23]:


arr10.size


# In[24]:


arr10.itemsize #4 bytes memory occupy


# In[25]:


arr9.itemsize #because it is float


# In[26]:


arr9.dtype


# In[27]:


arr10.dtype


# In[28]:


arr10.astype('float')


# In[29]:


arr9.astype('int')


# In[30]:


# numpy contain less memory

lista=range(100)
arr11=np.arange(100)


# In[31]:


import sys


# In[32]:


print(sys.getsizeof(87)*len(lista))


# In[33]:


print(arr11.itemsize*arr11.size)


# In[34]:


# numpy arrays as faster in comparision to list

import time


# In[35]:


x=range(10000000)
y=range(10000000,20000000)

start_time =time.time()

c=[(x,y) for x,y in zip(x,y)]

print(time.time()-start_time)


# In[36]:


a =np.arange(10000000)
b = np.arange(10000000,20000000)


start_time =time.time()

c=a+b

print(time.time() - start_time)




# In[37]:


# indexing slicing and iteration

arr12=np.arange(24)
arr12


# In[38]:


arr12 = np.arange(24).reshape(6,4)  #reshape function is used for change the shape of function
arr12


# In[39]:


arr1


# In[40]:


arr1[3]


# In[41]:


arr1[2:4]


# In[42]:


arr1[-1]


# In[43]:


arr12


# In[44]:


arr12[2]


# In[45]:


arr12[-1]


# In[46]:


arr12[:2]


# In[47]:


arr12[:,1:2]


# In[48]:


arr12[:,1:3]


# In[49]:


arr12[2:4 ,1:3]


# In[50]:


# iteration

arr12


# In[51]:


for i in arr12:
    print(i)


# In[52]:


for i in np.nditer(arr12):
    print(i)


# In[53]:


#Creating Arrays:

#numpy.array(): Create an array from a list or tuple.
#numpy.arange(): Create an array with evenly spaced values within a given interval.
#numpy.zeros(): Create an array filled with zeros.
#numpy.ones(): Create an array filled with ones.
#numpy.random.rand(): Create an array of random values from a uniform distribution.


# In[54]:


#import numpy as np


# In[55]:


arr1=np.array([1,2,3,4,5,6])
arr2=np.array([4,5,6,7,8,9])


# In[56]:


arr1-arr2


# In[57]:


arr1*arr2


# In[58]:


arr1*2


# In[59]:


arr2*3


# In[60]:


arr2>3


# In[61]:


arr3=np.arange(6).reshape(2,3)
arr4=np.arange(6,12).reshape(3,2)


# In[62]:


arr3.dot(arr4)


# In[63]:


arr1.dot(arr2)


# In[64]:


arr4


# In[65]:


arr4.max()


# In[66]:


arr4.min()


# In[67]:


arr4


# In[68]:


arr4.min(axis=0)


# In[69]:


arr4.max(axis=0)


# In[70]:


arr4.min(axis=1)


# In[71]:


arr4.max(axis=1)


# In[72]:


arr4.sum()


# In[73]:


arr4.sum(axis=0)


# In[74]:


arr4.mean()


# In[75]:


arr4.std()


# In[76]:


np.sin(arr4)


# In[77]:


np.median(arr4)


# In[78]:


np.exp(arr4)         #expoliate


# In[79]:


#reshaping numpy array

arr4


# In[80]:


arr4.ndim


# In[81]:


arr4.ravel()


# In[82]:


#transpose

arr4


# In[83]:


arr4.transpose()      #row is converted to column ,column converted to row


# In[84]:


#stacking

#stacking is stage that we combine two arrays

arr3


# In[85]:


arr5=np.arange(12,18).reshape(2,3)


# In[86]:


arr5


# In[87]:


np.hstack((arr3,arr5))       #horizontal stacking


# In[88]:


np.vstack((arr3,arr5))        #vertical stacking


# In[89]:


#spliting

np.hsplit(arr3,3)           #horizontal split


# In[90]:


np.vsplit(arr3,2)          #vertical split


# In[91]:


#slicing#indexing

arr8=np.arange(24).reshape(6,4)


# In[92]:


arr8


# In[93]:


arr8[[0,2,4]]


# In[94]:


arr8[[-1,-3,-4]]


# In[95]:


arr =np.random.randint(low=1,high=100,size=20).reshape(4,5)
arr


# In[96]:


arr[0]


# In[97]:


arr[3]


# In[98]:


arr>50


# In[99]:


#indexing using boolean array
arr[arr>50]


# In[100]:


arr[(arr>50) &(arr%2!=0)]


# In[101]:


arr[(arr>50) & (arr%2!=0)]=0


# In[102]:


arr


# In[103]:


#ploting graph using numpy

x=np.linspace(-40,40,100)


# In[104]:


x


# In[105]:


x.size


# In[106]:


y=np.sin(x)


# In[107]:


y


# In[108]:


y.size


# In[109]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[110]:


plt.plot(x,y)


# In[111]:


y=x*x+2*x+6


# In[112]:


plt.plot(x,y)


# In[113]:


# brodcasting

# it allows arrays of different shapes to be combined together during arithmetic operations. 
#usually done on corresponding elements
#if two arrays are of exactly the same shape.then these operations are smoomthly performed.
#if dimensions of two arrays are dissimilar , then the element-to-element operation are not possible.
#however operations on arrays of non-similar shape is still possible in numpy,because of the broaadcasting capabillity.
# the smaller array is broadcast to the size of the larger array so that they have compatable shape.


#senario 1

a1=np.arange(8).reshape(2,4)
a2=np.arange(8,16).reshape(2,4)

print(a1)
print(a2)


# In[114]:


a1+a2


# In[115]:


#senario 2

a3=np.arange(9).reshape(3,3)
a4=np.arange(3).reshape(1,3)

print(a3,a4)


# In[116]:


a3+a4


# In[117]:


#rules for broadcasting

#if x=m and y=n operation will take place

a1=np.arange(8).reshape(2,4)
a2=np.arange(8,16).reshape(2,4)

a1+a2


# In[118]:


# if x=1 and y=n then also operation will take place(same dimension)

a5 = np.arange(3).reshape(1,3)
a6 = np.arange(12).reshape(4,3)

print(a5)
print(a6)


# In[119]:


a5+a6


# In[120]:


# if y=1 and x=m then also operation will take place , even if 
#they are not of the same dimension

a7=np.arange(4).reshape(4,1)
a8=np.arange(12).reshape(4,3)

print(a7)
print(a8)


# In[121]:


a7+a8


# In[122]:


# if x=1 and y!=n then also operation will not take place 

a9=np.arange(3).reshape(1,3)
a10=np.arange(16).reshape(4,4)

print(a9)
print(a10)


# In[123]:


a9 + a10                 # value error 
                         #operands could not be broadcast together with shapes (1,3) (4,4) 


# In[124]:


#if x=1 and n=1 then y==m , operation to take place

a11=np.arange(3).reshape(1,3)
a12=np.arange(3).reshape(3,1)

print(a11)
print(a12)


# In[125]:


a11+a12


# In[126]:


# if x=1 and y=1 then the operation will take place no matter what

a13 = np.arange(1).reshape(1,1)
a14 = np.arange(20).reshape(4,5)

print(a13)
print(a14)


# In[127]:


a13+a14


# In[128]:


# if they are of different dimensions

a15 =np.arange(4)
a16 =np.arange(20).reshape(5,4)

print(a15)
print(a16)


# In[129]:


a15+a16


# In[131]:


# various functions in numpy 

np.random.random()


# In[134]:


np.random.seed(1)     # if we use seed function i getting same random value again and again
np.random.random()


# In[140]:


np.random.uniform(3,10)


# In[141]:


np.random.uniform(1,100,10)


# In[144]:


np.random.randint(1,10)


# In[148]:


np.random.randint(1,10,15).reshape(3,5)


# In[152]:


a=np.random.randint(1,10,6)
a


# In[153]:


np.max(a)


# In[154]:


a[np.argmax(a)]


# In[155]:


a[np.argmin(a)]


# In[156]:


np.argmin(2)


# In[157]:


a=np.random.randint(1,10,6)
a


# In[159]:


a[a%2==1]=-1
a


# In[162]:


a=np.random.randint(1,50,6)
a


# In[163]:


np.where(a%2==1,-1,a)


# In[164]:


a


# In[165]:


out=np.where(a%2==1,-1,a)


# In[166]:


out


# In[168]:


a=np.random.randint(1,50,10)
a


# In[169]:


a=np.sort(a)
a


# In[170]:


np.percentile(a,25)


# In[171]:


np.percentile(a,50)


# In[173]:


np.percentile(a,99.8)


# In[ ]:




