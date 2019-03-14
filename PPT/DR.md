---
typora-copy-images-to: image
typora-root-url: image
---

<center><h2>DR</h2>



http://jeffreydf.github.io/diabetic-retinopathy-detection/

https://www.kaggle.com/kmader/inceptionv3-for-retinopathy-gpu-hr

Make a nice retinopathy model by using a pretained inception v3 as a base and retraining some modified final layers with attention.

- high-resolution images

- better data sampling

- ensuring there is no leaking between training and validation sets, `sample(replace = True)` is real dangerous

- better target variable (age) normalization

- pretrained models

- attention/related techniques to focus on areas

- # Split Data into Training and Validation

```python
from sklearn.model_selection import train_test_split
rr_df = retina_df[['PatientId', 'level']].drop_duplicates()
train_ids, valid_ids = train_test_split(rr_df['PatientId'], 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = rr_df['level'])
raw_train_df = retina_df[retina_df['PatientId'].isin(train_ids)]
valid_df = retina_df[retina_df['PatientId'].isin(valid_ids)]
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
```

![1552490922983](/1552490922983.png)



O_o team

___

pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt

安装lasagne



复制/拷贝：

cp  文件名  路径      cp  hello.csv  ./python/ml：把当前目录的hello.csv拷贝到当前目的python文件夹里的ml文件夹里

cp 源文件名  新文件名   cp  hello.txt   world.txt：复制并改名,并存放在当前目录下  

cp file1 file2 复制一个文件 
cp dir/* . 复制一个目录下的所有文件到当前工作目录 
cp -a /tmp/dir1 . 复制一个目录到当前工作目录 
cp -a dir1 dir2 复制一个目录 

剪切/移动：

mv 文件名 路径

mv hello.csv ./python：把当前目录的hello.csv剪切到当前目的python文件夹里

mv  hello.txt  ../java/   把当前目录下的文件hello.txt剪切到上一级目录的子目录java目录里

mv  hello.txt  ..     把文件hello.txt移动到上一级目录
--------------------- 
作者：htbeker 
来源：CSDN 
原文：https://blog.csdn.net/htbeker/article/details/83578375 
版权声明：本文为博主原创文章，转载请附上博文链接！

     net = create_net(config)
  File "/mnt/data/yuchen/kaggle/kaggle_diabetic-master/nn.py", line 50, in create_net
​    net = Net(**args)
  File "/root/anaconda3/lib/python3.6/site-packages/nolearn/lasagne/base.py", line 247, in __init__
​    "The 'Objective' class is no longer supported, please "
ValueError: The 'Objective' class is no longer supported, please use 'nolearn.lasagne.objective' or similir



net

