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

test是最终测试集

‘’‘

INFO:tensorflow:2019-03-14 09:32:23.765030: Step 3999: Cross entropy = 0.558947
INFO:tensorflow:2019-03-14 09:32:23.841141: Step 3999: Validation accuracy = 66.0% (N=100)
2019-03-14 09:32:26.658318: W tensorflow/core/graph/graph_constructor.cc:1265] Importing a graph with a lower producer version 26 into an existing graph with producer version 27. Shape inference will have run different parts of the graph with different producer versions.
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
INFO:tensorflow:Restoring parameters from /tmp/_retrain_checkpoint
INFO:tensorflow:Final test accuracy = 75.1% (N=1439)

’‘’

tensorflow focal loss



INFO:tensorflow:Final test accuracy = 92.4% (N=1105)
INFO:tensorflow:=== MISCLASSIFIED TEST IMAGES ===
INFO:

G:\\now\\all\\train\DM\vk000422.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk001327.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk001938.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk002161.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk002427.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk002577.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk002942.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk002955.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk003036.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk003236.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk003373.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk003445.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk004312.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk004398.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk005148.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk005363.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk005561.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk005584.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk006182.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk006509.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk006621.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk006739.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk006876.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk007089.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk007330.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk007586.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk008183.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk010618.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk010928.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk011164.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk011203.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk011627.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk012027.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk012332.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk013030.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk013113.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk013202.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk013248.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk013353.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk013402.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk013449.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk014327.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk014823.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk014980.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk015193.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk015429.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk015491.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk015577.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk015691.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk015794.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk015821.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk017076.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk017106.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk017206.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk017265.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk017640.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk017977.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk017989.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk018002.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk018095.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk018169.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk018277.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk018304.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk018383.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk018636.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk018787.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk018819.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk019684.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk019761.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk019885.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk019960.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk019980.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk020069.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk020143.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk020159.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk020188.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk020263.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk021082.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk021256.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk021308.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk021342.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk021478.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DM\vk021675.jpeg  dr
INFO:tensorflow:                                  G:\\now\\all\\train\DR\vk020634.jpeg  dm











