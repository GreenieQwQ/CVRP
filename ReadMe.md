# CVRP

- 已经能够进行模型的训练

- 需要继续编写的部分

    - 使用dataset、dataloader等封装`train.py`    

    - batch化训练
    
    - 可视化

- 可改进的部分

    - 正确性检查

    - 见每个地方的TODO

## 运行

### 准备数据

将数据存放到根目录的data文件夹下。

### 训练模型
使用命令
```
    python train.py 
```
在model文件夹下对应模型文件夹下使用命令
```
    tensorboard --logdir=record
```
便可以观察loss的变化情况。

### 运行ortools

安装依赖
```
    pip install ortools
```
使用命令
```
    python oryools_cvrp.py 
```
即可进行运行。