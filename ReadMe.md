# CVRP

- 已经能够进行模型的训练

- 需要继续编写的部分

    - **使用dataset、dataloader等封装`train.py`**    

    - **batch化训练**
  
      - 注：batch化可能导致Model部分需要部分改动，不过`train.py`就是按照batch=1的特殊情况进行编写的（已经加上了batch的维度）
      
      - 需要注意：batch化的话可能不同样例的解不一样（需要加上mask来辅助）
  
    - **在decode阶段加入考虑约束的mask**
  
      - 注：利用简单的与或操作即可实现
  
      - 比如为车辆的货物为0必须回到仓库等
  
      - 注：True才代表mask
    
    - 可视化（绘图）

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