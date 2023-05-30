# FedLearning

1. args.py: 用于设置参数
2. main.py: 包括加载数据、server分发模型参数、server端聚合模型参数FedAvg和最终的绘图以及测试调用
3. CNN.py: 定义CNN模型
4. test.py: 模型测试函数，读入一轮的全局模型，返回该模型的loss和accuracy
5. local_update.py: 定义client端本地更新函数，读入client的id，与server通信获取全局模型，返回该client的loss并回送本地更新后的模型参数

运行：

`python main.py --select=xx`

其中select参数为每一轮选择的客户端数量，可取0-20之间的整数。

其他可选运行参数见args.py，包括batch_size、epoch、lr等