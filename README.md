# zkkkkkqqqqq

最近在学 PyTorch，写了几个小练习。

- **[MNIST](https://github.com/zkkkkkqqqqq/zkkkkkqqqqq/blob/main/mnist/train_mnist.py)**：用全连接网络识别手写数字，练习了怎么分批训练、怎么算准确率。
- **[Titanic](https://github.com/zkkkkkqqqqq/zkkkkkqqqqq/blob/main/titanic/titanic.py)**：处理了年龄缺失、船票和船舱信息，训练模型预测乘客是否幸存。Kaggle 提交得分是 **77.03%**。
- **[Spaceship Titanic](https://github.com/zkkkkkqqqqq/zkkkkkqqqqq/blob/main/spaceship_titanic/ship.py)**：把船舱拆成甲板、舱号和左右舷，再结合乘客资料和消费记录做预测，生成了 4,277 条提交结果。

正在准备复现 LoRA+。[CUDA 检查脚本](https://github.com/zkkkkkqqqqq/zkkkkkqqqqq/blob/main/lora_plus/check_cuda.py)已跑通，接下来想做一组 LoRA 和 LoRA+ 的对照实验。

### 怎么跑

需要 Python、PyTorch、torchvision、NumPy、pandas 和 scikit-learn。

- MNIST：运行 `python mnist/train_mnist.py`。第一次会把数据下载到 `mnist/data/`。
- Titanic 和 Spaceship Titanic：分别在 [Titanic](https://www.kaggle.com/competitions/titanic/data) 和 [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data) 下载 `train.csv` 和 `test.csv`，放到对应脚本旁边。运行 `python titanic/titanic.py` 或 `python spaceship_titanic/ship.py`，预测会保存到各自目录的 `submission.csv`。
