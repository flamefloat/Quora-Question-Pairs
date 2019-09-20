# Quora-Question-Pairs
kaggle competition  
## 嵌入加速
* 将glove.text文件转化为一个存储单词的列表文件以及一个存储对应嵌入向量的tensor.pth文件。
* 建立一个单词-索引字典，将每次查询后的单词索引添加到字典里，提高重复查询的速度。

## 说明
* scaled dot attention is used: sqrt(d_k)
* model: 
   * dropout_rate = 0.3
* latent_task_model: 
   * dropout_rate = 0.2  
   * no relu in last hidden layer  
   * add three loss
## 问题及解决办法
* 过拟合
   * 训练epoch过大（120）， 改为：
   * 设定验证集监督训练情况
   * 加入输入dropout=0.1, encoder_dropout=0.3, outFC_dropout=0.5
