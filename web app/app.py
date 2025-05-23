# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:21:30 2023

@author: Administrator
"""


# app.py

from flask import Flask, request, render_template
import torch
import torch.nn as nn
import pandas as pd

# 实例化一个app应用，并导入数据
app = Flask(__name__)

# 定义模型图结构（如果模型保存的是字典而不是整个模型则需要这个步骤）
class ComplexLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(ComplexLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cpu')
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cpu')
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out

# 加载模型
model = torch.load('./full_model.pth')

# 定义Z-score标准化函数
def zscore(x):
    return (x - x.mean()) / x.std()

# 初始页面，装饰器@app.route("/")定义路由，当用户在浏览器输入的路由路径为"/"时，前端直接渲染初始界面
@app.route("/")
def Home():
    return render_template("index.html") # 在app.py文件同一目录新建templates文件夹然后放入这个index.html文件

# 定义一个预测路由，以post方式提交数据
@app.route("/predict", methods = ["POST"])
def predict():
    # 从index.html中采集用户输入数据，并且保存到名为float_features的python列表中
    float_features = [float(x) for x in request.form.values()] # 数据的名称等index.html中定义
    
    # 读取本地的部分训练数据（仅X_train,无y_train），主要是为了数据标准化
    example_data = pd.read_csv('./example.csv') # 示例训练数据
    
    # 把用户数据的数据合并到示例数据的最后一行
    example_data.loc[len(example_data)] = float_features
    
    # 对每一列数据进行Z-score标准化
    features_encoded = example_data.apply(zscore)
    
    # 把最后一行数据重新提取出来，并且转化为tensor,然后reshape成模型输入需要的形状
    features = torch.Tensor(features_encoded.iloc[-1,:].values)
    features = features.reshape(1, 3, 2)
    
    # 开启模型评估
    model.eval()
    with torch.no_grad():
        new_outputs = model(torch.Tensor(features))
    
    # 输出各类别预测概率
    probabilities = nn.functional.softmax(new_outputs, dim=1)
    
    # 把输出概率转换为numpy array
    probabilities = probabilities.detach().numpy()[0]
    
    # 获得预测类别
    prediction = probabilities.argmax() 

    # 定义各预测类别对应的文字标签
    mapping = {0:'Not reversible malnutrition', 1:'Reversible malnutrition'}
    
    prediction = mapping[prediction]
    # 在用户输入数据上预测出结果后，返回在初始界面上
    return render_template("index.html", 
                           prediction_text = "The prediction is: {}".format(prediction),
                           prediction_text2 = "The probabilities of non-reversible malnutrition (class 0) and reversible malnutrition (class 1) are: {:.4f} and {:.4f}".format(probabilities[0], probabilities[1]))

 # 如果要在spyder终端运行app, debug=False。如果想在网页显示错误信息，则为True,但是要在命令行中以python app.py运行
if __name__ == "__main__":
    app.run(debug=False)