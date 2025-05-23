# -*- coding: utf-8 -*-
"""

@author: Administrator
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import onnxruntime as ort
import pandas as pd
import os

class LSTMPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("WAL-net v1.0 by liangyuyin1988@tmmu.edu.cn")
        
        # 检查必要文件
        self.check_files()
        
        # 加载ONNX模型
        try:
            self.session = ort.InferenceSession("model.onnx")
        except Exception as e:
            self.show_error(f"Model loading failed: {str(e)}")
            
        # 创建输入界面
        self.create_widgets()
    
    def check_files(self):
        """检查依赖文件是否存在"""
        if not os.path.exists("example.csv"):
            self.show_error("example.csv not found")
        if not os.path.exists("model.onnx"):
            self.show_error("model.onnx not found")

    def show_error(self, msg):
        """显示错误并退出"""
        messagebox.showerror("Initialization error", msg)
        self.master.destroy()
        exit()

    def create_widgets(self):
        # 输入变量分组（与CSV列顺序保持一致）
        variables = [
            ("BMI, -6mo & -1mo & now, kg/m\u00B2", ["bmi1", "bmi2", "bmi3"]),
            ("ASMI, -6mo & -1mo & now, kg/m\u00B2", ["asmi1", "asmi2", "asmi3"])
        ]
        
        # 使用Notebook实现选项卡布局
        notebook = ttk.Notebook(self.master)
        notebook.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 输入面板
        input_frame = ttk.Frame(notebook)
        notebook.add(input_frame, text="Input parameters")
        
        self.entries = {}
        for idx, (group, vars) in enumerate(variables):
            frame = ttk.LabelFrame(input_frame, text=group)
            frame.grid(row=0, column=idx, padx=10, pady=5, sticky=tk.N)
            
            for i, var in enumerate(vars):
                ttk.Label(frame, text=var+":", width=8).grid(row=i, column=0, padx=5, pady=2)
                entry = ttk.Entry(frame, width=12)
                entry.grid(row=i, column=1, padx=5, pady=2)
                self.entries[var] = entry
        
        # 控制按钮区域
        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_inputs).pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        self.result_var = tk.StringVar()
        result_label = ttk.Label(self.master, textvariable=self.result_var, 
                               font=("Arial", 12, "bold"), foreground="blue")
        result_label.pack(pady=10)
    
    def get_inputs(self):
        """获取并验证输入值"""
        values = {}
        try:
            for var in self.entries:
                values[var] = float(self.entries[var].get())
            return values
        except ValueError:
            messagebox.showerror("Input error", "all parameters must be valid numbers")
            return None
    
    def preprocess_data(self, inputs):
        """数据预处理流水线"""
        try:
            # 读取历史数据
            df = pd.read_csv("example.csv")
            
            # 创建新数据行（保持与CSV相同的列顺序）
            new_row = pd.DataFrame([inputs], columns=df.columns)
            
            # 追加新数据并保存
            updated_df = pd.concat([df, new_row], ignore_index=True)
            updated_df.to_csv("example.csv", index=False)
            
            # 计算标准化参数
            means = updated_df.mean()
            stds = updated_df.std(ddof=0)
            
            # 防止除零错误
            if (stds == 0).any():
                messagebox.showerror("Calculation error", "There are feature columns with zero standard deviation, making standardization impossible.")
                return None
                
            # 执行标准化
            normalized = (updated_df - means) / stds
            
            # 提取最新数据并重塑维度
            last_row = normalized.iloc[-1].values.astype(np.float32)
            
            # 原始维度重塑（1,3,2）
            base_data = np.array([
                [last_row[0], last_row[3]],  # bmi1, asmi1
                [last_row[1], last_row[4]],  # bmi2, asmi2
                [last_row[2], last_row[5]]   # bmi3, asmi3
            ])[np.newaxis, ...]  # shape: (1, 3, 2)
            
            # 复制数据到32个样本（模型输入要求）
            batch_data = np.tile(base_data, (32, 1, 1))  # shape: (32, 3, 2)
            
            return batch_data
        except Exception as e:
            messagebox.showerror("Preprocessing error", f"Data processing failed: {str(e)}")
            return None
    
    def predict(self):
        """执行预测流程"""
        inputs = self.get_inputs()
        if inputs is None:
            return
        
        processed = self.preprocess_data(inputs)
        if processed is None:
            return
        
        try:
            # ONNX推理
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            prediction = self.session.run([output_name], {input_name: processed})
            
            # 解析预测结果（取第一个样本）
            probabilities = prediction[0][0]  # 取batch中第一个样本的结果
            result = "Reversible malnutrition" if probabilities[1] > 0.5 else "Non-reversible malnutrition"
            confidence = max(probabilities) * 100
            
            self.result_var.set(f"Prediction result: {result} (Confidence: {confidence:.1f}%)")
            
        except Exception as e:
            messagebox.showerror("Inference error", f"Model execution failed: {str(e)}")
    
    def clear_inputs(self):
        """清空所有输入框"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_var.set("")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("520x320")
    app = LSTMPredictorApp(root)
    root.mainloop()

