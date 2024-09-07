"""
    @author Ruby Du
    @email candyreddh@gmail.com
    @date Sep. 6, 2024
    @version: 0.1.0
    @description:
                This module contains common functions for visualizing the BGP features.

    This Python code (versions 3.8+)
"""
import matplotlib.pyplot as plt

def show_hist( feature, title = "Feature Hist" ):
    plt.hist( feature, bins = 20, color = 'blue',edgecolor='black', alpha=0.7 )
    plt.title( title )
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency") 
    plt.show()

def show_feature_label( feature , label, title = "Feature Line" ):
    # 创建图形
    fig, ax1 = plt.subplots()
    fig.suptitle( title)
    
    # 绘制第一条曲线（左侧纵坐标）
    ax1.plot(range(len(feature)), feature)  # 绿色曲线
    ax1.set_ylabel('Feature', color='g')  # 左侧纵轴标签及颜色
    ax1.tick_params(axis='y', labelcolor='g')  # 左侧纵轴刻度颜色

    # 创建第二条纵坐标轴并绘制第二条曲线
    ax2 = ax1.twinx()  # 共享同一个x轴，但有独立的y轴
    ax2.plot(range(len(label)), label, color='r')  # 蓝色曲线
    ax2.set_ylabel('Label', color='b')  # 右侧纵轴标签及颜色
    ax2.tick_params(axis='y', labelcolor='b')  # 右侧纵轴刻度颜色

    # 显示图表
    fig.tight_layout()  # 调整布局，防止标签重叠
    plt.show()  