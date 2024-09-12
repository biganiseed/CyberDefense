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

def feature_name( number ):
    return [
        "Number of announcements",
        "Number of withdrawals",
        "Number of announced NLRI prefixes",
        "Number of withdrawn NLRI prefixes",
        "Average AS-path length",
        "Maximum AS-path length",
        "Average unique AS-path length",
        "Number of duplicate announcements",
        "Number of implicit withdrawals",
        "Number of duplicate withdrawals",
        "Maximum edit distance",
        "Arrival rate [Number]",
        "Average edit distance",
        "Maximum AS-path length = 11",
        "Maximum AS-path length = 12",
        "Maximum AS-path length = 13",
        "Maximum AS-path length = 14",
        "Maximum AS-path length = 15",
        "Maximum AS-path length = 16",
        "Maximum AS-path length = 17",
        "Maximum AS-path length = 18",
        "Maximum AS-path length = 19",
        "Maximum AS-path length = 20",
        "Maximum edit distance = 7",
        "Maximum edit distance = 8",
        "Maximum edit distance = 9",
        "Maximum edit distance = 10",
        "Maximum edit distance = 11",
        "Maximum edit distance = 12",
        "Maximum edit distance = 13",
        "Maximum edit distance = 14",
        "Maximum edit distance = 15",
        "Maximum edit distance = 16",
        "Number of Interior Gateway Protocol (IGP) packets",
        "Number of Exterior Gateway Protocol (EGP) packets",
        "Number of incomplete packets",
        "Packet size (B) [Average]",
        "Label"
    ][number-1]
    


    

def show_hist( feature, title = "Feature Hist", sub = None, fig = None ):
    if sub is not None:
        ax1 = sub
    else:
        # 创建图形
        _ , ax1 = plt.subplots()
    ax1.hist( feature, bins = 20, color = 'blue',edgecolor='black', alpha=0.7 )
    ax1.set_xlabel("Feature Value")
    ax1.set_ylabel("Frequency") 
    if sub is None:
        plt.title( title )
        plt.show()
    else:
        sub.set_title( title )

def show_feature_label( feature , label, title = "Feature Line", sub = None, fig = None ):
    if sub is not None:
        ax1 = sub
    else:
        # 创建图形
        fig, ax1 = plt.subplots()
        fig.suptitle( title)
    
    # 绘制第一条曲线（左侧纵坐标）
    ax1.plot(range(len(feature)), feature)  # 绿色曲线
    ax1.set_ylabel('Feature Value', color='g')  # 左侧纵轴标签及颜色
    ax1.tick_params(axis='y', labelcolor='g')  # 左侧纵轴刻度颜色

    # 创建第二条纵坐标轴并绘制第二条曲线
    ax2 = ax1.twinx()  # 共享同一个x轴，但有独立的y轴
    ax2.plot(range(len(label)), label, color='r')  # 蓝色曲线
    ax2.set_ylabel('Label', color='b')  # 右侧纵轴标签及颜色
    ax2.tick_params(axis='y', labelcolor='b')  # 右侧纵轴刻度颜色

    if sub is None:
        # 显示图表
        fig.tight_layout()  # 调整布局，防止标签重叠
        plt.show()  
    else:
        sub.set_title( title )