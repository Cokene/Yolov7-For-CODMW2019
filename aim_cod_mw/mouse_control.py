# _*_ coding : utf-8 _*_
# @Time : 2022/8/20 22:05
# @Author : shliz
# @File : mouse_control
# @Project : Yolov7ForCODMW2019
import deprecated


@deprecated
def aim_lock(aims, mouse, x, y):
    # 获取鼠标当前位置的坐标
    mouse_pos_x, mouse_pos_y = mouse.position
    dist_list = []      # 鼠标中心离目标距离
    # 遍历检测到的每个目标
    for det in aims:
        # 只获取目标的坐标
        _, x_c, y_c, _, _ = det
        dist = (x * float(x_c) - mouse_pos_x)**2 + (y * float(y_c) - mouse_pos_y)**2
        dist_list.append(dist)
        
    det = aims[dist_list.index(min(dist_list))]
    tag, x_centor, y_centor, width, height = det
    tag = int(tag)
    x_centor, width = x * float(x_centor), x * float(width)
    y_centor, height = y * float(y_centor), y * float(height)
    # 锁定模式 0 head 1 body（body参数暂未开放）
    if tag == 0:
        mouse.position = (x_centor, y_centor)