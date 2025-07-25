# import cv2
# import numpy as np
# from ultralytics import YOLO
#
# model = YOLO('yolov8n-seg.pt')  # 预训练的 YOLOv8n 模型
#
# # 对一系列图像执行批量推理
# results = model(['im1.jpg', 'im2.jpg'])  # 返回一个结果对象列表
#
# # 处理结果列表
# for result in results:
#     boxes = result.boxes  # Boxes 对象，用于边界框输出
#     masks = result.masks  # Masks 对象，用于分割掩码输出
#
#     # 假设类别0为树，类别1为切割点
#     tree_indices = [i for i, cls in enumerate(boxes.cls) if cls == 0]
#     cutpoint_indices = [i for i, cls in enumerate(boxes.cls) if cls == 1]
#
#     # 为树分配序号
#     tree_numbers = {idx: i + 1 for i, idx in enumerate(tree_indices)}
#     cutpoint_tree_numbers = {}
#
#     # 遍历每个切割点，判断其是否在树的矩形内
#     for cutpoint_idx in cutpoint_indices:
#         cutpoint_box = boxes.xyxy[cutpoint_idx].tolist()
#         cutpoint_x1, cutpoint_y1, cutpoint_x2, cutpoint_y2 = cutpoint_box
#
#         # 遍历每棵树，检查切割点是否在其矩形内
0#         for tree_idx in tree_indices:
#             tree_box = boxes.xyxy[tree_idx].tolist()
#             tree_x1, tree_y1, tree_x2, tree_y2 = tree_box
#
#             # 判断切割点矩形是否在树矩形内
#             if (cutpoint_x1 >= tree_x1 and cutpoint_y1 >= tree_y1 and
#                     cutpoint_x2 <= tree_x2 and cutpoint_y2 <= tree_y2):
#                 cutpoint_tree_numbers[cutpoint_idx] = tree_numbers[tree_idx]
#                 break  # 找到对应的树后退出循环
#
#     # 可视化结果
#     img = result.orig_img
#     # 绘制树的边界框和序号
#     for idx in tree_indices:
#         box = boxes.xyxy[idx].tolist()
#         number = tree_numbers[idx]
#         conf = boxes.conf[idx].item()  # 获取置信度
#         cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#         cv2.putText(img, f'tree: {number} (conf: {conf:.2f})', (int(box[0]), int(box[1]) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     # 绘制切割点的边界框和对应的树序号
#     for idx in cutpoint_indices:
#         if idx in cutpoint_tree_numbers:
#             box = boxes.xyxy[idx].tolist()
#             number = cutpoint_tree_numbers[idx]
#             conf = boxes.conf[idx].item()  # 获取置信度
#             cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
#             cv2.putText(img, f'cutting point: {number} (conf: {conf:.2f})', (int(box[0]), int(box[1]) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#     # 显示结果图像
#     cv2.imshow('Result', img)
#     cv2.waitKey(0)
#

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('F:/JD/ultralytics/runs/yolov8-seg/weights/best.pt')  # 预训练的 YOLOv8n 模型

# 对一系列图像执行批量推理
results = model(['F:/JD/ultralytics/Datasets/tree/images/val/9.jpg'])  # 返回一个结果对象列表

# 处理结果列表
for result in results:
    boxes = result.boxes  # Boxes 对象，用于边界框输出
    masks = result.masks  # Masks 对象，用于分割掩码输出

    # 假设类别0为树，类别1为切割点
    tree_indices = [i for i, cls in enumerate(boxes.cls) if cls == 0]
    cutpoint_indices = [i for i, cls in enumerate(boxes.cls) if cls == 1]

    # 为树分配序号
    tree_numbers = {idx: i + 1 for i, idx in enumerate(tree_indices)}
    cutpoint_tree_numbers = {}

    # 遍历每个切割点，判断其是否在树的矩形内
    for cutpoint_idx in cutpoint_indices:
        cutpoint_box = boxes.xyxy[cutpoint_idx].tolist()
        cutpoint_x1, cutpoint_y1, cutpoint_x2, cutpoint_y2 = cutpoint_box

        # 遍历每棵树，检查切割点是否在其矩形内
        for tree_idx in tree_indices:
            tree_box = boxes.xyxy[tree_idx].tolist()
            tree_x1, tree_y1, tree_x2, tree_y2 = tree_box

            # 判断切割点矩形是否在树矩形内
            if (cutpoint_x1 >= tree_x1 and cutpoint_y1 >= tree_y1 and
                    cutpoint_x2 <= tree_x2 and cutpoint_y2 <= tree_y2):
                cutpoint_tree_numbers[cutpoint_idx] = tree_numbers[tree_idx]
                break  # 找到对应的树后退出循环

    # 可视化结果
    img = result.orig_img.copy()  # 使用副本以避免修改原始图像

    # # 绘制掩码
    if masks is not None:
        for i, mask in enumerate(masks.masks):
            # 将掩码转换为 uint8 并缩放到与图像相同的尺寸
            mask_img = (mask * 255).astype(np.uint8)
            mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]))

            # 创建掩码的彩色表示
            color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
            mask_colored = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)

            # 将彩色掩码与图像融合
            img = cv2.addWeighted(img, 1, mask_colored, 0.5, 0)

            # 绘制掩码轮廓
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # 绘制树的边界框和序号
    for idx in tree_indices:
        box = boxes.xyxy[idx].tolist()
        number = tree_numbers[idx]
        conf = boxes.conf[idx].item()  # 获取置信度
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(img, f'tree: {number} (conf: {conf:.2f})', (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 绘制切割点的边界框和对应的树序号
    for idx in cutpoint_indices:
        if idx in cutpoint_tree_numbers:
            box = boxes.xyxy[idx].tolist()
            number = cutpoint_tree_numbers[idx]
            conf = boxes.conf[idx].item()  # 获取置信度
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(img, f'cutting point: {number} (conf: {conf:.2f})', (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 显示结果图像
    cv2.imshow('Result', img)
    cv2.waitKey(0)