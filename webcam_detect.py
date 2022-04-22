import os
import json
import time
import threading
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from build_utils import img_utils, torch_utils, utils
from model import Darknet
from draw_box_utils import draw_box
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = "weights/yolov3spp-voc-512.pt"  # 改成自己训练好的权重文件
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    img_path = "test.jpg"  # 改成当前目录下需要测试的文件名
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}  # key和value对换，转换成索引：类别

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        img = torch.zeros((1, 3, img_size, img_size), device=device)  # 传入空的图片进行网络初始化，能提升预测速度
        model(img)

        # img_o = cv2.imread(img_path)  # BGR，图像应该当前路径下，或者给出完整的图像路径，绝对路径调用方式，要双反斜杠
        cap = cv2.VideoCapture(0)  # 调用自带的摄像头

        while True:
            # assert img_o is not None, "Image Not Found " + img_path
            ret, img_o = cap.read()  # 读取一帧

            img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]  # 将图片缩放到指定大小
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # （H,W,C）中的C由BGR to RGB, to 3x416x416,注意np中的transpose可操作多维，tensor中的不行
            img = np.ascontiguousarray(img)  # 判断img在内存中是否连续，不是则转成连续

            img = torch.from_numpy(img).to(device).float()
            img /= 255.0  # scale (0, 255) to (0, 1)
            img = img.unsqueeze(0)  # add batch dimension # 在img中dim=0的位置加上一个维数为1的维度，因输入模型的数据格式要求(B,C,H,W)

            t1 = torch_utils.time_synchronized()
            pred = model(img)[0]  # only get inference result
            t2 = torch_utils.time_synchronized()
            print(t2 - t1)
            # 非极大值抑制NMS处理
            pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
            t3 = time.time()
            print(t3 - t2)

            if pred is None:
                print("No target detected.")
                exit(0)

            # process detections
            pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()  # 将预测结果映射回原尺度上
            print(pred.shape)

            bboxes = pred[:, :4].detach().cpu().numpy()
            scores = pred[:, 4].detach().cpu().numpy()
            classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

            img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
            # time.sleep(5)
            # plt.imshow(img_o)
            # plt.show()
            img_o = np.array(img_o)  # 将img_o从PIL的image数据类型用np.array()转成ndarray类型
            img_o = img_o[:, :, ::-1]  # 调换为opencv所需的bgr格式
            cv2.imshow('img_o', img_o)
            # cv2.waitKey(5)  # 用于设置在显示完一帧图像后程序等待5ms 再显示下一帧视频
            # img_o.save("test_result.jpg")
            if cv2.waitKey(1) == 27:  # 用于设置在显示完一帧图像后程序等待1ms， 接受按键esc信息， 再显示下一帧视频
                break
        cap.release()
        cv2.destroyWindow('img_o')


if __name__ == "__main__":
    main()
