#!/usr/bin/env python3

import cv2
from keras.models import load_model
import numpy as np
from collections import deque # 両端における append や pop を高速に行えるリスト風のコンテナ
# 汎用の Python 組み込みコンテナ dict, list, set, および tuple に代わる、特殊なコンテナデータ型を実装しています。
import os

model = load_model('QuickDraw.h5')


def main():
    emojis = get_QD_emojis()
    cap = cv2.VideoCapture(0)
    Lower_green = np.array([110, 50, 50])    #青色のみを選択抽出
    Upper_green = np.array([130, 255, 255])
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    pred_class = 0

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 色相・彩度・明度の3つで表現する「HSV」

        # cv2.imshow("hsv", hsv)
        # k = cv2.waitKey(10)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, Lower_green, Upper_green) # 範囲を指定して2値化処理

        # cv2.imshow("mask1", mask)
        # k = cv2.waitKey(10)

        mask = cv2.erode(mask, kernel, iterations=2)      # 指定の構造要素を用いて画像の収縮 # オブジェクトの境界を侵食

        # cv2.imshow("mask2", mask)
        # k = cv2.waitKey(10)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # ホワイトのノイズ除去 収縮の後に膨張をする処理

        # cv2.imshow("mask3", mask)
        # k = cv2.waitKey(10)

        # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel) #  膨張の後に収縮 をする処理です．前景領域中の小さな(黒い)穴を埋めるのに役立ちます．
        mask = cv2.dilate(mask, kernel, iterations=1) # 膨張処理

        # cv2.imshow("mask4", mask)
        # k = cv2.waitKey(10)

        res = cv2.bitwise_and(img, img, mask=mask) 

        # cv2.imshow("res", res)
        # k = cv2.waitKey(10)

        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # cv2.RETR_EXTERNAL 最も外側の輪郭のみが返されます
        # cv2.CHAIN_APPROX_SIMPLE フラグを指定すると，輪郭を圧縮して冗長な点の情報を削除し，メモリの使用を抑えられます．
        center = None
        # print(cnts)
  
        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt) # 物体の最小外接円を計算する
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt) # 重心を求める
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                    cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
        elif len(cnts) == 0:
            if len(pts) != []:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)

                # cv2.imshow("blackboard_gray", blackboard_gray)
                # k = cv2.waitKey(10)

                blur1 = cv2.medianBlur(blackboard_gray, 15)

                # cv2.imshow("blur1", blur1)
                # k = cv2.waitKey(10)

                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)

                # cv2.imshow("blur2", blur1)
                # k = cv2.waitKey(10)

                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # cv2.imshow("thresh1", thresh1)
                # k = cv2.waitKey(10)

                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

                # print(blackboard_cnts)

                if len(blackboard_cnts) >= 1:
                    cnt = max(blackboard_cnts, key=cv2.contourArea)
                    print(cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt) # contour輪郭を短形で切り取る
                        digit = blackboard_gray[y:y + h, x:x + w]

                        cv2.imshow("digit", digit)
                        k = cv2.waitKey(1000)

                        pred_probab, pred_class = keras_predict(model, digit)
                        print(pred_class, pred_probab)

            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            img = overlay(img, emojis[pred_class], 400, 250, 100, 100)
        cv2.imshow("Frame", img)
        k = cv2.waitKey(10)
        if k == 27:
            break


def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def get_QD_emojis():
    emojis_folder = 'qd_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder + str(emoji) + '.png', -1))
    return emojis


def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y + h, x:x + w] = blend_transparent(image[y:y + h, x:x + w], emoji)
    except:
        pass
    return image


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
if __name__ == '__main__':
    main()
