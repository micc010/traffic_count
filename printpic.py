# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 18:50:26 2021

@author: lt
"""

import cv2

import skvideo.io

import utils

VIDEO_SOURCE = "09141043_0075.mp4"

cap = skvideo.io.vreader(VIDEO_SOURCE)

fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

frame_number = -1

for frame in cap:
      
    if not frame.any():
        break
    
    if frame_number == 50:
        break
    
    frame_number += 1
    utils.save_frame(frame, "./out/frame_%04d.png" % frame_number)
    fgmask = fgbg.apply(frame)
    utils.save_frame(fgmask, "./out/mask_%04d.png" % frame_number, flip=False)

'''    
while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    fgmask = fgbg.apply(frame)
    cv2.imshow("frame",fgmask)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
'''
cap.close()