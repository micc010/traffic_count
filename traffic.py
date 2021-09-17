import os
import logging
import logging.handlers
import random

import numpy as np
import skvideo.io
import cv2

import utils
# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)
random.seed(123)

from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)

# ============================================================================
IMAGE_DIR = "./out/images"
REPORT_DIR = "./out/report"
VIDEO_SOURCE = "09141043_0075.mp4"
TRAIN_FRAME_NUM = 500
SHAPE = (1080, 1920)  # HxW
EXIT_PTS = np.array([
    [[1091, 1078], [1038, 1032], [1297, 932], [1691, 1077]]
])

MIN_CONTOUR_WIDTH=25
MIN_CONTOUR_HEIGHT=12 
MAX_CONTOUR_WIDTH=200
MAX_CONTOUR_HEIGHT=200
PATH_SIZE=3
MAX_DST=10
X_WEIGHT=1.4
Y_WEIGHT=1.0

# ============================================================================


def train_bg_subtractor(inst, cap, num=TRAIN_FRAME_NUM):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print ('Training BG Subtractor...')
    i = 0
    for frame in cap:
        # inst.apply(frame, None, 0.001)
        inst.apply(frame, None, -1)
        i += 1
        if i >= num:
            return cap


def main():
    log = logging.getLogger("main")

    # creating exit mask from points, where we will be counting our vehicles
    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    # there is also bgslibrary, that seems to give better BG substruction, but
    # not tested it yet
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=TRAIN_FRAME_NUM, detectShadows=False)

    # processing pipline for programming conviniance
    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor, min_contour_width=MIN_CONTOUR_WIDTH, 
                         min_contour_height=MIN_CONTOUR_HEIGHT, max_contour_width=MAX_CONTOUR_WIDTH, 
                         max_contour_height=MAX_CONTOUR_HEIGHT, save_image=True, 
                         image_dir=IMAGE_DIR),
        # we use y_weight == 2.0 because traffic are moving vertically on video
        # use x_weight == 2.0 for horizontal.
        VehicleCounter(exit_masks=[exit_mask], path_size=PATH_SIZE, 
                       max_dst=MAX_DST, x_weight=X_WEIGHT,
                       y_weight=Y_WEIGHT),
        Visualizer(image_dir=IMAGE_DIR),
        CsvWriter(path=REPORT_DIR, name='report.csv')
    ], log_level=logging.DEBUG)

    # Set up image source
    # You can use also CV2, for some reason it not working for me
    cap = skvideo.io.vreader(VIDEO_SOURCE)

    # skipping TRAIN_FRAME_NUM frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, cap, num=TRAIN_FRAME_NUM)

    _frame_number = -1
    frame_number = -1
    for frame in cap:
        if not frame.any():
            log.error("Frame capture failed, stopping...")
            break

        # real frame number
        _frame_number += 1

        # skip every 2nd frame to speed up processing
        if _frame_number % 2 != 0:
            continue
        
        if frame_number >= 100:
            break

        # frame number that will be passed to pipline
        # this needed to make video from cutted frames
        frame_number += 1

        # plt.imshow(frame)
        # plt.show()
        # return

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        pipeline.run()

# ============================================================================

if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)
        
    if not os.path.exists(REPORT_DIR):
        log.debug("Creating report directory `%s`...", REPORT_DIR)
        os.makedirs(REPORT_DIR)

    main()
