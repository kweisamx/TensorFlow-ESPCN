from utils import (
    checkimage,
    modcrop,
)

import numpy as np
import math
import cv2
import glob
import os

def psnr(target, ref, scale):
	#assume RGB image
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    return 20*math.log10(255.0/rmse)


if __name__ == "__main__":
    data_HR = glob.glob(os.path.join('./Test/Set5',"*.bmp"))
    print(data_HR)
    data_LR = glob.glob('./result/result.png')
    print(data_LR)
    hr = modcrop(cv2.imread(data_HR[0]))
    lr = cv2.imread(data_LR[0])
    lr = modcrop(cv2.resize(lr,None,fx = 1.0/3 ,fy = 1.0/3, interpolation = cv2.INTER_CUBIC))
    print(psnr(lr, hr, scale = 3))
