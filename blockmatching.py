from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == '__main__':
    imgL = cv2.imread(
        "images/converted_intensity/imagePola_Intensity1.jpg", 0)
    # print(imgL.shape)
    # imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)

    imgR = cv2.imread(
        "images/1-1R.bmp", 0)
    # imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
    DbasicSubpixel = np.zeros_like(imgL)

    disparityRange = 50

    halfBlockSize = 3
    blockSize = 2*halfBlockSize + 1

    [imgHeight, imgWidth] = imgL.shape

    for m in range(imgHeight):
        minr = min([0, m - halfBlockSize])
        maxr = max([imgWidth, m+halfBlockSize])
        for n in range(imgWidth):
            minc = max([0, n-halfBlockSize])
            maxc = min([imgWidth, n+halfBlockSize])

            mind = 0
            maxd = min([disparityRange, imgWidth - maxc])

            template = imgR[minr:maxr, minc:maxc]

            numblock = maxd - mind + 1
            # print(type(numblock))

            blockDiffs = np.zeros((numblock, 1))

            for i in range(mind, maxd):
                block = imgL[minr:maxr, minc+i:maxc+i]

                blockindex = 1 - mind
                # print(blockDiffs.shape)
                # print(np.sum(np.sum(np.abs(template-block))))
                blockDiffs[blockindex, 0] = np.sum(
                    np.sum(np.abs(template - block)))

            sortedIndexes = np.argsort(blockDiffs)

            bestMatchIndex = sortedIndexes[0, 0]

            d = bestMatchIndex + mind - 1

            if (bestMatchIndex == 0) or (bestMatchIndex == numblock):
                DbasicSubpixel[m, n] = d
            else:
                C1 = blockDiffs[bestMatchIndex - 1]
                C2 = blockDiffs[bestMatchIndex]
                C1 = blockDiffs[bestMatchIndex + 1]

                DbasicSubpixel[m, n] = d - \
                    (0.5 * (C3 - C1) / (C1 - (2*C2) + C3))

        # if m % 10 == 0:

    plt.plot(DbasicSubpixel)
    plt.show()
