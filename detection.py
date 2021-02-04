import cv2
import os
from enum import Enum
import numpy as np
import csv

class Method(Enum):
    ORB = 1
    SIFT_BRUTE_FORCE = 2
    SIFT_FLANN = 3


class FeatureExtraction(object):
    def get_filepaths(self, directory):
        file_paths = []  # 将存储所有的全文件路径的列表
        # Walk the tree.
        for root, directories, files in os.walk(directory):
            for filename in files:
                # 加入两个字符串以形成完整的文件路径
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.
        return file_paths  # 所有文件的路径

    def resize_function(self, x):
        step = 200
        x = x - 500.0
        if x < 0:
            scale = 1 / ((1.005) ** abs(x))
        else:
            scale = (x + step) / step
        return scale

    def resize_with_keypoints_and_descriptor(self, image, detector, scale=None):
        if scale is not None:
            image = cv2.resize(image, None,
                               fx=scale,
                               fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('GRAY NORMAL', imageGray)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # imageGray = clahe.apply(imageGray)
        # imageGray = cv2.equalizeHist(imageGray, None)
        # cv2.imshow('GRAY - HISTOGRAM OPERATIONS', imageGray)
        keypoints, descriptor = detector.detectAndCompute(imageGray, None)
        imageWithKeypoints = np.zeros(image.shape, image.dtype)
        cv2.drawKeypoints(image, keypoints, imageWithKeypoints, self.red,
                          cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        return keypoints, descriptor, image, imageWithKeypoints

    def resize_set(self):
        print('Resizing entire sets...')

        scale = self.PATTERNIMAGESCALE
        if self.METHOD == Method.ORB:
            self.scalesList = np.linspace(scale / 20, scale, 10)
        else:
            self.scalesList = np.linspace(scale, scale, 1)
        print('Scale linspace: ' + str(self.scalesList))

        self.logoImagesSet = []

        for image in self.logoImagesSetOriginal:
            for size in self.scalesList:
                self.logoImagesSet.append(
                    self.resize_with_keypoints_and_descriptor(image, self.keyPointsLogoDetector, size))

        self.patternImagesSet = []

        for image in self.patternImagesSetOriginal:
            self.patternImagesSet.append(self.resize_with_keypoints_and_descriptor(
                image, self.keyPointsPatternDetector,
                scale))

    def getSetResults(self, images_set, imgKeyPoints, imgDescriptor, distance):
        results = []

        for imageFromSet in images_set:
            # Match descriptors.

            imageDescriptorNumber = 0
            goodMatchesNumber = 0
            inliersNumber = 0
            statistics = (imageDescriptorNumber, goodMatchesNumber, inliersNumber)
            dstPoints = None
            M = None
            good = None
            matchesMask = None
            w = None
            h = None

            if imgDescriptor is not None and imageFromSet[1] is not None:
                imageDescriptorNumber = len(imgDescriptor)
                good = []
                if self.METHOD == Method.ORB or (
                                (self.METHOD == Method.SIFT_BRUTE_FORCE or self.METHOD == Method.SIFT_FLANN) and len(
                                imgDescriptor) > 1 and len(
                            imageFromSet[1]) > 1):
                    if self.METHOD == Method.ORB:
                        matches = self.keyPointsMatcher.match(imageFromSet[1], imgDescriptor)
                        # Sort them in the order of their distance.
                        matches = sorted(matches, key=lambda x: x.distance)
                        for m in matches:
                            if m.distance < self.DISTANCE:
                                good.append(m)
                    # 利用近似k近邻算法去寻找一致性，FLANN方法比BF（Brute-Force）方法快的多：
                    elif self.METHOD == Method.SIFT_BRUTE_FORCE or self.METHOD == Method.SIFT_FLANN:
                        matches = self.keyPointsMatcher.knnMatch(imageFromSet[1], imgDescriptor,
                                                                 k=2)  # 函数返回一个训练集和询问集的一致性列表
                        for match in matches:
                            ## 用比值判别法（ratio test）删除离群点
                            if len(match) == 2:
                                m, n = match
                                # 这里使用的kNN匹配的k值为2（在训练集中找两个点），第一个匹配的是最近邻，第二个匹配的是次近邻。直觉上，一个正确的匹配会更接近第一个邻居。
                                # 换句话说，一个[不]正确的匹配，[两个邻居]的距离是相似的。因此，我们可以通过查看二者距离的不同来评判距匹配程度的好坏。
                                # 比值检测认为第一个匹配和第二个匹配的比值小于一个给定的值（一般是0.5），这里是0.7：
                                # print(m.distance, distance * n.distance / 100)
                                if m.distance < distance * n.distance / 100:
                                    good.append(m)

                    goodMatchesNumber = len(good)


                    matchesMask = []
                    M = None
                    if len(good) >= 1:
                        # # 获取关键点的坐标
                        # 1. # 将所有好的匹配的对应点的坐标存储下来，就是为了从序列中随机选取4组，以便下一步计算单应矩阵
                        src_pts = np.float32([imageFromSet[0][m.queryIdx].pt for m in good]).reshape(-1, 1,
                                                                                                     2)  # [[x,y],]
                        dst_pts = np.float32([imgKeyPoints[n.trainIdx].pt for n in good]).reshape(-1, 1, 2)
                        # 2.# 单应性估计
                        #    由于我们的对象是平面且固定的，所以我们就可以找到两幅图片特征点的单应性变换。得到单应性变换的矩阵后就可以计算对应的目标角点：
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.0, None, 2000,
                                                     0.995, )  # 单应矩阵估计,利用RANSAC方法计算单应矩阵，,置信度设为0.99 循环次数设置为2000
                        
                        # 有了H（M）单应性矩阵，我们可以查看源点被映射到query image中的位置
                        '''
                        计算单应性矩阵(homography)，在这个函数参数中，输入的src_pts和dst_pts是两个对应的序列，这两组序列的每一对数据一一匹配，其中既有正确的匹配，也有错误的匹配
                        ，正确的可以称为内点，错误的称为外点，RANSAC方法就是从这些包含错误匹配的数据中，分离出正确的匹配，并且求得单应矩阵。

                        返回值中M为变换矩阵。mask是掩模，online的点。

                        mask：标记矩阵，标记内点和外点.他和m1，m2的长度一样，当一个m1和m2中的点为内点时，mask相应的标记为1，反之为0，说白了，通过mask我们最终可以知道序列中哪些是内点，哪些是外点。
                        M(model)：就是我们需要求解的单应矩阵.
                        ransacReprojThreshold=0.0：为阈值，当某一个匹配与估计的假设小于阈值时，则被认为是一个内点，这个阈值，openCV默认给的是3，后期使用的时候自己也可以修改。
                        confidence：为置信度，其实也就是人为的规定了一个数值，这个数值可以大致表示RANSAC结果的准确性，这个值初始时被设置为0.995
                        maxIters：为初始迭代次数，RANSAC算法核心就是不断的迭代，这个值就是迭代的次数，默认设为了2000

                        这个函数的前期，主要是设置了一些变量然后赋初值，然后转换相应的格式等等。

                        后面，由变换矩阵，求得变换后的物体边界四个点  
                        plt.imshow(inliersNumber), plt.show()
                        '''
                        if self.maskEnable:
                            matchesMask = mask.ravel().tolist()
                            # // 把内点转换为drawMatches可以使用的格式
                            # 通过cv2.drawMatchesKnn画出匹配的特征点，再将好的匹配返回
                            #

                        # 3.# 有了H单应性矩阵，我们可以查看源点被映射到query image中的位置
                        h, w = imageFromSet[2].shape[:2]
                        # print(h, w)
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                        # 4.# perspectiveTransform返回点的列表
                        if M is not None:
                            dst = cv2.perspectiveTransform(pts, M)
                            dstPoints = [np.int32(dst)]  # //由变换矩阵，求得变换后的物体边界四个点.反复计算，求得4个点坐标
                        # print(dstPoints)

                        # 5.# 计算非野点个数
                        # // 状态为0表示野点(离群点)
                        inliersNumber = len([x for x in mask if x != 0])
                    else:
                        print('Not enough matches!')

                    statistics = (imageDescriptorNumber, goodMatchesNumber, inliersNumber)
            results.append(
                (statistics, dstPoints, (M, (w, h)), good, matchesMask))
        return results  # 这个结果传入下面函数：

    def getBestResultIndex(self, totalResults):
        counter = 0
        bestResult = 0
        index = 0
        for result in totalResults:
            if result[0][2] >= bestResult:
                bestResult = result[0][2]
                index = counter
            counter = counter + 1
        return index

    def generateOutputImages(self, name, patternImagesSet, index, additionalName, cameraImageWithKeypoints,
                             imgKeyPoints):
        matchesImage = np.zeros(self.INITSHAPE, self.INITDTYPE)
        warpedImage = np.zeros(self.INITSHAPE, self.INITDTYPE)
        # 单应性矩阵图（就是上面都是圈圈的图）
        homograpyImage = cameraImageWithKeypoints.copy()

        if self.totalResults[index][1] is not None and self.totalResults[index][2][0] is not None:
            cv2.polylines(homograpyImage, self.totalResults[index][1], True, [100, 255, 0], 5, cv2.LINE_AA)

            # 截出图上的logo图
            warpedImage = cv2.warpPerspective(self.cameraImage, self.totalResults[index][2][0],
                                              self.totalResults[index][2][1], None,
                                              cv2.WARP_INVERSE_MAP,
                                              cv2.BORDER_CONSTANT, (0, 0, 0))
            # plt.imshow(warpedImage), plt.show()

            if self.totalResults[index][3] is not None and self.totalResults[index][4] is not None:
                ## 通过cv2.drawMatchesKnn画出匹配的特征点，再将好的匹配返回
                matchesImage = cv2.drawMatches(patternImagesSet[index][3], patternImagesSet[index][0], homograpyImage,
                                               imgKeyPoints,
                                               self.totalResults[index][3],
                                               None,
                                               self.blue, self.red,
                                               self.totalResults[index][4],
                                               cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        print('Number [ ' + name + ' ]: ' + str(additionalName))
        print('Length of best image descriptor [ ' + name + ' ]: ' + str(self.totalResults[index][0][0]))
        print('Good matches [ ' + name + ' ]: ' + str(self.totalResults[index][0][1]))
        print('Inliers [ ' + name + ' ]: ' + str(self.totalResults[index][0][2]))


        #
        # cv2.imshow('Original image', self.cameraImage)
        #
        # cv2.imshow('Keypoints - image', cameraImageWithKeypoints)
        # cv2.imshow('Keypoints - pattern', patternImagesSet[index][3])
        #
        # cv2.imshow('Homography', homograpyImage)
        #
        # cv2.imshow('Matches', matchesImage)
        #
        # cv2.imshow('Pattern image', patternImagesSet[index][2])
        # cv2.imshow('Warped image', warpedImage)

        outputImages = [matchesImage, homograpyImage, warpedImage, patternImagesSet[index][2],
                        patternImagesSet[index][3], cameraImageWithKeypoints, self.cameraImage]
        return outputImages

    def saveResults(self, outputImages, name, additionalName, index, pattern_name=''):
        if name == 'PATTERN':
            newCataloguePath = os.path.join(self.DESTINATIONPATH, str(additionalName), name, pattern_name)
            if pattern_name == 'x':
                self.csv_input[str(pattern_name)] = 1 if self.totalResults[index][0][1] == 14 or self.totalResults[index][0][1] == 15 else 0
            else:
                self.csv_input[str(pattern_name)] = 1 if self.totalResults[index][0][2]/self.totalResults[index][0][1] > self.pattern_threshold else 0
        else:
            newCataloguePath = os.path.join(self.DESTINATIONPATH, str(additionalName), name)
            self.csv_input['notepad'] = 1 if self.totalResults[index][0][2]/self.totalResults[index][0][1] > self.logo_threshold else 0

        os.makedirs(newCataloguePath, exist_ok=True)
        cv2.imwrite(os.path.join(newCataloguePath, '7_Original image_' + name + '.png'), outputImages[6])

        # cv2.imwrite(os.path.join(newCataloguePath, '6_Keypoints - image_' + name + '.png'), outputImages[5])
        # cv2.imwrite(os.path.join(newCataloguePath, '5_Keypoints - pattern_' + name + '.png'), outputImages[4])

        cv2.imwrite(os.path.join(newCataloguePath, '2_Homography_' + name + '.png'), outputImages[1])

        cv2.imwrite(os.path.join(newCataloguePath, '1_Matches_' + name + '.png'), outputImages[0])

        cv2.imwrite(os.path.join(newCataloguePath, '4_Pattern image_' + name + '.png'), outputImages[3])
        # cv2.imwrite(os.path.join(newCataloguePath, '3_Warped image_' + name + '.png'), outputImages[2])

    def compareWithLogo(self, name, maskEnable, saveFlag):
        self.maskEnable = maskEnable

        imgKeyPoints, imgDescriptor, self.cameraImage, cameraImageWithKeypoints = self.resize_with_keypoints_and_descriptor(
            sourceImage, self.keyPointsLogoDetector,  # 保留特征多
            self.SOURCEIMAGESCALE)

        self.totalResults = self.getSetResults(self.logoImagesSet, imgKeyPoints, imgDescriptor, 90)

        index = self.getBestResultIndex(self.totalResults)

        outputImages = self.generateOutputImages('LOGO', self.logoImagesSet, index, name, cameraImageWithKeypoints,
                                                 imgKeyPoints)

        if saveFlag:
            self.saveResults(outputImages, 'LOGO', name, index)

        outputImages.append(self.logoPaths[index // len(self.scalesList)])

        return outputImages

    def compareWithPattern(self, name, maskEnable, saveFlag):
        self.maskEnable = maskEnable

        imgKeyPoints, imgDescriptor, self.cameraImage, cameraImageWithKeypoints = self.resize_with_keypoints_and_descriptor(
            sourceImage, self.keyPointsPatternDetector,  # 保留特征少
            self.SOURCEIMAGESCALE)

        self.totalResults = self.getSetResults(self.patternImagesSet, imgKeyPoints, imgDescriptor, self.DISTANCE)

        # index = self.getBestResultIndex(self.totalResults)
        for index in range(len(self.totalResults)):
            outputImages = self.generateOutputImages('PATTERN', self.patternImagesSet, index, name,
                                                     cameraImageWithKeypoints,
                                                     imgKeyPoints)

            if saveFlag:
                pattern_filename = os.path.split(self.patternPaths[index])[-1]
                fname = os.path.splitext(pattern_filename)[0]
                self.saveResults(outputImages, 'PATTERN', name, index, fname)

        outputImages.append(self.patternPaths[index])
        return outputImages

    def changePatternScale(self, scale):
        self.PATTERNIMAGESCALE = scale
        self.resize_set()

    def __init__(self, method, logoNfeatures, patternNfeatures, patternImageScale, sourceImagescale, distance):
        print('Preprocessing...')

        # IMPORTANT VARIABLES

        # 1 - ORB, 2 - SIFT BRUTE-FORCE, 3 - SIFT FLANN
        self.METHOD = method

        self.PATTERNIMAGESCALE = patternImageScale
        self.SOURCEIMAGESCALE = sourceImagescale
        self.DISTANCE = distance

        self.LOGOPATTERNSPATH = 'logo'  # logo图案路径
        self.DRUGPATTERNSPATH = 'pattern'  # pattern图案路径
        self.DESTINATIONPATH = 'output'  # 目的地路径

        self.INITSHAPE = (480, 640, 3)
        self.INITDTYPE = np.uint8

        self.red = (20, 140, 255)
        self.blue = (220, 102, 20)

        # IMPORTANT VARIABLES

        print('Loading pattern images...')
        self.patternPaths = self.get_filepaths(self.DRUGPATTERNSPATH)
        print(self.patternPaths)

        self.patternImagesSetOriginal = []
        self.patternImagesSet = []

        for f in self.patternPaths:
            self.patternImagesSetOriginal.append(cv2.imread(f))

        print('Loading logo images...')
        self.logoPaths = self.get_filepaths(self.LOGOPATTERNSPATH)
        print(self.logoPaths)

        self.logoImagesSetOriginal = []
        self.logoImagesSet = []

        for f in self.logoPaths:
            self.logoImagesSetOriginal.append(cv2.imread(f))

        self.keyPointsLogoDetector = None
        self.keyPointsPatternDetector = None
        self.keyPointsMatcher = None

        if self.METHOD == Method.ORB:
            self.keyPointsLogoDetector = cv2.ORB_create(nfeatures=logoNfeatures)
            self.keyPointsPatternDetector = cv2.ORB_create(nfeatures=patternNfeatures)
            self.keyPointsMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.METHOD == Method.SIFT_BRUTE_FORCE:
            self.keyPointsLogoDetector = cv2.xfeatures2d.SIFT_create(nfeatures=logoNfeatures)
            self.keyPointsPatternDetector = cv2.xfeatures2d.SIFT_create(nfeatures=patternNfeatures)
            self.keyPointsMatcher = cv2.BFMatcher()
        elif self.METHOD == Method.SIFT_FLANN:
            self.keyPointsLogoDetector = cv2.xfeatures2d.SIFT_create(nfeatures=logoNfeatures)
            self.keyPointsPatternDetector = cv2.xfeatures2d.SIFT_create(nfeatures=patternNfeatures)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.keyPointsMatcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.resize_set()

        self.csv_input = {}
        # self.x_threshold = 0.5
        self.pattern_threshold = 0.8
        self.logo_threshold = 0.5

    def write_csv(self, file_name, writer):
        writer.writerow({'file_name': file_name,
                         'has_wenjian': self.csv_input['menu_wenjian'],
                         'has_bianji': self.csv_input['menu_bianji'],
                         'has_geshi': self.csv_input['menu_geshi'],
                         'has_chakan': self.csv_input['menu_chakan'],
                         'has_bangzhu': self.csv_input['menu_bangzhu'],
                         'has_x': self.csv_input['x'],
                         'has_tubiao': self.csv_input['notepad']})

    def __del__(self):
        cv2.destroyAllWindows()
        print('Finished!')


# ###################################################################################################################################################


# Parametry konstruktora:
# wybrana metoda z enuma (ORB, SIFT BRUTE-FORCE, SIFT-FLANN), liczba punktów kluczowych dla logo, liczba punktów kluczowych dla wzorów leków, skala dla obrazów zbioru (nie ustawiać zbyt małej skali), skala dla analizowanego obrazu (nie ustawiać zbyt małej skali), odległość punktów (integer od 0 do 100)

featureExtraction = FeatureExtraction(Method.SIFT_FLANN, 10000, 1000, 5, 2.2,
                                      80)  # logoNfeatures=10000，patternNfeatures=1000，logo是人工选出来的，所以整个图都可是有用特征，

featureExtraction.changePatternScale(8)  # 也就是上面的0.5

# SOURCESPATH = 'Source_images'
SOURCESPATH = 'test_img'
print('Loading source images...')
sourcePaths = featureExtraction.get_filepaths(SOURCESPATH)
print(sourcePaths)

sourceImagesSet = []

for f in sourcePaths:
    sourceImagesSet.append(cv2.imread(f))
print('Analyze...')

waitingMs = 0

with open(os.path.join(featureExtraction.DESTINATIONPATH, 'result.csv'), 'w', newline='') as csvfile:
    fieldnames = ['file_name', 'has_wenjian', 'has_bianji', 'has_geshi', 'has_chakan', 'has_bangzhu', 'has_x',
                  'has_tubiao']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for index, sourceImage in enumerate(sourceImagesSet):
        source_file_name = os.path.split(sourcePaths[index])[-1]
        source_name = os.path.splitext(source_file_name)[0]
        logoOutput = featureExtraction.compareWithLogo(source_name, True, True)
        # print(logoOutput[7])

        patternOutput = featureExtraction.compareWithPattern(source_name, True, True)

        featureExtraction.write_csv(source_name, writer)
        # cv2.imshow('7_Original image_', patternOutput[6])
        # cv2.imshow('6_Keypoints - image_', patternOutput[5])
        # cv2.imshow('1_Matches_', patternOutput[0])
        # cv2.imshow('4_Pattern image_', patternOutput[3])
        # cv2.imshow('5_Keypoints - pattern_', patternOutput[4])
        # cv2.imshow('2_Homography_', patternOutput[1])
        # cv2.imshow('Warped image - Logo', logoOutput[2])
        # cv2.imshow('3_Warped image_', patternOutput[2])

        # print(patternOutput[7])

        key = cv2.waitKey(waitingMs) & 0xFF;
        if key == ord('q'):
            break
        elif key == ord('a'):
            if waitingMs is 0:
                waitingMs = 1
            else:
                waitingMs = 0



del featureExtraction