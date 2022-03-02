from array import ArrayType
import re
from skimage.io import imread 
from skimage import exposure
from skimage.transform import resize
import skimage.filters.rank as rank
from skimage.morphology import disk , binary
import skimage.morphology as mp
from skimage.feature import hog,canny
import skimage.segmentation as sg
import numpy as np
import cv2
import skimage.measure as me
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import skimage.io as io
from skimage.color import rgb2gray
from pathlib import Path
from openpyxl import load_workbook
from skimage import measure

class SegPrep:

    orignalImages:ArrayType
    smoothedImages:ArrayType
    hogImages:ArrayType
    binaryImages:ArrayType
    qountImages:ArrayType
    morphedImages:ArrayType
    cannyImages:ArrayType
    borderImgaes:ArrayType
    filledImgaes:ArrayType
    clearBorderImgaes:ArrayType
    prevResults:ArrayType
    reginlaFillingImgaes:ArrayType
    

    def __init__(self,orignalImages:ArrayType):
        self.orignalImages = orignalImages
        self.prevResults = orignalImages.copy()
    
    def reginalfilling(self,imageList=None):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.reginlaFillingImgaes = imageList.copy()
        self.floodFill(imageList = imageList.copy())

        for i in range(len(self.reginlaFillingImgaes)):
            invrimage = np.where(self.prevResults[i]>0,0,255)
            self.reginlaFillingImgaes[i] = np.add(invrimage,self.reginlaFillingImgaes[i])

        self.prevResults = self.reginlaFillingImgaes.copy()
        return self.prevResults.copy()
             
    def itrSmoothing(self,imageList=None ,kernalSize = 5,itrations = 1):
        
        if(imageList == None):
            imageList = self.prevResults.copy() 
        self.smoothedImages = imageList.copy()
        
        for i in range(len(self.smoothedImages)):            
            for j in range(itrations):
                self.smoothedImages[i] = cv2.blur(self.smoothedImages[i],(kernalSize,kernalSize),cv2.CV_32FC2)
        
        self.prevResults = self.smoothedImages.copy()
        return self.prevResults.copy()

    def changeColorSpace(self,imageList = None , colorReagons = 8):

        if(imageList == None):
            imageList = self.prevResults.copy()

        self.qountImages = imageList.copy()
        
        for i in range(len(imageList)):
            max = imageList[i].max()
            min = imageList[i].min()
            d = max - min
            regonsOffset = (int)(d/colorReagons)
            for j in range(colorReagons+1):
                imageList[i] = imageList[i].astype(np.uint8)
                currVal = regonsOffset*j
                imageList[i] = np.where( ((imageList[i]>currVal) & (imageList[i]<=currVal+regonsOffset)),currVal,imageList[i] )
        self.qountImages = imageList
        self.prevResults = self.qountImages.copy()
        return self.prevResults.copy()

    def morphEadgeDetection(self,imageList = None,openingKernal=None,closingKernal=None,dilatKernal=None,erodeKernal=None):

        if(imageList == None):
            imageList = self.prevResults.copy()
        self.morphedImages = imageList.copy()

        if(closingKernal == None):
            closingKernal = disk(5)
        if(openingKernal == None):
            openingKernal = disk(3)
        if(erodeKernal == None):
            erodeKernal = disk(3)
        if(dilatKernal == None):
            dilatKernal = disk(3)
        for i in range (len(self.morphedImages)):
            # closing = self.morphedImages[i]
            
            # close = mp.closing(self.morphedImages[i],selem=closingKernal)
            erode = mp.erosion(self.morphedImages[i],selem=  erodeKernal).astype(np.uint8)
            erode = mp.erosion(erode,selem= erodeKernal).astype(np.uint8)
            dilate = mp.erosion(self.morphedImages[i],selem= dilatKernal,).astype(np.uint8)
            edge = dilate - erode

            # edge = mp.opening(edge).astype(np.uint8)
            self.morphedImages[i] = edge
        self.prevResults = self.morphedImages.copy()
        return self.prevResults.copy()

    def findBorders(self,imageList = None , connectivity=1, mode='thick', background=0):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.borderImgaes = imageList.copy()
        for i in range(len(self.borderImgaes)):
            self.borderImgaes[i] = sg.find_boundaries(self.borderImgaes[i],connectivity=connectivity,mode=mode,background=background).astype(np.uint8)*255
        self.prevResults = self.borderImgaes.copy()
        return self.prevResults.copy()
                
    def floodFill(self, imageList = None , seed_point=(0,0) , new_value=255 , selem=None, connectivity=None, tolerance=None, in_place=False, inplace=None):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.filledImgaes = imageList.copy()
        for i in range(len(self.filledImgaes)):
            if seed_point[0] < 0 or seed_point[1] < 0:
                new_seed =np.array(self.filledImgaes[i].shape)
                new_seed -= 1

                self.filledImgaes[i] = sg.flood_fill(self.filledImgaes[i],tuple(new_seed),new_value)
            else:    
                self.filledImgaes[i] = sg.flood_fill(self.filledImgaes[i],seed_point,new_value)
        self.prevResults = self.filledImgaes.copy()
        return self.prevResults.copy()
        
    def clearBoarders(self , imageList = None ,  buffer_size=0, bgval=0, in_place=False, mask=None):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.clearBorderImgaes = imageList.copy()
        for i in range(len(self.clearBorderImgaes)):
            self.clearBorderImgaes[i] = sg.clear_border(self.clearBorderImgaes[i],buffer_size=buffer_size,bgval=bgval,in_place=in_place,mask=mask)
        self.prevResults = self.clearBorderImgaes.copy()
        return self.prevResults.copy()

    def erode(self , imageList = None ,disk_size = 4):
        if (imageList == None):
            imageList = self.prevResults.copy()
        for i in range(len(imageList)):
            self.prevResults[i] = mp.erosion(imageList[i], selem=disk(disk_size)).astype(np.uint8)
        return self.prevResults.copy()

    def open(self, imageList=None, disk_size=4):
        if (imageList == None):
            imageList = self.prevResults.copy()

        for i in range(len(imageList)):
            self.prevResults[i] = mp.opening(imageList[i], selem=disk(disk_size)).astype(np.uint8)
        return self.prevResults.copy()

    def close(self, imageList=None, disk_size=4):
        if (imageList == None):
            imageList = self.prevResults.copy()

        for i in range(len(imageList)):
            self.prevResults[i] = mp.closing(imageList[i], selem=disk(disk_size)).astype(np.uint8)
        return self.prevResults.copy()

    def label(self, imageList=None):
        if (imageList == None):
            imageList = self.prevResults.copy()

        for i in range(len(imageList)):
            self.prevResults[i] = me.label(imageList[i])
        return self.prevResults.copy()

    def sortRegionsInImage(self , regionTable):

        dataFrames = []

        for i in range(len(regionTable)):

            dict = {'up left X': [], 'up left Y': [], 'down right X': [], 'down right Y': [],"distance": []}
            df = pd.DataFrame(data=dict)
            
            for index, region in enumerate(regionTable[i]):
            
                    minr, minc, maxr, maxc = region.bbox
            
                    distance_from_zero = ((minr) ** 2 + (minc) ** 2) ** (1 / 2)
            
                    df.loc[-1] = [minr , minc , maxr , maxc ,int(distance_from_zero)]  # adding a row
            
                    df.index = df.index + 1  # shifting index
            
                    df = df.sort_index()
            
            df = df.sort_values("distance")
            df.insert(0, 'ID', range(1, 1 + len(df)))
            dataFrames.append(df)
        
        return dataFrames

    def getRegionProps(self,imageList = None):
        
        if (imageList == None):
            imageList = self.prevResults.copy()
        regionsTable = []

        for i in range(len(imageList)):
            region = me.regionprops(imageList[i])
            regionsTable.append(region)
        
        return regionsTable
        
    def getValidRegions(self, regionTable , minArea = 15000):

        validRegionTable = []
        
        for i in range(len(regionTable)):

            validRegions = []

            for _,region in enumerate(regionTable[i]):
            
                if region.area >= minArea:

                    minr, minc, maxr, maxc = region.bbox
            
                    if ((maxc - minc) / (maxr - minr)) > 5 or (maxr - minr) /(maxc - minc) > 4.5:# aspect ratio distorted
                        continue
                    
                    validRegions.append(region)
            
            validRegionTable.append(validRegions)
        
        return validRegionTable

    def drawRec(self,  sortedBoxTable, imageList=None ):
        

        for i,table in enumerate(sortedBoxTable):

            fig,ax = plt.subplots(figsize = (10,6))
            ax.imshow(self.orignalImages[i], cmap=plt.cm.gray)

            for j in range(len(table)): 
                
                currRow =table.loc[table["ID"] == j+1].iloc[0]

                minc = currRow["up left Y"]
                minr = currRow["up left X"]
                maxc = currRow["down right Y"]
                maxr = currRow["down right X"]

                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

                row_center = minr + (maxr - minr) // 2
                column_center = minc + (maxc - minc) // 2
                ax.annotate(j+1, (column_center , row_center), color='w', weight='bold',fontsize=6, ha='center', va='center')

            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig("./results/image%i.jpg"%(i))
            plt.show()

    def cropRecFromPhoto(self,dataFrameTable,imageList = None):
        if (imageList == None):
            imageList = self.orignalImages.copy()
        for i in range(len(imageList)):
            currImage = imageList[i]
            currDataFrame = dataFrameTable[i]
            
            
            for j in range(len(currDataFrame)): 
                
                row =currDataFrame.loc[currDataFrame["ID"]==j+1].iloc[0]


                minc = row["up left Y"]
                minr = row["up left X"]
                maxc = row["down right Y"]
                maxr = row["down right X"]
                
                cropSlice = np.zeros(currImage.shape)
                
                for r in range(minr,maxr):
                    for c in range(minc,maxc):
                        cropSlice[r,c] = currImage[r,c]
                
                io.imsave("./results/image%d/slice%d.jpg"%(i,j+1),cropSlice)

    def colorSpacePyramid(self,imageList = None , pyramid = [100,50,25,10,8,5,4]):

        if(imageList == None):

            imageList = self.prevResults.copy()

        for i in range(len(pyramid)):

            self.prevResults = self.changeColorSpace(imageList=imageList,colorReagons = pyramid[i])
        
        return self.prevResults.copy()
    
    def dataframeIntoCsv(self ,dataframe ,  name):
        dataframe.to_csv('excel/{}.csv'.format(name), index=False, header=False)

    def dataframeIntoExcel(self, dataframeTable , fileName):
        path = "excel/{}.xlsx".format(fileName)
        my_file = Path('excel/{}.xlsx'.format(fileName))
        for index , dataframe in enumerate(dataframeTable):
            if my_file.is_file():
                book = load_workbook(path)
                writer = pd.ExcelWriter(path, engine='openpyxl')
                writer.book = book
                dataframe.to_excel(writer, sheet_name=fileName+""+str(index), index =False)
                writer.save()
                writer.close()
            else:
                dataframe.to_excel(path ,sheet_name =fileName+""+str(index) , index =False )

    def validregionImage(self, validregionTable,imageList=None ):
        if (imageList == None):
            imageList = self.prevResults.copy()
        imList = [None] *  len(validregionTable)
        for i in range(len(validregionTable)):
            imList[i] = np.zeros(self.orignalImages[i].shape)
            for j in  validregionTable[i]:
                for x, y in j.coords:
                    imList[i][x][y] = 255
        return imList

    def drawcontor(self ,imageList=None):
        if (imageList == None):
            imageList = self.prevResults.copy()
        for index , i in  enumerate(imageList):
            contours = measure.find_contours(i, 0.9)
            fig, ax = plt.subplots()
            ax.imshow(self.orignalImages[index], cmap=plt.cm.gray)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig("./results/image%i.jpg" % (index))
            plt.show()

    def dropColumn(self ,dataframeTable, columnName="distance"):
        for index , df in enumerate(dataframeTable):
            dataframeTable[index] = df.drop(columns=['distance'])
        return dataframeTable

    def findContorsTable(self, imageList=None):
        if (imageList == None):
            imageList = self.prevResults.copy()
        contorsTable = []
        for index, i in enumerate(imageList):
            contours = measure.find_contours(i, 0.9)
            df = pd.DataFrame(contours, columns=['contours'])
            df.insert(0, 'ID', range(0,len(df)))
            contorsTable.append(df)
        return contorsTable
