from tkinter import image_names
import numpy as np
import skimage.io as io
import skimage.transform as ts
class FileReader:

    filesDir:str
    fileFormat:str
    def __init__(self,filesDir:str,fileFormat:str):
        
        self.filesDir = filesDir
        self.fileFormat = fileFormat

    def getImages(self):

        readCol = io.imread_collection(self.filesDir+"/"+self.fileFormat)        
        npCol = []
        for i in range(len(readCol)):
            npCol.append(np.array(readCol[i],dtype=np.uint8))
        return npCol

    def getImagesRescaled(self,rescale_factro:float):

        imageList = self.getImages()
        
        for i in range(len(imageList)):
            imageList[i] = ts.rescale(imageList[i],rescale_factro)

        return imageList