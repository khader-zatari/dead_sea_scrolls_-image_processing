from FileReader import FileReader 
from SegPrep import SegPrep
import skimage.io as io

def main():
    imreader = FileReader("./images","*.jpg")
    imList = imreader.getImages()
    prep = SegPrep(imList)

    prep.itrSmoothing(itrations=2)
    prep.colorSpacePyramid()
    prep.morphEadgeDetection()
    prep.findBorders()
    prep.floodFill()
    prep.clearBoarders()
    prep.reginalfilling()
    prep.clearBoarders(buffer_size=220)
    prep.label()
    
    regionTable = prep.getRegionProps()
    validregionTable = prep.getValidRegions(regionTable,minArea=10000)
    images = prep.validregionImage(validregionTable)
    prep.drawcontor(imageList = images)
    contorsTable = prep.findContorsTable(imageList=images)
    prep.dataframeIntoExcel(contorsTable,"projectIM2022_q2")



   
    
if __name__ == "__main__":
    main()
