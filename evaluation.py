# -*- coding: utf-8 -*-

import difflib
import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial

# Set the path to the source data (e.g. the training data for self-testing)
# and the output directory of that subject
testDir        = 'evaluation' # For example: '/input/2'
participantDir = 'evaluation' # For example: '/output/2'


labels = {1: 'Cortical gray matter',
          2: 'Basal ganglia',
          3: 'White matter',
          4: 'White matter lesions',
          5: 'Cerebrospinal fluid in the extracerebral space',
          6: 'Ventricles',
          7: 'Cerebellum',
          8: 'Brain stem',
          # The two labels below are ignored:
          #9: 'Infarction',
          #10: 'Other',
          }


def do():
    """Main function"""    
    resultFilename = getResultFilename(participantDir)  
        
    testImage, resultImage = getImages(os.path.join(testDir, 'segm.nii.gz'), resultFilename)
    
    dsc = getDSC(testImage, resultImage)
    h95 = getHausdorff(testImage, resultImage)
    vs  = getVS(testImage, resultImage)
    
    print('Dice',                dsc,       '(higher is better, max=1)')
    print('HD',                  h95, 'mm',  '(lower is better, min=0)')
    print('VS',                   vs,       '(higher is better, max=1)')
    
    
    
def getResultFilename(participantDir):
    """Find the filename of the result image.
    
    This should be result.nii.gz or result.nii. If these files are not present,
    it tries to find the closest filename."""
    files = os.listdir(participantDir)
    
    if not files:
        raise Exception("No results in "+ participantDir)
    
    resultFilename = None
    if 'result.nii.gz' in files:
        resultFilename = os.path.join(participantDir, 'result.nii.gz')
    elif 'result.nii' in files:
        resultFilename = os.path.join(participantDir, 'result.nii')
    else:
        # Find the filename that is closest to 'result.nii.gz'
        maxRatio = -1
        for f in files:
            currentRatio = difflib.SequenceMatcher(a = f, b = 'result.nii.gz').ratio()
            
            if currentRatio > maxRatio:
                resultFilename = os.path.join(participantDir, f)
                maxRatio = currentRatio
                
    return resultFilename
    

def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and pathology masked."""
    testImage   = sitk.ReadImage(testFilename)
    resultImage = sitk.ReadImage(resultFilename)
    
    # Check for equality
    assert testImage.GetSize() == resultImage.GetSize()
    
    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)    
    
    # Remove pathology from the test and result images, since we don't evaluate on that
    pathologyImage = sitk.BinaryThreshold(testImage, 9, 11, 0, 1)  # pathology == 9 or 10
    
    maskedTestImage   = sitk.Mask(testImage,   pathologyImage)     # tissue    == 1 --  8
    maskedResultImage = sitk.Mask(resultImage, pathologyImage)
    
    # Force integer
    if not 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        maskedResultImage = sitk.Cast(maskedResultImage, sitk.sitkUInt8)
            
    return maskedTestImage, maskedResultImage
    
    
def getDSC(testImage, resultImage):    
    """Compute the Dice Similarity Coefficient."""        
    dsc = dict()
    for k in labels.keys():
        testArray   = sitk.GetArrayFromImage(sitk.BinaryThreshold(  testImage, k, k, 1, 0)).flatten()
        resultArray = sitk.GetArrayFromImage(sitk.BinaryThreshold(resultImage, k, k, 1, 0)).flatten()
        
        # similarity = 1.0 - dissimilarity
        # scipy.spatial.distance.dice raises a ZeroDivisionError if both arrays contain only zeros.
        try:
            dsc[k] = 1.0 - scipy.spatial.distance.dice(testArray, resultArray)
        except ZeroDivisionError:
            dsc[k] = None
    
    return dsc

        
def getHausdorff(testImage, resultImage):
    """Compute the 95% Hausdorff distance."""    
    hd = dict()
    for k in labels.keys():
        lTestImage   = sitk.BinaryThreshold(  testImage, k, k, 1, 0)
        lResultImage = sitk.BinaryThreshold(resultImage, k, k, 1, 0)
        
        # Hausdorff distance is only defined when something is detected
        statistics = sitk.StatisticsImageFilter()
        statistics.Execute(lTestImage)
        lTestSum = statistics.GetSum()
        statistics.Execute(lResultImage)
        lResultSum = statistics.GetSum()
        if lTestSum == 0 or lResultSum == 0:
            hd[k] = None
            continue
                                
        # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
        eTestImage   = sitk.BinaryErode(lTestImage, (1,1,0))
        eResultImage = sitk.BinaryErode(lResultImage, (1,1,0))
        
        hTestImage   = sitk.Subtract(lTestImage, eTestImage)
        hResultImage = sitk.Subtract(lResultImage, eResultImage)    
        
        hTestArray   = sitk.GetArrayFromImage(hTestImage)
        hResultArray = sitk.GetArrayFromImage(hResultImage)   
            
        # Convert voxel location to world coordinates. Use the coordinate system of the test image
        # np.nonzero   = elements of the boundary in numpy order (zyx)
        # np.flipud    = elements in xyz order
        # np.transpose = create tuples (x,y,z)
        # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
        # (Simple)ITK does not accept all Numpy arrays; therefore we need to convert the coordinate tuples into a Python list before passing them to TransformIndexToPhysicalPoint().
        testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
        resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]
                
        # Use a kd-tree for fast spatial search
        def getDistancesFromAtoB(a, b):    
            kdTree = scipy.spatial.KDTree(a, leafsize=100)
            return kdTree.query(b, k=1, eps=0, p=2)[0]
        
        # Compute distances from test to result and vice versa. 
        dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
        dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)
        hd[k] = max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
        
    return hd


def getVS(testImage, resultImage):   
    """Volume similarity.
    
    VS = 1 - abs(A - B) / (A + B)
    
    A = ground truth in ML
    B = participant segmentation in ML
    """    
    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    
    vs = dict()
    for k in labels.keys():
        testStatistics.Execute(sitk.BinaryThreshold(testImage, k, k, 1, 0))
        resultStatistics.Execute(sitk.BinaryThreshold(resultImage, k, k, 1, 0))
        
        numerator = abs(testStatistics.GetSum() - resultStatistics.GetSum())
        denominator = testStatistics.GetSum() + resultStatistics.GetSum()               
        
        if denominator > 0:        
            vs[k] = 1 - float(numerator) / denominator
        else:
            vs[k] = None
        
    return vs
    
    
if __name__ == "__main__":
    do() 
