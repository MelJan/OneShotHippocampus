import numpy as numx
import scipy as sc

def seperateDataByLabelsForMNISTandCIFAR10(data, label, status = False):

    zeros = None
    ones = None
    twos = None
    threes = None
    fours = None
    fives = None
    sixes = None
    sevens = None
    eights = None
    nines = None

    for i in range(data.shape[0]):
        
        if status == True:
            print '%3.2f' % (100.0*numx.double(i)/numx.double(data.shape[0])), '%'
            
        if label[i] == 0:
            if zeros == None:
                zeros = data[i,:].reshape(1, data.shape[1])    
            else:
                zeros = numx.vstack((zeros,data[i,:].reshape(1, data.shape[1])))        
        if label[i] == 1:
            if ones == None:
                ones = data[i,:].reshape(1, data.shape[1])    
            else:
                ones = numx.vstack((ones,data[i,:].reshape(1, data.shape[1])))    
        if label[i] == 2:
            if twos == None:
                twos = data[i,:].reshape(1, data.shape[1])    
            else:
                twos = numx.vstack((twos,data[i,:].reshape(1, data.shape[1])))        
        if label[i] == 3:
            if threes == None:
                threes = data[i,:].reshape(1, data.shape[1])    
            else:
                threes = numx.vstack((threes,data[i,:].reshape(1, data.shape[1])))  
        if label[i] == 4:
            if fours == None:
                fours = data[i,:].reshape(1, data.shape[1])    
            else:
                fours = numx.vstack((fours,data[i,:].reshape(1, data.shape[1])))        
        if label[i] == 5:
            if fives == None:
                fives = data[i,:].reshape(1, data.shape[1])    
            else:
                fives = numx.vstack((fives,data[i,:].reshape(1, data.shape[1])))   
            
        if label[i] == 6:
            if sixes == None:
                sixes = data[i,:].reshape(1, data.shape[1])    
            else:
                sixes = numx.vstack((sixes,data[i,:].reshape(1, data.shape[1])))    
        if label[i] == 7:
            if sevens == None:
                sevens = data[i,:].reshape(1, data.shape[1])    
            else:
                sevens = numx.vstack((sevens,data[i,:].reshape(1, data.shape[1])))        
        if label[i] == 8:
            if eights == None:
                eights = data[i,:].reshape(1, data.shape[1])    
            else:
                eights = numx.vstack((eights,data[i,:].reshape(1, data.shape[1])))  
        if label[i] == 9:
            if nines == None:
                nines = data[i,:].reshape(1, data.shape[1])    
            else:
                nines = numx.vstack((nines,data[i,:].reshape(1, data.shape[1])))  
               
    return zeros,ones,twos,threes,fours,fives,sixes,sevens,eights,nines   

def ExpanedByRotatedVersions(data, numberRotations = 8, status = False):
    
    stepsize = 360.0 / numberRotations
    rotations = numx.arange(stepsize,360-stepsize,stepsize)
    print rotations
    if status == True:
        print '%3.2f' % (0.0), '%'
    img = data[0].reshape(28,28)
    set = img.reshape(1,784)
    for rot in rotations:
        set = numx.vstack((set,sc.misc.imrotate(img,rot).reshape(1,784)))
    for i in range(1,data.shape[0]):
        if status == True:
            print '%3.2f' % (100.0*numx.double(i)/numx.double(data.shape[0])), '%'
        img = data[i].reshape(28,28)
        setc = img.reshape(1,784)
        for rot in rotations:
            setc = numx.vstack((setc,sc.misc.imrotate(img,rot).reshape(1,784)))
        set = numx.vstack((set,setc))
    return set