import scipy.io
import numpy as np
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
import hdbscan
from sklearn import preprocessing


def trimSeismicAmplitudes(seismicCube, percentile):
    temp=seismicCube.copy()
    temp=temp.flatten()
    temp[temp<0]=0
    temp[temp != 0]
    pos_cutoff = np.percentile(temp, percentile) # return percentile
    temp=seismicCube.copy()
    temp=temp*-1
    temp=temp.flatten()
    temp[temp<0]=0
    temp[temp != 0]
    neg_cutoff = np.percentile(temp, percentile) # return percentile
    print ('new min and max are:', neg_cutoff, pos_cutoff)
    seismicCube[seismicCube>pos_cutoff]=pos_cutoff
    seismicCube[seismicCube<-neg_cutoff]=-neg_cutoff
    return seismicCube


#window_size=[50, 50, 50]
#step=10
import time
def generateFeaturevectors (seismicCube, textureCube, window_size, step):
    
    start = time.time()
    
    datashape=textureCube.shape
    edge_r=np.int(window_size[0]/2- step/2)
    edge_c=np.int(window_size[1]/2- step/2)
    edge_d=np.int(window_size[2]/2- step/2)
    minseis=np.min(seismicCube)
    maxseis=np.max(seismicCube)
    P=37
    
    layercake = np.zeros(seismicCube.shape)
    size_x = seismicCube.shape[0]
    for x in range (0, size_x):
        layercake[x, :, :]=x 
    
    feature_vectors=[]
    for r in range(0,datashape[0] - window_size[0], step):
        for c in range(0,datashape[1] - window_size[1], step):
            for d in range(0,datashape[2] - window_size[2], step):
                histograms=[]

                window = seismicCube[r:r+window_size[0],c:c+window_size[1], d:d+window_size[2]]        
                hist, _ = np.histogram(window, bins=P, range=(minseis, maxseis))
                histograms.append(hist)

                window = textureCube[r:r+window_size[0],c:c+window_size[1], d:d+window_size[2]]        
                hist, _ = np.histogram(window, bins=P, range=(0, 36))
                histograms.append(hist)

                feature_vector=np.concatenate(histograms)
                layercakewindow = layercake[r:r+window_size[0],c:c+window_size[1], d:d+window_size[2]]
                depth=np.int(np.mean(layercakewindow))
                feature_vector_depthconstrained=feature_vector+depth
                feature_vector_depthconstrained=feature_vector_depthconstrained.astype(np.float)
                feature_vectors.append(feature_vector_depthconstrained)
                
    feature_vectors = preprocessing.scale(feature_vectors)
    
    
    stop = time.time()
    t=(stop-start)/60
    print('generate feature vectors - time in minutes: ', t)
    
    return feature_vectors
          


# min_cluster_size=10000
# datashape=seismicCube.shape
def IdentifyStratUnits(feature_vectors, min_cluster_size, datashape, window_size, step):
    start = time.time()
    iteration=1
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size, min_samples=1, prediction_data=True)
    cluster_labels = clusterer.fit_predict(feature_vectors)
    
    #clusterer.condensed_tree_.plot()
    #plt.show()
    
    A=cluster_labels.copy()
    edge_r=np.int(window_size[0]/2- step/2)
    edge_c=np.int(window_size[1]/2- step/2)
    edge_d=np.int(window_size[2]/2- step/2)
    A=cluster_labels.copy()
    segmented=np.zeros(datashape, dtype=np.uint64)
    num=0
    print('iteration', iteration)
    for r in range(0,datashape[0] - window_size[0], step):
        for c in range(0,datashape[1] - window_size[1], step):
            for d in range(0,datashape[2] - window_size[2], step):
                segmented[r+edge_r:r+step+edge_r,c+edge_c:c+step+edge_c, d+edge_d:d+step+edge_d]=A[num]+1
                num=num+1
                
    unique_labels=A+1
    
    stop = time.time()
    t=(stop-start)/60
    print('identify units - time in minutes: ', t)
    return segmented, unique_labels



def FindNewUnits (firstIteration, splitlabel, seismicCube, textureCube, window_size, step, min_cluster_size):
    start = time.time()
    datashape=firstIteration.shape
    edge_r=np.int(window_size[0]/2- step/2)
    edge_c=np.int(window_size[1]/2- step/2)
    edge_d=np.int(window_size[2]/2- step/2)
    minseis=np.min(seismicCube)
    maxseis=np.max(seismicCube)
    minseis=np.min(seismicCube)
    maxseis=np.max(seismicCube)
    layercake = np.zeros(seismicCube.shape)
    size_x = seismicCube.shape[0]
    for x in range (0, size_x):
        layercake[x, :, :]=x 
    feature_vectors=[]
    for r in range(0,datashape[0] - window_size[0], step):
        for c in range(0,datashape[1] - window_size[1], step):
            for d in range(0,datashape[2] - window_size[2], step):
                #check if already classified    
                window = firstIteration[r+edge_r:r+step+edge_r,c+edge_c:c+step+edge_c, d+edge_d:d+step+edge_d]
                if window[0, 0, 0] == splitlabel:
                    histograms=[]

                    window = seismicCube[r:r+window_size[0],c:c+window_size[1], d:d+window_size[2]]        
                    hist, _ = np.histogram(window, bins=37, range=(minseis, maxseis))
                    histograms.append(hist)

                    window = textureCube[r:r+window_size[0],c:c+window_size[1], d:d+window_size[2]]        
                    hist, _ = np.histogram(window, bins=37, range=(0, 36))
                    histograms.append(hist)

                    feature_vector=np.concatenate(histograms)
                    layercakewindow = layercake[r:r+window_size[0],c:c+window_size[1], d:d+window_size[2]]
                    depth=np.int(np.mean(layercakewindow))
                    feature_vector_depthconstrained=feature_vector+depth
                    feature_vector_depthconstrained=feature_vector_depthconstrained.astype(np.float)
                    feature_vectors.append(feature_vector_depthconstrained)
    new_units=np.zeros(datashape, dtype=np.uint64)
    print(len(feature_vectors))
    
    if len(feature_vectors) > 0:
        feature_vectors = preprocessing.scale(feature_vectors)
        clusterer = hdbscan.HDBSCAN(min_cluster_size, min_samples=1, allow_single_cluster=False)
        cluster_labels = clusterer.fit_predict(feature_vectors)
        
        #clusterer.condensed_tree_.plot()
        #plt.show()
        
        A=cluster_labels.copy()
        num=0
        for r in range(0,datashape[0] - window_size[0], step):
            for c in range(0,datashape[1] - window_size[1], step):
                for d in range(0,datashape[2] - window_size[2], step):
                    window = firstIteration[r+edge_r:r+step+edge_r,c+edge_c:c+step+edge_c, d+edge_d:d+step+edge_d]
                    if window[0, 0, 0] == splitlabel:
                        new_units[r+edge_r:r+step+edge_r,c+edge_c:c+step+edge_c, d+edge_d:d+step+edge_d]=A[num]+1
                        num=num+1
                        
    stop = time.time()
    t=(stop-start)/60
    print('new iteration - time in minutes: ', t)
    return new_units



def iteratively_identify_stratUnits(seismicCube, textureCube, window_size, step, min_cluster_fraction):
    print('generate feature vectors....')
    feature_vectors=generateFeaturevectors(seismicCube, textureCube, window_size, step)
    min_cluster_size=np.int(len(feature_vectors)/min_cluster_fraction)
    print('minimum cluster size is..', min_cluster_size)
    
    iterations=[]

    #min_cluster_size=np.int(len(feature_vectors)/25)
    print('feature vectors', len(feature_vectors))
    print('min_cluster_size', min_cluster_size)


    datashape=seismicCube.shape
    print('HDBSCAN.......')
    segmented, unique_labels= IdentifyStratUnits(feature_vectors, min_cluster_size, datashape, window_size, step)
    iterations.append(segmented)
    
    print('first iteration....')
    plt.imshow(segmented[:,50,:], aspect='auto')
    plt.colorbar()
    plt.show()

    for i in np.unique(unique_labels):
        x = np.count_nonzero(unique_labels == i)
        percent=x/len(unique_labels)
        if percent > 0.80:
            print('Second iteration. split label:', i)
            splitlabel=i

            segmented2=FindNewUnits(segmented, splitlabel, seismicCube, textureCube, window_size, step, min_cluster_size)
            plt.imshow(segmented2[:,50,:], aspect='auto')
            plt.colorbar()
            plt.show()

            segmented1=segmented.copy()
            segmented1[segmented1==i]=0
            start=np.max(segmented2)
            for j in range (1, np.int(np.max(segmented1))+1):
                segmented1[segmented1==j]=start+j
                
            segmented=segmented1+segmented2
            iterations.append(segmented)
            #plt.imshow(segmented[:,50,:], aspect='auto')
            #plt.colorbar()
            #plt.show()

    if segmented.any():
        splitlabel=0
        new_units=FindNewUnits(segmented, splitlabel, seismicCube, textureCube, window_size, step, min_cluster_size)
        start=np.max(segmented)
        for i in range (1, np.int(np.max(new_units))+1):
            new_units[new_units==i]=start+i
        final=new_units+segmented
        print('Final iteration')
        plt.imshow(final[:,50,:], aspect='auto')
        plt.colorbar()
        plt.show()
        iterations.append(final)
        print(len(iterations), 'cubes are returned')
    return iterations





