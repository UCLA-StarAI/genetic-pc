import numpy as np 
import pandas as pd
import sklearn.metrics
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial import distance
import os.path


def computeAAandDist(dat, samplelabel, categ=None, refCateg='Real', saveAllDist=False, outDir='./', outFilePrefix='', reloadDTT=None):
    """
    Compute AATS scores
    Args:
        dat           pd.dataframe of the genetic data with individuals as rows, and SNPs as columns
        samplelabel   array of label ('Real', 'RBM_it690', etc) for each individual of dat (ie each line of the dataset)
        categ         list of category of datasets to investigate, eg ['Syn'] or ['SYn', 'RBM']. 
                      Default: all categories available in dat
                      
    Returns:
        a tuple (AA, closest_all)
        AA (pd.dataframe):    contains AATS score for each investigated category
        DIST (pd.dataframe):   contains the distances to closest neighbor for different categories and pairs (Real, Real) (Real, Syn)  (Syn, Real), (Syn, Syn)

    
    Example:
        AA, DIST = computeAATS(dat, samplelabel, ['Syn','RBM'])

"""
    
    if (not refCateg is None) and (not refCateg in np.unique(samplelabel)): 
        print(f'Error: It is mandatory to have individuals labeled as your reference categ {refCateg}',
              ' in your dataset. You can change the reference label using the refCateg argument')
        return
        
    if categ is None:
        categ = np.unique(samplelabel)    

    nReal = (samplelabel==refCateg).sum()
    AA = pd.DataFrame(columns=['cat','AATS','AAtruth','AAsyn', 'ref'])
    DIST = pd.DataFrame(columns=['cat','dTS','dST','dSS'])
    
    # reference-reference distance
    DTTfilename = f'{outDir}/{reloadDTT}'
    if (reloadDTT is None) or (not os.path.exists(DTTfilename)):
        dAB = distance.cdist(dat.loc[samplelabel.isin([refCateg]),:], 
                             dat.loc[samplelabel.isin([refCateg]),:], 'cityblock')
        if not reloadDTT is None:
            print(f"DTT was not yet saved: we computed it and saved it to {DTTfilename}")
            np.savez_compressed(DTTfilename, DTT=dAB)
    else:
        dAB = np.load(DTTfilename)['DTT']

    if saveAllDist:
        np.savez_compressed(f'{outDir}/{outFilePrefix}dist_{refCateg}_{refCateg}',
                            dist=dAB[np.triu_indices(dAB.shape[0], k=1)])
    
    np.fill_diagonal(dAB, np.Inf)
    dTT = dAB.min(axis=1)
    DTMP = pd.DataFrame({'cat':refCateg,'dTS':dTT,'dST':dTT,'dSS':dTT})
    DIST = DIST.append(DTMP, ignore_index=True) 
    
    for cat in categ:
        print('>>> computeAAandDist <<<', 'cat is', cat)
        if cat==refCateg: 
            continue
            
        ncat = (samplelabel==cat).sum()
        if ncat==0:
            print(f'Warning: no individuals labeled {cat} were found in your dataset.',
                  ' Jumping to the following category')
            continue  
            
        if  ncat != nReal:
                print(f'Warning: nb of individuals labeled {cat} differs from the the nb in refCateg.',
                      ' Jumping to the following category')
        print(cat)           
        dAB = distance.cdist(dat.loc[samplelabel.isin([cat]),:], dat.loc[samplelabel.isin([refCateg]),:], 'cityblock')
        if saveAllDist:
            np.savez_compressed(f'{outDir}/{outFilePrefix}dist_{cat}_{refCateg}', dist=dAB.reshape(-1))

        dST = dAB.min(axis=1)  # dST
        dTS = dAB.min(axis=0)  #dTS
        dAB = distance.cdist(dat.loc[samplelabel.isin([cat]),:], dat.loc[samplelabel.isin([cat]),:], 'cityblock')
        if saveAllDist:
            np.savez_compressed(f'{outDir}/{outFilePrefix}dist_{cat}_{cat}',
                                dist=dAB[np.triu_indices(dAB.shape[0], k=1)])
        
        np.fill_diagonal(dAB, np.Inf)
        dSS = dAB.min(axis=1)  #dSS
        n = len(dSS)
        AAtruth = ((dTS > dTT)/n).sum()
        AAsyn = ((dST > dSS)/n).sum()
        AATS = (AAtruth + AAsyn)/2      
        AA = AA.append({'cat':cat, 'AATS':AATS, 'AAtruth':AAtruth, 'AAsyn':AAsyn, 'ref':refCateg}, ignore_index=True)  
        print('cat', cat)
        print('dTS', dTS.shape)
        print('dST', dST.shape)
        print('dSS', dSS.shape)
        DTMP_dict = {'cat':cat,'dTS':dTS,'dST':dST,'dSS':dSS}
        DTMP = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in DTMP_dict.items()]))
        DIST = DIST.append(DTMP, ignore_index=True) 
                 
    return(AA, DIST)
    
    
# not used in the paper
def mapAATS(focuscat, pcdf, closest_all, AA,  numcols=300, numrows=300, vmin=0, vmax=1, method='cubic'):
    """
    Interpolate AATS terms and plot on PCA maps

    Args:
        focuscat (string) :     label of generated data (eg 'Syn', 'RBM' etc)
        pcdf (pd.Dataframe) :   matrix of size nindiv x (PC) components. The two first columns 
                                    are used as the original sampling points for the interpolation map
        closest_all :           object returned by computeAATS
        AA :                    object returned by computeAATS
        numcols (int) :         column resolution for interpolation
        numrows (int) :         row resolution for interpolation
        vmin (float) :          minimum threshold value for colorscale in imshow
        vmax (float) :          maximum threshold value for colorscale in imshow
        method (string) :       'gp' for gaussian process or method to pass to griddata (eg 'cubic')
    
    Returns:
        None
    
    Example :         
        pca = PCA(n_components=6)
        pcs = pca.fit_transform(dat)
        pcdf = pd.DataFrame(pcs, columns=["PC{}".format(x) for x in np.arange(pcs.shape[1])])
        
        plt.figure(figsize=(20,10))
        mapAATS('Syn', closest_all, AA)
    """
    
    
    closest = closest_all[focuscat]
    n = int(closest.shape[0]/2)
    correctly_classified = (closest>=n)
    i=1

    for cat in ['Real', focuscat]:

        rate = float((AA.loc[AA['cat']==focuscat,'AAtruth'] if cat=='Real' else  AA.loc[AA['cat']==focuscat,'AAsyn']))
        if cat == 'Real':
            z =  correctly_classified.astype(int)[:n]
        else:
            z =  correctly_classified.astype(int)[n:]

        points =  np.asarray(pcdf[pcdf.label==cat].iloc[:,0:2])
        plt.subplot(2,2,i)
        i+=1
        plt.scatter(x = points[:,0],
                    y = points[:,1],
                    c = z,
                    alpha=.7, cmap='jet', vmin=vmin, vmax=vmax,
                   )
        plt.title("{} individuals correctly classified at rate {} ".format(cat, rate))
        plt.colorbar()
        
        plt.subplot(2,2,i)
        i+=1
       
        xi = np.linspace( points[:,0].min()-1,  points[:,0].max()+1, numcols)
        yi = np.linspace( points[:,1].min()-1,  points[:,1].max()+1, numrows)
        grid_x, grid_y  = np.meshgrid(xi, yi)
        #grid_x, grid_y = np.mgrid[xi,yi]

        if method=='gp':
            nclust = len(np.unique(z))
            kernel = 1.0 * RBF([.1, .1])
            gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(points, z)
            # Plot the predicted probabilities. For that, we will assign a color to
            # each point in the mesh [x_min, m_max]x[y_min, y_max].
            Z = gpc_rbf_anisotropic.predict_proba(np.c_[xi.ravel(), yi.ravel()])
            # Put the result into a color plot
            #Z = Z.reshape((xx.shape[0], xx.shape[1], nclust))
            if nclust==2:
                Z = Z[:,0].reshape((grid_x.shape[0], grid_x.shape[1]))
            else:
                Z = Z.reshape((grid_x.shape[0], grid_x.shape[1], nclust))

            plt.imshow(Z,  extent=(xi[0],xi[-1],yi[0],yi[-1]), origin="lower", vmin=0, vmax=1)
            plt.colorbar()
            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=np.array(["b", "y", "r"])[y],
                        edgecolors=(0, 0, 0), alpha=.5)
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.xlim(xi[0], xi[-1])
            plt.ylim(yi[0], yi[-1])
            #plt.xticks(())
            #plt.yticks(())
            plt.title("%s, LML: %.3f" %
                      ("GP", clf.log_marginal_likelihood(clf.kernel_.theta)))
            plt.tight_layout()
            
        else:
            grid_z0 = griddata(points, z, (grid_x, grid_y), method=method)
            plt.imshow(grid_z0.T, origin='lower', extent=(xi[0],xi[-1],yi[0],yi[-1]), interpolation=None, 
                       alpha=.7, cmap='jet', vmin=vmin, vmax=vmax)#
            #plt.scatter(points[:,0], points[:,1],c=z,cmap='jet', vmin=vmin, vmax=vmax, s=10, linewidths=1, alpha=.7) # edgecolor='gray',
            plt.title("Interpolation - {} individuals correctly classified at rate {} ".format(cat, rate))
            plt.colorbar()

    return None

    #m = plt.contourf(grid_x, grid_y, (grid_z0.T).reshape((1,grid_z0.shape[0]*grid_z0.shape[1])))
    #plt.scatter(points[:,0], points[:,1], c=z, s=100,
    #       vmin=zi.min(), vmax=zi.max())
    #fig.colorbar(m)


