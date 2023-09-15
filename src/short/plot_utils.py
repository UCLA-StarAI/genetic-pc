from numpy.lib.type_check import real
from numpy.lib.utils import info
from scipy.stats import pearsonr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import pandas as pd
import copy
import matplotlib.backends.backend_pdf
import seaborn as sns
from scipy import stats as scs
import os

try:
    import ot
    ot_loaded = True
except ModuleNotFoundError:
    ot_loaded = False
try:
    import statsmodels.api as sm
    sm_loaded = True
except ModuleNotFoundError:
    sm_loaded = False

    
def all_same( items ):
    return len( set( items ) ) == 1 

def plotreg(x, y, keys, statname, col, ax=None, ticks=None): 
    """
    Plot for x versus y with regression scores and returns correlation coefficient
    
    Parameters
    ----------
    x : array, scalar
    y : array, scalar
    statname : str
        'Allele frequency' LD' or '3 point correlation' etc.
    col : str, color code
        color

    """
        
    lims = [np.min(x), np.max(x)]
    r,_ = pearsonr(x,y)
    if sm_loaded:
        reg = sm.OLS(x, y).fit()      
    if ax is None:
        ax = plt.subplot(1,1,1)
    if len(x)<100:
        alpha=1
    else:
        alpha=.6
    # if sm_loaded:
    # ax.plot(x, y , label=f"{keys[1]}: cor={round(r,2)} slope={round(reg.params[0],2)}", c=col, marker='o', lw=0, alpha=alpha)
    ax.plot(x, y , label=f"$\sigma^2$={round(r,2)}", c=col, marker='o', lw=0, alpha=alpha, rasterized=True)
    # ax.scatter(x, y, label=f"{keys[1]}: cor={round(r,2)}", c=col, rasterized=True)
    #ax.axis('equal') 
    #ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', alpha=.5)
    ax.plot(lims, lims, ls='--', alpha=1, c='black', rasterized=True)
    # ax.plot(lims, lims, ls='--', c='black', rasterized=True)
    ax.set_xlabel(f'AF in {keys[0]}')
    ax.set_ylabel(f'AF in {keys[1]}')
    if ticks is not None:
        print(ticks)
        ax.set_xticks(ticks, ticks)
        ax.set_yticks(ticks, ticks)
    ax.legend(loc='upper left')
    return r


def plotregquant(x, y, keys, statname, col, step=0.00005, cumsum=False, ax=None):
    """
    Plot quantiles for x versus y (every step) with regression scores and returns correlation coefficient
    
    Parameters
    ----------
    x : array, scalar
    y : array, scalar
    statname : str
        'Allele frequency' LD' or '3 point correlation' etc.
    col : str, color code
        color
    step : float
        step between quantiles
    cumsum : boolean
        plot cumulative sum of quantiles instead
        
    Return
    ------
    r: float
        Pearson correlation coefficient
    
    """
    q = np.arange(0,1+step,step=step)
    print("\tQuantile", q)
    x = np.nanquantile(x, q)
    y = np.nanquantile(y, q)
    if cumsum:
        x=np.cumsum(x)
        y=np.cumsum(y)
    r = plotreg(x=x, y=y, keys=keys, statname=f'Quantiles {statname}', col=col, ax=ax)
    return r


def plotquant(x, y, keys, statname, col, step=0.05, cumsum=False, ax=None):
    """
    Plot quantiles for y (every step) with regression scores of x quant vs y quant
    
    Parameters
    ----------
    x : array, scalar
    y : array, scalar
    statname : str
        'Allele frequency' LD' or '3 point correlation' etc.
    col : str, color code
        color
    step : float
        step between quantiles
    cumsum : boolean
        plot cumulative sum of quantiles instead
    """
        
    q = np.arange(0,1,step=step)
    x = np.nanquantile(x, q)
    y = np.nanquantile(y, q)
    if cumsum:
        x=np.cumsum(x)
        y=np.cumsum(y)
        cum=' (cumsum)'
    else:
        cum=''
        
    r,_ = pearsonr(x,y)
    reg = sm.OLS(x, y).fit()      
    linregress(x, y)
    if ax is None:
        ax = plt.subplot(1,1,1)
    #ax.plot(q, y , label=f"{keys[1]}: cor={round(r,2)} slope={round(reg.params[0],2)}", c=col, marker='o', lw=1, alpha=.5)
    ax.plot(q, y , label=f"{keys[1]}: cor={round(r,2)}", c=col, marker='o', lw=1, alpha=.5)
    ax.set_xlabel(f'Quantile')
    ax.set_ylabel(f'{statname}{cum} in {keys[1]}')
    ax.legend()
    return r

    
### LD related ###
def get_dist(posfname, kept_preprocessing='all', kept_snp='all', region_len_only=False):
    legend = pd.read_csv(posfname, sep=' ')
    snps_on_same_chrom = legend.id[0].split(':')[0] == legend.id[legend.shape[0]-1].split(':')[0] # are all SNPs on the same chromosome ?
    ALLPOS = np.array(legend.position)
    region_len = ALLPOS[-1]-ALLPOS[0]
    if region_len_only:
        return None, region_len, snps_on_same_chrom
    
    # Order below is important
    if not kept_preprocessing is 'all': 
        ALLPOS = ALLPOS[kept_preprocessing]
    if not kept_snp is 'all': 
        ALLPOS = ALLPOS[kept_snp]
    # compute matrix of distance between SNPS (nsnp x nsnp)
    dist = np.abs(ALLPOS[:, None] - ALLPOS)

    # flatten and take upper triangular
    flatdist = dist[np.triu_indices(dist.shape[0])]
    return flatdist, region_len, snps_on_same_chrom
    
    
def plotoneblock(A, REF=None, keys=[None,None], statname='LD', ax=None, 
                 matshow_kws={'cmap':cm.get_cmap('viridis_r') ,'vmin':0, 'vmax':1}, suptitle_kws={}):
    if REF is None:
        keys = [keys[0]]*2 # mirrored/symmetrical matrix, identical name for both axes
    else:
        if REF.shape != A.shape:
            print("Warning: plotblock: matrices of different sizes cannot be arranged",
                  " below and above the diagonal of a single plot.",
                  " Set REF to None in plotoneblock, ie mirror to True in plotLDblock")
            return
        #mask =  np.triu_indices(A.shape[0], k=0)
        A[np.triu_indices(A.shape[0], k=0)]=0 
        A = A.T + REF

    if ax is None: plt.subplot(1,1,1)
    imgplot = ax.matshow(A, **matshow_kws)
    plt.colorbar(imgplot, shrink=.65, ax=ax)
    ax.set_xlabel(f'{statname} in {keys[0]} (above diagonal)') # or as title ?
    plt.title(f'{statname} in {keys[0]} vs Truth') # or as title ?
    ax.set_ylabel(f'{statname} in Truth (below diagonal)')
    # if len(suptitle_kws)>0: plt.suptitle(**suptitle_kws)
    plt.tight_layout()


def plotLDblock(hcor_snp, left=None, right=None, ref='Real', mirror=False, diff=False, is_fixed=None, is_fixed_dic=None, suptitle_kws={}):
    """
    Parameters
    ----------
    hcor_snp : list 
        list of matrices containing r**2 pairwise values
    left : int
        starting SNP, if None the region starts at the very first SNP
    right : int
        finishing SNP (excluded), if None the region encompasses the very last SNP 
    mirror : bool
        if True print the symmetrical matrix for each category
        if False, print LD in one dataset versus the reference dataset (only if datasets have the same number of sites)
    diff : bool
        show A-REF
    is_fixed : np.array(bool)
        bool array indicating sites that are fixed in at least one of the dataset  
    is_fixed_dic : dict(np.array(bool))
        dict containing for each dataset (Real, GAN, ...) a bool array indicating fixed sites 
    """
    print(hcor_snp.keys())
    print('start')
    if diff:
        cmap = cm.get_cmap('RdBu_r') 
        cmap.set_bad('gray')
        vmin,vmax = -1,1
        mirror = True
    else:
        cmap = cm.get_cmap('viridis_r')  #cool' 
        cmap.set_bad('white')
        ## cmap = cm.get_cmap('jet', 10) # jet doesn't have white color
        ## cmap.set_bad('gray') # default value is 'k' 
        vmin,vmax = 0,1

    if (not mirror) or diff:
        if not is_fixed is None:
            REF = hcor_snp[ref][np.ix_(~is_fixed,~is_fixed)]
        elif not is_fixed_dic is None:
            REF = hcor_snp[ref][np.ix_(~is_fixed_dic[ref],~is_fixed_dic[ref])]
        else:
            REF = hcor_snp[ref]
        REF = REF[left:right,left:right] # full ref
        
    if mirror:
        triREF=None
    else:
        triREF = copy.deepcopy(REF)
        triREF[np.triu_indices(triREF.shape[0], k=0)] = 0 # keep only lower triangle      
    
    count = 0
    for _,(cat,hcor) in enumerate(hcor_snp.items()):

        if cat == 'Real':
            continue

        count+=1
        if not is_fixed is None:
            A = hcor[np.ix_(~is_fixed,~is_fixed)]
        elif not is_fixed_dic is None:
            A = hcor[np.ix_(~is_fixed_dic[cat],~is_fixed_dic[cat])]
        else:
            A = hcor
            
        A = copy.deepcopy(A[left:right,left:right]) # copy
        if diff:
            A = REF - A
        ax = plt.subplot(1,len(hcor_snp)-1,count,rasterized=True)
        plotoneblock(A=A, 
                     REF=triREF, 
                     keys=[cat,ref],
                     statname='LD',
                     matshow_kws={'cmap':cmap, 'vmin':vmin, 'vmax':vmax},
                     ax=ax, suptitle_kws=suptitle_kws)
        #ax.set_title("SNP correlation in {} for SNPs {} to {} ;   RSS: {:.4f}".format(cat,a,b, RSS))
        #plt.tight_layout()

        
def datatransform(dat,to_minor_encoding=False, min_af=0, max_af=1):
    # min_af, max_af, to_minor_encoding
    dat = np.array(dat)
    af = np.mean(dat, axis=0)
    if to_minor_encoding:
        dat = (dat + (af>1-af)) %2
        af = np.mean(dat, axis=0)
        
    if (min_af>0) | (max_af<1):
        kept = (af>=min_af) & (af<=max_af)
        dat = dat[:, kept]
    else:
        kept = 'all' #np.full(len(af), fill_value=True)
    return dat, kept


def plotPCAscatter(pcdf, method, orderedCat, outDir):
    pdf = matplotlib.backends.backend_pdf.PdfPages(outDir+"PCA_scatter_compare_" + method + ".pdf")
    if 'coupled_with' in pcdf.columns:
        pcdf = pcdf.query('label == coupled_with')
    g = sns.FacetGrid(pcdf, col="label", col_order = orderedCat)
    g.map(sns.scatterplot, 'PC1', 'PC2',alpha=.1)
    pdf.savefig()
    g = sns.FacetGrid(pcdf, col="label", col_order = orderedCat)
    g.map(sns.scatterplot, 'PC3', 'PC4',alpha=.1)
    pdf.savefig()
    g = sns.FacetGrid(pcdf, col="label", col_order = orderedCat)
    g.map(sns.scatterplot, 'PC5', 'PC6',alpha=.1)
    #plt.suptitle(method)
    plt.tight_layout()
    pdf.savefig()
    pdf.close()
    
    
def plotPCAsuperpose(pcdf, method, orderedCat, outDir, colpal):
    fig, axs = plt.subplots(nrows=3, ncols=len(orderedCat)-1, 
                            figsize = (len(orderedCat)*3.2, 3*3.2), constrained_layout=True)
    ext=1
    for i,pcx in enumerate([0,2,4]):
        win=0
        # compute x and y ranges to force same dimension for all methods
        pcs = pcdf.drop(columns=['label','coupled_with'], errors='ignore').values
        xlim=(np.min(pcs[:,pcx])-ext, np.max(pcs[:,pcx])+ext)
        ylim=(np.min(pcs[:,pcx+1])-ext, np.max(pcs[:,pcx+1])+ext)

        for cat in orderedCat:
            if cat != 'Real':
                if 'coupled_with' in pcdf.columns:
                    reals = (pcdf.label=='Real') &  (pcdf['coupled_with']==cat)            
                else:
                    reals = (pcdf.label=='Real')
                #if cat=='Real': continue
                ax=axs[i,win]
                
                ax.scatter(pcdf.values[reals,pcx], 
                        pcdf.values[reals,pcx+1],alpha=.4, rasterized=True)
                if cat!= 'Real':
                    keep = (pcdf.label==cat) 
                    ax.scatter(pcdf.values[keep,pcx], 
                            pcdf.values[keep,pcx+1],alpha=.4,color=colpal[cat], rasterized=True)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel("PC{}".format(pcx+1))  # make PC label starts at 1
                ax.set_ylabel("PC{}".format(pcx+2))
                ax.set_title(cat)
                # ax.savefig(outDir+"PCA_allel_compare_models_" + method + '_' + cat+ '_' + ".pdf")
                win+=1
    # fig.suptitle(method)
    plt.savefig(outDir+"PCA_allel_compare_models_" + method + ".pdf", bbox_inches='tight')
    print('    - PCA superpose file saved in', outDir+"PCA_allel_compare_models_" + method + ".pdf")

    
def plotPCAdensity(pcdf, method, orderedCat, outDir):
    orderedCat = [cat for cat in orderedCat if cat!='Real']
    pdf = matplotlib.backends.backend_pdf.PdfPages(outDir+"PCA_densities_compare_" + method + ".pdf")
    g = sns.FacetGrid(pcdf, col="label", col_wrap=len(orderedCat), col_order = orderedCat)
    g.map(sns.kdeplot, 'PC1', 'PC2', cmap="RdBu", cbar=True) #Reds_d")
    pdf.savefig(bbox_inches='tight')
    g = sns.FacetGrid(pcdf, col="label", col_wrap=len(orderedCat), col_order = orderedCat)
    g.map(sns.kdeplot, 'PC3', 'PC4', cmap="Reds_d", cbar=True)
    pdf.savefig(bbox_inches='tight')
    g = sns.FacetGrid(pcdf, col="label", col_wrap=len(orderedCat), col_order = orderedCat)
    g.map(sns.kdeplot, 'PC5', 'PC6', cmap="Reds_d", cbar=True)
    pdf.savefig(bbox_inches='tight')
    print('    - PCA density file saved in', outDir+"PCA_densities_compare_" + method + ".pdf")
    pdf.close()


def plotPCAallfigs(pcdf, method, orderedCat, outDir, colpal):
    # plotPCAscatter(pcdf, method, orderedCat, outDir)
    if method!='independent_PCA':
        plotPCAsuperpose(pcdf, method, orderedCat, outDir, colpal)
    plotPCAdensity(pcdf, method, orderedCat, outDir)

    
def computePCAdist(pcdf, method, outDir, stat='wasserstein',reg=1e-3):
    Scores = pd.DataFrame(columns=['stat', 'statistic', 'pvalue', 'label', 'PC', 'method'])
    print(stat)
    for key in pd.unique(pcdf.label):
        if 'coupled_with' in pcdf.columns:
            reals = (pcdf.label=='Real') & (pcdf['coupled_with']==key)            
            gen = (pcdf.label==key) & (pcdf['coupled_with']==key)            
        else:
            reals = (pcdf.label=='Real')
            gen = (pcdf.label==key)
        
        n = np.sum(gen)  # nb samples
        n_real = np.sum(reals)
        ncomp = pcdf.drop(columns=['label','coupled_with'], errors="ignore").shape[1]
        a, b = np.ones((n_real,)) / n_real, np.ones((n,)) / n  # uniform distribution on samples

        for pc, colname in enumerate(pcdf.drop(columns=['label','coupled_with'] , errors="ignore")):
            # if pc>1: # we just need scores for pc1,pc2
            #     continue
            if stat=='wasserstein':
                sc = scs.wasserstein_distance(pcdf[reals].iloc[:,pc],pcdf[gen].iloc[:,pc])
                Scores = Scores.append({'stat':stat, 'statistic':sc, 'pvalue':None, 'label':key, 'PC':pc+1, 'method':method}, ignore_index=True)
                
            elif stat=='wasserstein2D':
                # if pc>0: # we just need the score for (pc1,pc2)
                #     continue
                if pc in [1,3,5]: # compute the score for (1,2), (3,4), (5,6)
                    continue
                if not ot_loaded:   # in case the library is not available
                    pcdf.to_csv(outDir+f'PCA_{method}.csv')
                    return None 
                if pc < ncomp-1 :
                    xs = pcdf[reals].iloc[:,pc:pc+2].values
                    xt = pcdf[gen].iloc[:,pc:pc+2].values
                    # loss matrix
                    M = ot.dist(xs, xt)
                    M /= M.max()
                    print('>>> computePCAdist begin <<<, \n', 'Size of a b M', a.size, b.size, M.size)
                    sc = ot.sinkhorn2(a, b, M, reg)[0]
                    print('>>> computePCAdist begin <<<, stat=wasserstein2D\n',key,sc,'\n>>> computePCAdist end <<<\n')
                    
                    Scores = Scores.append({'stat':stat, 'statistic':sc, 'reg':reg, 'label':key, 'PC':f'{pc+1}-{pc+2}', 'method':method}, ignore_index=True)

            else:
                print(f'{stat} is not a recognized stat option')
                return None

    Scores.to_csv(outDir+f"{stat}_PCA_" + method + ".csv")
    return Scores


def plotPrivacyLoss(Train, Test, outDir, colpal, allcolpal):
    
    if Train in ['Real', '_Real']:
        Train=''
 
    if not os.path.exists(outDir+f'AA{Train}.csv.bz2') or not os.path.exists(outDir+f'AA{Test}.csv.bz2'): 
        print(f"Warning: at least one of the file {outDir+f'AA{Train}.csv.bz2'} or {outDir+f'AA{Test}.csv.bz2'} does not exist")
        dfPL = None
    else:
        AA_Test = pd.read_csv(outDir+f'AA{Test}.csv.bz2') #AA_Test-Gen
        AA_Train = pd.read_csv(outDir+f'AA{Train}.csv.bz2') #AA_Train-Gen 
        PL=dict()
        for cat in AA_Train.cat:
            if not cat in AA_Test.cat.values: continue 
            # Privacy Loss = Test AA - Train AA
            PL[cat] = np.float(AA_Test[AA_Test.cat==cat].AATS) -   np.float(AA_Train[AA_Train.cat==cat].AATS)
        if len(PL)>0:
            dfPL = pd.DataFrame.from_dict(PL,orient='index', columns=['PrivacyLoss'])
            dfPL['cat'] = dfPL.index
        dfPL.to_csv(outDir+f'PL_Train{Train}_Test{Test}.csv')

        colors = [colpal[key] if key in colpal else allcolpal[key] for key in dfPL.cat.values]
        sns.barplot(x='cat',y='PrivacyLoss',data=dfPL, palette=colors)    
        plt.axhline(0, color='black')
        plt.title("Nearest Neighbor Adversarial Accuracy - Privacy Loss")
        plt.ylim(-.5,.5)
        plt.savefig(outDir + f"PrivacyLoss_Train{Train}_Test{Test}.pdf")

    return dfPL
