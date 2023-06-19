import numpy as np
from sklearn.metrics import make_scorer, r2_score
import pyaldata as pyal
from numpy import linalg as LA


def prepro(df, dataset):
    # Returns preprocessed data taking into account the specific datasets we ahve for this project
    #explanation of the preprocessing is in each of the lines

    old_fields = [col for col in df.columns.values if 'unit' in col]

    new_fields = ['M1_spikes' if 'Ctx' in col else 'Str_spikes' for col in old_fields]
    df_ = df.rename(columns = {old:new for old,new in zip(old_fields,new_fields)})

    # change spikes datatype
    for signal in new_fields:
        df_[signal] = [np.nan_to_num(x=s.toarray().T, nan=0) for s in df_[signal]]
    # add trial_id
    df_['trial_id'] = np.arange(1,df_.shape[0]+1)

    # only keep successful trials
    df_= pyal.select_trials(df_, df_.trialType== 'sp') 

    # fill no-laser trials (and index fields) with zero
    n_bins = df_[new_fields[0]][0].shape[0]
    var_len_fields = [ 'spkPullIdx', 'spkRchIdx','spkTimeBlaserI'] #variables that change length
    fill_zeros = lambda a: a if len(a)>1 else np.zeros((n_bins,))

    for field in var_len_fields:
        if field not in df_.columns:continue #f there is this dataset has not inactivated trials go on
        
        df_[field] = [fill_zeros(s) for s in df_[field]]  
        
        
    # fill fields that are cut with np.nans and remove trials that are too long or don't exist
    cut_fields = ['hTrjB', 'hVelB','hDistFromInitPos','jTrjB','forceB']
    df_['badIndex'] = [max(trialT.shape)>n_bins or
                        max(trialV.shape)>n_bins or
                        max(trialD.shape)>n_bins or
                        max(trialP.shape)>n_bins or #added:Natalia
                        max(trialF.shape)>n_bins or #added:Natalia
                        max(trialT.shape) < 2 or 
                        max(trialV.shape) < 2 or                   
                        max(trialP.shape) < 2 or 
                        max(trialF.shape) < 2 or 
                        np.isnan(trialT).sum() > 5 for trialT,trialV,trialD,trialP,trialF in zip(df_.hTrjB,df_.hVelB,df_.hDistFromInitPos,df_.jTrjB,df_.forceB)]

    df_= pyal.select_trials(df_, df_.badIndex == False)
    df_.drop('badIndex', axis=1, inplace=True)
    fill_nans = lambda a: a if max(a.shape)==n_bins else np.pad(a, (((0,n_bins-max(a.shape)),)+(len(a.shape)-1)*((0,0),)), 'constant', constant_values=(np.nan,))
    for field in cut_fields:
        if field not in df_.columns:continue
        df_[field] = [fill_nans(s.T) for s in df_[field]]   
    # add bin_size
    BIN_SIZE=0.01 #useless
    df_['bin_size']=0.02  # useless
    # add idx_movement_on which is exactly at t=df.timeAlign
    df_['idx_movement_on'] = [np.argmin(np.abs(s-i)) for i,s in zip(df_['timeAlign'],df_['spkTimeBins'])]
    # add pull start idx
    df_['idx_pull_on'] = [pullIdx.nonzero()[0][0] if len(pullIdx.nonzero()[0])>0 else np.nan for pullIdx in df_.spkPullIdx]
    # add pull stop idx
    df_['idx_pull_off'] = [min((pull.nonzero()[0][-1], velNans[0] if len(velNans:=np.isnan(vel).nonzero()[0])>0 else [np.inf])) for pull,vel in zip(df_.spkPullIdx,df_.hVelB)]
    # remove trials with no pull idx
    df_.dropna(subset=['idx_pull_on'], inplace=True)
    df_.idx_pull_on = df_.idx_pull_on.astype(np.int32)
    df_.index = np.arange(df_.shape[0])
    # add target_id
    rem = np.remainder(df_['blNumber'].to_numpy(), 4)
    rem[np.logical_or(rem==3 , rem ==0)] = 0
    rem[np.logical_or(rem==1 , rem==2)] = 1
    df_['target_id'] = rem

    for signal in new_fields:
        df_ = pyal.remove_low_firing_neurons(df_, signal, 1)

        df_ = pyal.select_trials(df_, df_.idx_movement_on < df_.idx_pull_on)
        df_ = pyal.select_trials(df_, df_.idx_pull_on < df_.idx_pull_off)
        
        # !!! discard outlier behaviour---tricky stuff !!!
            # reach duration < 500ms
        df_ = pyal.select_trials(df_, df_.idx_pull_on - df_.idx_movement_on < 50)
                # pull duration < 450ms
        df_ = pyal.select_trials(df_, df_.idx_pull_off - df_.idx_pull_on < 45)

        try:
            noLaserIndex = [i for i,laserData in enumerate(df_.spkTimeBlaserI) if not np.any(laserData)]
            df_= pyal.select_trials(df_, noLaserIndex)
        except AttributeError:
            # due to absence of this field in no-laser sessions
            pass

    if dataset<7:
        df_ = pyal.combine_time_bins(df_, 4) #int(BIN_SIZE/.01))
    else:
        df_ = pyal.combine_time_bins(df_, 2) #int(BIN_SIZE/.01))

    for signal in new_fields:
        df_ = pyal.sqrt_transform_signal(df_, signal)

    df_= pyal.add_firing_rates(df_, 'smooth', std=0.05)


    df_['Reaching'] = df_['idx_pull_on'] - df_['idx_movement_on']
    df_['Grasping'] = df_['idx_pull_off'] - df_['idx_pull_on']
    # df_.head()

    return df_




def mahalanobis_distance(x,y):
    #Returns the distance for each point 
    #Returns how many points are 0.2 up and down from the mean point 
    from numpy.linalg import pinv
    dif =x-y
    stack_traj = np.stack((x, y), axis=0) 
    cov_m = np.cov(stack_traj.T)
    inv_covmat = pinv(cov_m) #sing matrix
    left_term = np.dot(dif, inv_covmat)
    s=dif.T
    arr=[]
    
    
    for i in range(x.shape[0]):
        result=np.round(np.sqrt(np.abs(left_term[i]*s[i])),5)
        arr.append(result)
#         print(arr[i]) #if we want to get the distance for each point
    arr=np.array(arr)
    m= np.mean(arr)

    z=[]
    for i in range(x.shape[0]):
        if arr[i]<(m-0.02) or arr[i]>(m+0.02):
            z1=1
            z.append(z1)

    tot= np.sum(z)
    print(f"{np.round((tot/(x.shape[0]))*100)} % ") #percetage of non-parallelity



def gramschmidt(M):
    #In mathematics, particularly linear algebra and numerical analysis,
    # the Gram–Schmidt process is a method for orthonormalizing a set of vectors in an inner product space,
    #I developed this code based on this video ->
    #  https://www.google.com/search?q=Gram%E2%80%93Schmidt+process+python&rlz=1C1CHBF_esES966ES966&oq=Gram%E2%80%93Schmidt+process+python&aqs=chrome..69i57j0i19i22i30.2690j0j7&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:0b76535b,vid:fdshjUzUWTs
    c=np.zeros((M.shape[0],M.shape[1])) #to storage the new normalized and orth column

    for i in range(M.shape[1]):
        if i==0:
            c1= M[:,i]/LA.norm(M[:,i]) #compute normalized first column
        elif i==1:
            proj1=((M[:,i-1].dot(M[:,i]))/(LA.norm(M[:,i-1])**2)) * (M[:,i-1]) #proj of the 2nd column onto the 1st 
            c2= (M[:,i]-proj1)/LA.norm(M[:,i]) #normalized
        else:  
            proj=np.zeros((M.shape[0],i)) #here we store the n projections needed
            for j in range(i):
                proj[:,j]= (M[:,j].dot(M[:,i]))/(LA.norm(M[:,j])**2) *(M[:,j])
                if j==0:
                    sub= M[:,i]-proj[:,j]
                else:
                    sub= sub-proj[:,j]


            c[:,i]= sub/LA.norm(M[:,i])
            del proj

    union_2first=(np.vstack((c1, c2))).T
    components_orth=(np.hstack((union_2first, c[:,2:])))     
    return components_orth 
