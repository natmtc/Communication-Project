
import numpy as np
import warnings
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import RidgeCV

def prepro_decoder_pcs_reaching(dataframe, pc1, pc2, var_to_decode, number_pcs):
    #This functions uses the original dataframe in the pyaldata format
    #pc1 and pc2 are the pcs of the 2 regions (i.e.: M1 and STR) that we want to cut so we have only the reaching interval
    # var_to_decode can be either velocity or position
    #number of pcs for the two different variables, since they need to have the same number
   #############################

    mov_onset=dataframe['idx_movement_on'][0] #same for all the datasets

    val_to_rem_reach=[]
    for i in range(dataframe.shape[0]):
        if dataframe['idx_movement_on'][i]==dataframe['idx_pull_on'][i]: #sometimes there are errors and mov onset is equals to the pull on
            val_to_rem_reach.append(i)
    # Find the values that we want to remove 
    val_to_rem_reach= np.array(val_to_rem_reach)
    # if there are, remove them
    if val_to_rem_reach==[]: 
        df2_reach= dataframe
        pc_reach_1= pc1
        pc_reach_2= pc2

    else: 
        df2_reach= dataframe.drop(val_to_rem_reach).reset_index(drop=True)
        pc_reach_1=pc1.drop(val_to_rem_reach).reset_index(drop=True)
        pc_reach_2=pc2.drop(val_to_rem_reach).reset_index(drop=True)

    reach_values= df2_reach['Reaching'].tolist()
    bins_total_reach= np.sum(reach_values)

    pcs_reach_1=np.zeros((bins_total_reach,number_pcs))
    pcs_reach_2=np.zeros((bins_total_reach,number_pcs))
    
    for i in range(df2_reach.shape[0]):  
        I1_interval= pc_reach_1[i]
        I2_interval= pc_reach_2[i]

        input_1= I1_interval[mov_onset:df2_reach['idx_pull_on'][i],:]
        input_2= I2_interval[mov_onset:df2_reach['idx_pull_on'][i],:]

        if i==0:
            pcs_reach_1[:reach_values[i],:]=input_1
            pcs_reach_2[:reach_values[i],:]=input_2

        else:       
            pcs_reach_1[np.sum(reach_values[:i]):np.sum(reach_values[:i+1]),:]=input_1
            pcs_reach_2[np.sum(reach_values[:i]):np.sum(reach_values[:i+1]),:]=input_2


    if var_to_decode=='position':
        pos_reach=np.zeros((bins_total_reach,3))
        for i in range(df2_reach.shape[0]):    
        #     if i!=21: #only for 38_052319
                pos_cut= df2_reach['jTrjB'][i][mov_onset:df2_reach['idx_pull_on'][i],:]
                if i==0:
                    pos_reach[:reach_values[i],:]=pos_cut
                else:
                    pos_reach[np.sum(reach_values[:i]):np.sum(reach_values[:i+1]),:]=pos_cut

        nanval=np.argwhere(np.isnan(pos_reach)) #some vectors have NaN values from when they stopped recording
        val=[]
        for i in range(nanval.shape[0]):
            val.append(nanval[i,0]) #to only take once the the time point (because it appears for every coordinate: x,y,z) 
        finalnanval = []
        [finalnanval.append(x) for x in val if x not in finalnanval];
    
        pos_reach = np.delete(pos_reach, finalnanval, 0)
        pcs_reach_1 = np.delete(pcs_reach_1, finalnanval, 0)
        pcs_reach_2 = np.delete(pcs_reach_2, finalnanval, 0)

        return pcs_reach_1, pcs_reach_2, pos_reach


    elif var_to_decode=='velocity':
        vel_reach=np.zeros((bins_total_reach,3))
        for i in range(df2_reach.shape[0]):    
        #     if i!=21: #only for 38_052319
                vel_cut= df2_reach['hVelB'][i][mov_onset:df2_reach['idx_pull_on'][i],:]
                if i==0:
                    vel_reach[:reach_values[i],:]=vel_cut
                else:
                    vel_reach[np.sum(reach_values[:i]):np.sum(reach_values[:i+1]),:]=vel_cut

        nanval=np.argwhere(np.isnan(vel_reach)) #some vectors have NaN values from when they stopped recording
        val=[]
        for i in range(nanval.shape[0]):
            val.append(nanval[i,0]) #to only take once the the time point (because it appears for every coordinate: x,y,z) 
        finalnanval = []
        [finalnanval.append(x) for x in val if x not in finalnanval];

        vel_reach = np.delete(vel_reach, finalnanval, 0)
        pcs_reach_1 = np.delete(pcs_reach_1, finalnanval, 0)
        pcs_reach_2 = np.delete(pcs_reach_2, finalnanval, 0)

        return pcs_reach_1, pcs_reach_2, vel_reach




def prepro_decoder_communication_reaching(dataframe, pc1, pc2, var_to_decode, number_pcs):
    #This functions uses the original dataframe in the pyaldata format
    #pc1 and pc2 are the pcs of the communicating regions (i.e.: shared and unshared dimensions) that we want to cut so we have only the reaching interval
    # var_to_decode can be either velocity or position
    #number of pcs for the two different variables, since they need to have the same number
   #############################
       
    mov_onset=dataframe['idx_movement_on'][0] #same for all the datasets
    reach_values= dataframe['Reaching'].tolist()
    bins_total_reach= np.sum(reach_values)
    steps= np.arange(0, pc1.shape[0],75) #must be the same for pc1 and pc2

    pcs_reach_1=np.zeros((bins_total_reach,number_pcs))
    pcs_reach_2=np.zeros((bins_total_reach,number_pcs))
    
    for i in range(dataframe.shape[0]):  
   
        input_1= pc1[mov_onset+steps[i]:mov_onset+steps[i]+reach_values[i],:]
        input_2= pc2[mov_onset+steps[i]:mov_onset+steps[i]+reach_values[i],:]

        if i==0:
            pcs_reach_1[:reach_values[i],:]=input_1
            pcs_reach_2[:reach_values[i],:]=input_2

        else:       
            pcs_reach_1[np.sum(reach_values[:i]):np.sum(reach_values[:i+1]),:]=input_1
            pcs_reach_2[np.sum(reach_values[:i]):np.sum(reach_values[:i+1]),:]=input_2


    if var_to_decode=='position':
        pos_reach=np.zeros((bins_total_reach,3))
        for i in range(dataframe.shape[0]):    
        #     if i!=21: #only for 38_052319
                pos_cut= dataframe['jTrjB'][i][mov_onset:mov_onset+reach_values[i],:]
                if i==0:
                    pos_reach[:reach_values[i],:]=pos_cut
                else:
                    pos_reach[np.sum(reach_values[:i]):np.sum(reach_values[:i+1]),:]=pos_cut

        nanval=np.argwhere(np.isnan(pos_reach)) #some vectors have NaN values from when they stopped recording
        val=[]
        for i in range(nanval.shape[0]):
            val.append(nanval[i,0]) #to only take once the the time point (because it appears for every coordinate: x,y,z) 
        finalnanval = []
        [finalnanval.append(x) for x in val if x not in finalnanval];
    
        pos_reach = np.delete(pos_reach, finalnanval, 0)
        pcs_reach_1 = np.delete(pcs_reach_1, finalnanval, 0)
        pcs_reach_2 = np.delete(pcs_reach_2, finalnanval, 0)

        return pcs_reach_1, pcs_reach_2, pos_reach


    elif var_to_decode=='velocity':
        vel_reach=np.zeros((bins_total_reach,3))
        for i in range(dataframe.shape[0]):    
        #     if i!=21: #only for 38_052319
                vel_cut= dataframe['hVelB'][i][mov_onset:mov_onset+reach_values[i],:]
                if i==0:
                    vel_reach[:reach_values[i],:]=vel_cut
                else:
                    vel_reach[np.sum(reach_values[:i]):np.sum(reach_values[:i+1]),:]=vel_cut

        nanval=np.argwhere(np.isnan(vel_reach)) #some vectors have NaN values from when they stopped recording
        val=[]
        for i in range(nanval.shape[0]):
            val.append(nanval[i,0]) #to only take once the the time point (because it appears for every coordinate: x,y,z) 
        finalnanval = []
        [finalnanval.append(x) for x in val if x not in finalnanval];

        vel_reach = np.delete(vel_reach, finalnanval, 0)
        pcs_reach_1 = np.delete(pcs_reach_1, finalnanval, 0)
        pcs_reach_2 = np.delete(pcs_reach_2, finalnanval, 0)

        return pcs_reach_1, pcs_reach_2, vel_reach



def prepro_decoder_pcs_grasping(dataframe, pc1, pc2, var_to_decode, number_pcs):
    #This functions uses the original dataframe in the pyaldata format
    #pc1 and pc2 are the pcs of the 2 regions (i.e.: M1 and STR) that we want to cut so we have only the grasping interval
    # var_to_decode can be either velocity or position
    #number of pcs for the two different variables, since they need to have the same number
   #############################
   
    val_to_rem_grasp=[]
    for i in range(dataframe.shape[0]):
        if dataframe['idx_pull_on'][i]==dataframe['idx_pull_off'][i]: #sometimes there are errors and mov onset is equals to the pull on
            val_to_rem_grasp.append(i)
    # Find the values that we want to remove 
    val_to_rem_grasp= np.array(val_to_rem_grasp)
    # if there are, remove them
    if val_to_rem_grasp==[]: 
        df2_grasp= dataframe
        pc_grasp_1= pc1
        pc_grasp_2= pc2

    else: 
        df2_grasp= dataframe.drop(val_to_rem_grasp).reset_index(drop=True)
        pc_grasp_1=pc1.drop(val_to_rem_grasp).reset_index(drop=True)
        pc_grasp_2=pc2.drop(val_to_rem_grasp).reset_index(drop=True)

    grasp_values= df2_grasp['Grasping'].tolist()
    bins_total_grasp= np.sum(grasp_values)

    pcs_grasp_1=np.zeros((bins_total_grasp,number_pcs))
    pcs_grasp_2=np.zeros((bins_total_grasp,number_pcs))
    
    for i in range(df2_grasp.shape[0]):  
        I1_interval= pc_grasp_1[i]
        I2_interval= pc_grasp_2[i]

        input_1= I1_interval[df2_grasp['idx_pull_on'][i]:df2_grasp['idx_pull_off'][i],:]
        input_2= I2_interval[df2_grasp['idx_pull_on'][i]:df2_grasp['idx_pull_off'][i],:]

        if i==0:
            pcs_grasp_1[:grasp_values[i],:]=input_1
            pcs_grasp_2[:grasp_values[i],:]=input_2

        else:       
            pcs_grasp_1[np.sum(grasp_values[:i]):np.sum(grasp_values[:i+1]),:]=input_1
            pcs_grasp_2[np.sum(grasp_values[:i]):np.sum(grasp_values[:i+1]),:]=input_2


    if var_to_decode=='position':
        pos_grasp=np.zeros((bins_total_grasp,3))
        for i in range(df2_grasp.shape[0]):    
        #     if i!=21: #only for 38_052319
                pos_cut= df2_grasp['jTrjB'][i][df2_grasp['idx_pull_on'][i]:df2_grasp['idx_pull_off'][i],:]
                if i==0:
                    pos_grasp[:grasp_values[i],:]=pos_cut
                else:
                    pos_grasp[np.sum(grasp_values[:i]):np.sum(grasp_values[:i+1]),:]=pos_cut

        nanval=np.argwhere(np.isnan(pos_grasp)) #some vectors have NaN values from when they stopped recording
        val=[]
        for i in range(nanval.shape[0]):
            val.append(nanval[i,0]) #to only take once the the time point (because it appears for every coordinate: x,y,z) 
        finalnanval = []
        [finalnanval.append(x) for x in val if x not in finalnanval];
    
        pos_grasp = np.delete(pos_grasp, finalnanval, 0)
        pcs_grasp_1 = np.delete(pcs_grasp_1, finalnanval, 0)
        pcs_grasp_2 = np.delete(pcs_grasp_2, finalnanval, 0)

        return pcs_grasp_1, pcs_grasp_2, pos_grasp


    elif var_to_decode=='velocity':
        vel_grasp=np.zeros((bins_total_grasp,3))
        for i in range(df2_grasp.shape[0]):    
        #     if i!=21: #only for 38_052319
                vel_cut= df2_grasp['hVelB'][i][df2_grasp['idx_pull_on'][i]:df2_grasp['idx_pull_off'][i],:]
                if i==0:
                    vel_grasp[:grasp_values[i],:]=vel_cut
                else:
                    vel_grasp[np.sum(grasp_values[:i]):np.sum(grasp_values[:i+1]),:]=vel_cut

        nanval=np.argwhere(np.isnan(vel_grasp)) #some vectors have NaN values from when they stopped recording
        val=[]
        for i in range(nanval.shape[0]):
            val.append(nanval[i,0]) #to only take once the the time point (because it appears for every coordinate: x,y,z) 
        finalnanval = []
        [finalnanval.append(x) for x in val if x not in finalnanval];

        vel_grasp = np.delete(vel_grasp, finalnanval, 0)
        pcs_grasp_1 = np.delete(pcs_grasp_1, finalnanval, 0)
        pcs_grasp_2 = np.delete(pcs_grasp_2, finalnanval, 0)

        return pcs_grasp_1, pcs_grasp_2, vel_grasp



def prepro_decoder_communication_grasping(dataframe, pc1, pc2, var_to_decode, number_pcs):
    #This functions uses the original dataframe in the pyaldata format
    #pc1 and pc2 are the pcs of the communicating regions (i.e.: shared and unshared dimensions) that we want to cut so we have only the grasping interval
    # var_to_decode can be either velocity or position
    #number of pcs for the two different variables, since they need to have the same number
   #############################
           
    mov_onset=dataframe['idx_movement_on'][0] #same for all the datasets
    grasp_values= dataframe['Grasping'].tolist()
    reach_values= dataframe['Reaching'].tolist()
    bins_total_grasp= np.sum(grasp_values)
    steps= np.arange(0, pc1.shape[0],75) #must be the same for pc1 and pc2

    pcs_grasp_1=np.zeros((bins_total_grasp,number_pcs))
    pcs_grasp_2=np.zeros((bins_total_grasp,number_pcs))
    
    for i in range(dataframe.shape[0]):  
   
        input_1= pc1[mov_onset+steps[i]+reach_values[i]:mov_onset+steps[i]+reach_values[i]+grasp_values[i],:]
        input_2= pc2[mov_onset+steps[i]+reach_values[i]:mov_onset+steps[i]+reach_values[i]+grasp_values[i],:]

        if i==0:
            pcs_grasp_1[:grasp_values[i],:]=input_1
            pcs_grasp_2[:grasp_values[i],:]=input_2

        else:       
            pcs_grasp_1[np.sum(grasp_values[:i]):np.sum(grasp_values[:i+1]),:]=input_1
            pcs_grasp_2[np.sum(grasp_values[:i]):np.sum(grasp_values[:i+1]),:]=input_2


    if var_to_decode=='position':
        pos_grasp=np.zeros((bins_total_grasp,3))
        for i in range(dataframe.shape[0]):    
        #     if i!=21: #only for 38_052319
                pos_cut= dataframe['jTrjB'][i][mov_onset+steps[i]+reach_values[i]:mov_onset+steps[i]+reach_values[i]+grasp_values[i],:]
                if i==0:
                    pos_grasp[:grasp_values[i],:]=pos_cut
                else:
                    pos_grasp[np.sum(grasp_values[:i]):np.sum(grasp_values[:i+1]),:]=pos_cut

        nanval=np.argwhere(np.isnan(pos_grasp)) #some vectors have NaN values from when they stopped recording
        val=[]
        for i in range(nanval.shape[0]):
            val.append(nanval[i,0]) #to only take once the the time point (because it appears for every coordinate: x,y,z) 
        finalnanval = []
        [finalnanval.append(x) for x in val if x not in finalnanval];
    
        pos_grasp = np.delete(pos_grasp, finalnanval, 0)
        pcs_grasp_1 = np.delete(pcs_grasp_1, finalnanval, 0)
        pcs_grasp_2 = np.delete(pcs_grasp_2, finalnanval, 0)

        return pcs_grasp_1, pcs_grasp_2, pos_grasp


    elif var_to_decode=='velocity':
        vel_grasp=np.zeros((bins_total_grasp,3))
        for i in range(dataframe.shape[0]):    
        #     if i!=21: #only for 38_052319
                vel_cut= dataframe['hVelB'][i][mov_onset+steps[i]+reach_values[i]:mov_onset+steps[i]+reach_values[i]+grasp_values[i],:]
                if i==0:
                    vel_grasp[:grasp_values[i],:]=vel_cut
                else:
                    vel_grasp[np.sum(grasp_values[:i]):np.sum(grasp_values[:i+1]),:]=vel_cut

        nanval=np.argwhere(np.isnan(vel_grasp)) #some vectors have NaN values from when they stopped recording
        val=[]
        for i in range(nanval.shape[0]):
            val.append(nanval[i,0]) #to only take once the the time point (because it appears for every coordinate: x,y,z) 
        finalnanval = []
        [finalnanval.append(x) for x in val if x not in finalnanval];

        vel_grasp = np.delete(vel_grasp, finalnanval, 0)
        pcs_grasp_1 = np.delete(pcs_grasp_1, finalnanval, 0)
        pcs_grasp_2 = np.delete(pcs_grasp_2, finalnanval, 0)

        return pcs_grasp_1, pcs_grasp_2, vel_grasp




def add_history(bins_before,pc1,pc2,var_to_decode, coordinate):
    #Adds the bins of history (bins_before) that we want to add to our different variables
    X=pc1
    neural_pca_ampl_1= np.zeros((pc1.shape[0],pc1.shape[1]*(bins_before +1)))

    for t in range(bins_before+1, len(pc1)):  
        X_hist=[]
        X_hist.append(X[t-1, :])
        X_hist.append(X[t-2, :])
        X_hist.append(X[t-3, :])
        X_hist.append(X[t-4, :])
        X_hist.append(X[t-5, :])    
        X_hist.append(X[t-6, :])    
        X_hist.append(X[t-7, :])    
        X_hist.append(X[t-8, :])    
        X_hist.append(X[t-9, :])    
        X_hist.append(X[t-10, :])   
        X_hist=np.array(X_hist)
        neural_pca_ampl_1[t,:]=np.concatenate((pc1[t,:],np.concatenate(X_hist)),axis=0)
        
    neural_pca_ampl_1=np.delete(neural_pca_ampl_1, np.array(list(range(0,bins_before))),0)   

    #########################################
    X=pc2
    neural_pca_ampl_2= np.zeros((pc2.shape[0],pc2.shape[1]*(bins_before +1)))

    for t in range(bins_before+1, len(pc2)):  
        X_hist=[]
        X_hist.append(X[t-1, :])
        X_hist.append(X[t-2, :])
        X_hist.append(X[t-3, :])
        X_hist.append(X[t-4, :])
        X_hist.append(X[t-5, :])    
        X_hist.append(X[t-6, :])    
        X_hist.append(X[t-7, :])    
        X_hist.append(X[t-8, :])    
        X_hist.append(X[t-9, :])    
        X_hist.append(X[t-10, :]) 
        X_hist=np.array(X_hist)
        neural_pca_ampl_2[t,:]=np.concatenate((pc2[t,:],np.concatenate(X_hist)),axis=0)
        
    neural_pca_ampl_2=np.delete(neural_pca_ampl_2, np.array(list(range(0,bins_before))),0)   

    #########################################
    X=var_to_decode[:,coordinate].reshape(var_to_decode.shape[0],1)
    var_ampl= np.zeros((var_to_decode.shape[0],X.shape[1]*(bins_before +1)))

    for t in range(bins_before+1, len(var_to_decode)):  
        X_hist=[]
        X_hist.append(X[t-1, :])
        X_hist.append(X[t-2, :])
        X_hist.append(X[t-3, :])
        X_hist.append(X[t-4, :])
        X_hist.append(X[t-5, :])    
        X_hist.append(X[t-6, :])    
        X_hist.append(X[t-7, :])    
        X_hist.append(X[t-8, :])    
        X_hist.append(X[t-9, :])    
        X_hist.append(X[t-10, :]) 
        X_hist=np.array(X_hist)
        var_ampl[t,:]=np.hstack((X[t,:],np.hstack(X_hist)))
        
    var_ampl=np.delete(var_ampl,np.array(list(range(0,bins_before))),0)  

    return neural_pca_ampl_1, neural_pca_ampl_2, var_ampl



def RidgeCV_decoder(X,y):
    
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    model = RidgeCV(alphas=np.arange(0, 1, 0.99), cv=cv,normalize=True)
    model.fit(X, y)
    r2_value=model.score(X,y)
    #print(f'score: {r2_value}')
    #print(model.alpha_)

    pr=model.predict(X)
    warnings.filterwarnings("ignore")

    return r2_value,pr 