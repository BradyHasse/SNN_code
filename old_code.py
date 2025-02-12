# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:46:00 2024

@author: BAH150
"""








        # #Save input spike array and input indices array
        # with open(spkFile_s, 'wb') as f:    
        #     num_sources = len(inp_spikes)
        #     num_targets = len(inp_spikes[0])
        #     num_reps = rep_cnt
        #     np.save(f,num_sources)
        #     np.save(f,num_targets)
        #     np.save(f,num_reps)
        #     for i in range(num_sources):
        #         for j in range(num_targets):
        #             #np.save(f,len(inp_spikes[i][j]))
        #             for k in range(num_reps[j]):                
        #                 np.save(f,np.array(inp_spikes[i][j][k]))
        
        # num_sources = len(inp_spikes)
        # num_targets = len(inp_spikes[0])
        # num_reps = rep_cnt
        
        # with open(indFile_s, 'wb') as f:
        #     np.save(f,num_sources)
        #     np.save(f,num_targets) 
        #     np.save(f,num_reps)
        #     for i in range(num_sources):
        #         for j in range(num_targets):
        #             for k in range(num_reps[j]):                
        #                 np.save(f,np.array(inp_indices[i][j][k]))   
        
        # with open(indFile, 'rb') as f:
        #     num_sources= int(np.load(f))
        #     num_targets= int(np.load(f))
        #     rep_num= np.load(f)
        #     inp_indices=[[[[] for k in range(rep_num[j])] for j in range(num_targets)] for i in range(num_sources)]
        
        
        #     #input_spikes=[[[[] for k in range(rep_num[j])] for j in range(num_targets)] for i in range(num_sources)]
        
        #     for i in range(num_sources):
        #         for j in range(num_targets):
        #             nrep = rep_num[j]
        #             for k in range(nrep):
        #                 g_tmp=np.load(f)
        #                 #print('Rep',k)    
        #                 inp_indices[i][j][k]=g_tmp
                    
        # with open(spkFile, 'rb') as f:
        
        #     num_sources= int(np.load(f))
        #     num_targets= int(np.load(f))
        #     rep_num= np.load(f)
        
        #     inp_spikes=[[[[] for k in range(rep_num[j])] for j in range(num_targets)] for i in range(num_sources)]
        
        #     for i in range(num_sources):
        #         for j in range(num_targets):
        #             nrep = rep_num[j]
        #             for k in range(nrep):
        #                 g_tmp=np.load(f)
        #                 #print('Rep',k)    
        #                 inp_spikes[i][j][k]=g_tmp * second