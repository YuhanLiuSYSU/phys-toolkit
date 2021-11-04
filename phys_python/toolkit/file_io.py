# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:55:03 2021

@author: sglil
"""
import os
import pickle



class File_access:
    # path to save the data
    def __init__(self, use_default = 1):
        
        if use_default == 1:
            my_dir = os.getcwd()
        else:
            # this is the old old path
            my_dir = ("C:\\Users\\sglil\\OneDrive\\Desktop" + 
                        "\\CS\\python\\spin-chain\\spinIsing\\")
            
        self.save_dir = os.path.join(my_dir,'save_results\\')
        if os.path.exists(self.save_dir)==False: os.makedirs(self.save_dir)
         
        
    def save_data(self, result_data): 
        # result_data can be a tuple, like [result, N]
        
        save_file_name = input('--> Input the save data name: '+
                               '(press <ENTER> for not to save)')
        if save_file_name!="":
            with open(self.save_dir+save_file_name+'.pkl', 'wb') as f: 
                pickle.dump(result_data, f)
                
    def save_fig(self,fig):
        save_name = input('--- Input the save fig name (press <ENTER> for not to save): ')
        if save_name!="":
            fig.savefig(self.save_dir+save_name+'.pdf', bbox_inches='tight')
        
        
    def get_back_ext(self,is_from_new):
        if is_from_new == 1:
            open_file_name = input('--- Input the open data name: (press <ENTER> for not to open)')
            with open(self.save_dir+'last_open'+'.pkl', 'wb') as f: 
                pickle.dump(open_file_name,f)
        else:
            with open(self.save_dir+'last_open'+'.pkl','rb') as f:         
                open_file_name = pickle.load(f)
                
        
        return self.get_back(open_file_name)
        
    
    def get_back(self,file_name):  
        with open(self.save_dir+file_name+'.pkl','rb') as f:         
            return pickle.load(f)
        
    def re_save(self,Model):
        with open(self.save_dir+'last_open'+'.pkl', 'rb') as f: 
            open_file_name = pickle.load(f)
            with open(self.save_dir+open_file_name+'.pkl','wb') as f_re:      
                pickle.dump(Model,f_re)