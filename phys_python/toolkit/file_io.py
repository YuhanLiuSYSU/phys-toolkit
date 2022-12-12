# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:55:03 2021

@author: Yuhan Liu (yuhanliu@uchicago.edu)
"""
import os
import pickle
from sys import platform


class File_access:
    """
    Example:
        my_list, my_int = [1, 2], 3
        Dir = File_access()
        Dir.save_data([my_list, my_int]) # input save_test
        
        [your_list, your_int] = Dir.get_back("save_test")
    
    """
    
    # path to save the data
    def __init__(self, use_default = 1):
        
        if use_default == 1:
            my_dir = os.getcwd()
        else:
            # this is the old old path
            my_dir = ("C:\\Users\\sglil\\OneDrive\\Desktop" + 
                        "\\CS\\python\\spin-chain\\spinIsing\\")
            
        if platform == "win32":
            self.current_dir = os.path.join(my_dir,'save_results\\')
        elif platform == "linux":
            self.current_dir = os.path.join(my_dir,'save_results/')
            
        if os.path.exists(self.current_dir)==False: os.makedirs(self.current_dir)
         
        
    def save_data(self, result_data, file_name = ""): 
        # result_data can be a tuple, like [result, N]
        
        if file_name == "":
            file_name = input('--> Input the save data name: '+
                                   '(press <ENTER> for not to save)')
                       
        if file_name!="":
            with open(self.current_dir+file_name+'.txt', 'wb') as f: 
                pickle.dump(result_data, f)
            
        self.save_dir = self.current_dir+file_name+'.txt'

                
    def append(self, append_data, file_name = None):
        # TODO: append does not work at this point!!!
        
        if file_name == None:
            open_dir = self.save_dir
        else:
            open_dir = self.current_dir+file_name+'.txt'
        
        # with open(open_dir, 'rb+') as f:  
        f = open(open_dir, 'rb+')
        f.seek(0)
        f.truncate()
        pickle.dump(append_data, f)

                
    def save_fig(self,fig, save_name = ""):
        
        
        if save_name == "":
            save_name = input(
                '--- Input the save fig name (press <ENTER> for not to save): ')
            
        if save_name!="":
            fig.savefig(self.current_dir+save_name+'.png', bbox_inches='tight', dpi=300)
            # fig.savefig(self.current_dir+save_name+'.pdf', bbox_inches='tight')
        
        
    def get_back_ext(self,is_from_new):
        if is_from_new == 1:
            open_file_name = input(
                '--- Input the open data name: (press <ENTER> for not to open)')
            with open(self.current_dir+'last_open'+'.txt', 'wb') as f: 
                pickle.dump(open_file_name,f)
        else:
            with open(self.current_dir+'last_open'+'.txt','rb') as f:         
                open_file_name = pickle.load(f)
                
        return self.get_back(open_file_name)
        
    
    def get_back(self, file_name):  
        with open(self.current_dir+file_name+'.txt','rb') as f:         
            return pickle.load(f)
        
        
    def re_save(self,Model):
        with open(self.current_dir+'last_open'+'.txt', 'rb') as f: 
            open_file_name = pickle.load(f)
            with open(self.current_dir+open_file_name+'.txt','wb') as f_re:      
                pickle.dump(Model,f_re)
                

if __name__ == "__main__":
    
    my_int = 1
    my_list = [2,3]
    my_list_2 = [4,5]
    Dir = File_access()
    Dir.save_data([my_int, my_list],"save_test")
    Dir.append(my_list_2)
    
    test = Dir.get_back("save_test")
    print(test)
    