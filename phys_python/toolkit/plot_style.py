"""
Created on Fri Oct  1 15:59:29 2021

@author: Yuhan Liu (yuhanliu@uchicago.edu)
"""

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np


from entangle.ent_fit import fit_ent


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIG_SIZE = 22


def plot_style_single(plt, x_labels = None, y_labels = None, title = None,
                      x_lim = None, y_lim = None):
    
   
    if bool(x_labels): plt.xlabel(x_labels)
    if bool(y_labels): plt.ylabel(y_labels)
    if bool(title): plt.title(title)
    if bool(x_lim): plt.xlim(x_lim)
    if bool(y_lim): plt.ylim(y_lim)
        
    fig = plt.gcf()
    plt.show() 
    
    return fig


def plot_style_s(x_datas, y_datas, 
                 x_labels = None, y_labels = None, title = None, 
                 x_lim = None, y_lim = None,
                 is_scatter = 1, scatter_size = 10,
                 is_line = 1, line_labels = None,
                 N = 0, fit_type = -10, usr_func = 0, 
                 Dir = None, add_text = None, add_label = None,
                 pre_ax = None, is_log = 0, my_color = None,
                 sequence = 1):
    """
    
    Parameters
    ----------
    x_data : TYPE
        DESCRIPTION.
    y_data : TYPE
        DESCRIPTION.
    N : int, optional
        Total length of the chain. The default is 0.
        
    fit_type : int, 0, 1, 2, 3, -10
        The default is -10, which is not to fit
        0: linear function
        1: EE
        2: MI
        3: LN
        else: user defined function "usr_func"
              
    usr_func : lambda function, optional
        User-defined fit function. The default is 0.

    sequence: float
        Takes value from 0 to 1.

    Returns
    -------
    None.
    
    Example input:
        

    """
    
    if not bool(pre_ax): 
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_axes([0,0,1,1])
    else:
        ax = pre_ax
    
    
    plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('figure', titlesize = BIG_SIZE)     # fontsize of the figure title
    plt.rc('legend', fontsize = SMALL_SIZE)    
    
    
    # Accomadate different input type
    if not isinstance(y_datas, list): y_datas = [y_datas]
    if not isinstance(x_datas, list): x_datas = [x_datas]
        
    for i, y_data in enumerate(y_datas):
        
        if len(x_datas) == 1:
            x_data = x_datas[0]
        else:
            x_data = x_datas[i]
        
        if is_scatter: 
            if bool(my_color):
                line = ax.scatter(x_data, y_data, scatter_size, color = my_color)
            else:
                line = ax.scatter(x_data, y_data, scatter_size)
                
        if is_line:
            if bool(my_color):
                line, = ax.plot(x_data, y_data, color = my_color)
            else:
                line, = ax.plot(x_data, y_data)

        if bool(line_labels): line.set_label(line_labels[i])
        
              
    ax.grid(True)
    if bool(line_labels): ax.legend(frameon = True)
    
    if is_log == 1:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    if bool(x_labels): ax.set_xlabel(x_labels)
    if bool(y_labels): ax.set_ylabel(y_labels)
    if bool(title): ax.set_title(title)
    
    if bool(x_lim): ax.set_xlim(x_lim)
    if bool(y_lim): ax.set_ylim(y_lim)
       
    # plt.ylim([0.9*abs(y_data.min())*np.sign(y_data.min()),
    #           1.1*abs(y_data.max())*np.sign(y_data.max())])
    
    ax = plt.gca()
    if bool(add_text):        
        ax.text(0.8, 0.9, add_text,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, fontsize = SMALL_SIZE)
        
    if bool(add_label):        
        ax.text(0.06, 0.94, add_label,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, fontsize = SMALL_SIZE)
    
    if fit_type>-10:
        n = 1
        coeffs, coeffs_cov, fit_func = fit_ent(
            x_data, y_data, N, fit_type = fit_type, renyi = n, 
            usr_func = usr_func)
       
        text_content = "$ a = %1.4f $" % coeffs[0]
        # text_content = str(coeffs)
        # print(text_content)
        
        ax = plt.gca()
        ax.text(0.8 , 0.8, text_content,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, fontsize = SMALL_SIZE)
       
        print(coeffs_cov)
        print(coeffs)
        y_fit = fit_func(x_data, *coeffs)
        ax.plot(x_data, y_fit, 'tab:orange')
    else:
        coeffs = None

    
    if bool(Dir) and sequence == 1: 
        fig = plt.gcf()
        Dir.save_fig(fig)
    
    return ax, coeffs



def plot_style(x_data, y_data, Dir = [], N = 0,x_labels = [], y_labels = [], is_save = 0):
    """
    # y_data is a list, contains four data
    Example of usage:
        
        x_data = sub_N
        y_data = [SA,RE-MI,SA,SA]
        x_labels = ['$l_{AB}$','$h_{A:B}$','$l_{AB}$','$l_{AB}$']
        y_labels = ['$S_A$','$S_A$','$S_A$','$S_A$']
        
        fit_and_plot(x_data,y_data,x_labels = x_labels, y_labels = y_labels)
    
    """
    
    
    plt.figure(1)
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title
      
    
    # plt.subplots() is a function that returns a tuple containing a figure and axes object(s). 
    # fig is useful if you want to change figure-level attributes or save the figure as an image file
    # figsize can tune the aspect ratio
    # n_row = int(np.sqrt(len(y_data)))
    n_row = 2
    fig, axs = plt.subplots(nrows = n_row, ncols = n_row, figsize=(8, 7))
    
    for i in range(0,n_row):
        for j in range(0,n_row):
            
            if (i == 0 and j == 0):
                msg = 'tab:orange'
                fit_type = 1
                n = 1
                
            elif (i==0 and j==1):
                msg = 'tab:blue'
                fit_type = 2
                n = 2 # renyi index
                
            elif (i==1 and j==0):
                msg = 'tab:green'
                fit_type = 2
                n = 0 # nothing
                
            else:
                msg = 'tab:red'
            
            
            axs[i,j].plot(x_data, y_data[2*i+j].real)
            axs[i,j].grid(True)

            if bool(x_labels):
                axs[i,j].set(xlabel = x_labels[2*i+j])
            
            if bool(y_labels):
                axs[i,j].set(ylabel = y_labels[2*i+j])
                
                
            # fit the CFT prediction
            if (i+j<2):
                coeffs, coeffs_cov, fit_func = fit_ent(
                    x_data, y_data[2*i+j], N,fit_type = fit_type, renyi = n)
   
                text_content = "$ c = %1.3f $" % coeffs[0]
                axs[i,j].text(s = text_content,
                              **text_coords(ax=axs[i,j],
                                            scalex=0.3,scaley=0.85),
                              fontsize = SMALL_SIZE)
                print(coeffs_cov)
                y_fit = fit_func(x_data,coeffs[0],coeffs[1])
                axs[i,j].plot(x_data, y_fit, 'tab:orange')
                
                          
   
    scalex = -0.15
    scaley = 1.08
    labels = ['$\mathrm{(a)}$','$\mathrm{(b)}$','$\mathrm{(c)}$','$\mathrm{(d)}$']
    
    for sub_axs,sub_label in zip(np.ravel(axs),labels):
       sub_axs.text(s = sub_label,**text_coords(ax=sub_axs, scalex=scalex, scaley=scaley),fontsize = SMALL_SIZE)
    
    plt.suptitle('$L = '+str(N)+'\quad \mathrm{(mid)}$')
    plt.tight_layout()
    plt.show()
    
            
    if is_save == 1:
        
        save_name = input('--- Input the save fig name: (press <ENTER> for not to save)')
        if save_name!="":
            fig.savefig(Dir.save_dir+save_name+'.pdf', bbox_inches='tight')
            
           
            
def text_coords(ax=None,scalex=0.9,scaley=0.9):
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    return {'x':scalex*np.diff(xlims)+xlims[0],
        'y':scaley*np.diff(ylims)+ylims[0]}


#if __name__ == "__main__": 
#    fit_and_plot(sub_N,y_data,Dir,x_labels=x_labels,y_labels = y_labels)