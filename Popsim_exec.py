# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:13:24 2022

@author: Gebruiker
"""




import sys
import numpy as np
from math import * 
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
import random
from scipy.optimize import curve_fit
from scipy.integrate import quad
from matplotlib import cm
import matplotlib.gridspec as gridspec
import os
from os.path import exists
import matplotlib.lines as mlines


from sympy import nsolve
from sympy.abc import x
from sympy import I


import time

sys.setrecursionlimit(500000)

"sys.argv should consist of [seed,count,tmax,eps,tau_ratio]"
 
"""This construction to give arguments was only used to generate figure 4"""

if len(sys.argv)>1:
    run_sim=True
    args=sys.argv[1:]
    print(args)
else:
    run_sim=False
    """these are the default arguments.
    although they're not used, I needed them to prevent some errors"""
    args=['0.5','1','8','10','3']

    
eps_arg=float(args[0])
tau_ratio_arg=float(args[1])
tmax_arg=float(args[2])
count_arg=int(args[3])
seed_arg=int(args[4])


#np.random.seed(seed_arg)   

with open('output'+str(seed_arg)+'.txt','w') as ff:
        ff.write('sys.argv:'+str(sys.argv))
        ff.write('\n')

np.set_printoptions(precision=3)


"""Watch out, log-size u is scaled such that divisions are at u=1, but seta does match the paper.
"""

"""This function runs the population simulation a number of 'count' times for a time of tmax. 
ml is \bar\lambda
gam is 1/(\bar\lambda \tau_{cor})
z is \tau_{cor} \sigma_\lambda^2/\bar\lambda
tmax is the simulation time
nmax is the maximum number of generations (for determining the array size)
onecell is either True or False, if true then there will be one cell with size ln(m)=u0 at t=0, otherwise we have 
the three cells out of phase
ub is the size of birth of the first cell (for determining division size)
seta is \sigma_\eta the division noise
beta is the division size correlation parameter
svb can be changed to allow for asymmetric division, but we keep it at 0 for all plots in the text
"""

def OUsim(ml=np.log(2),gam=1,z=0.003,count=1,tmax=12,nmax=28,onecell=True,u0=0,ub=0,seta=0.1,beta=0.5,svb=0):
    
    if onecell==True:
        """input u0 and ub are scaled on [0,log(2)]"""
        ulist=np.array([u0/np.log(2)])
    else:
        """ulist is always scaled on [0,1]"""
        ulist=np.array([0,0.2493467,0.6383031])
    
    
    dt=0.01
    tsteps=int(tmax/dt)
    
    omega=2*np.pi/np.log(2)
    
    
    D=z*ml
    
    thx=gam*ml
    sl=np.sqrt(D*thx)
    sx=sl*np.sqrt(2*thx)
    
    """the following constants are used to allow for a time evolution of \lambda_t and \int^t_0 \lambda_sds simultaneously
    as a two-dimensional Ornstein-Uhlenbeck process, that is exact even for large dt"""
    """it is not necessary to do this for small enough dt, but since the constants only need to be calculated once 
    for every set of parameters, you may as well"""
    
    r1=np.exp(-thx*dt)
    r1c=(1-r1)/thx
    
    VarUt=sx**2/(thx**2)*((1-np.exp(-2*thx*dt))/(2*thx)-(1-np.exp(-thx*dt))*2/thx+dt)
    CovUtlt=sx**2/(2*thx**2)*(1-2*np.exp(-thx*dt)+np.exp(-2*thx*dt))
    Varlt=sx**2/(2*thx)*(1-np.exp(-2*thx*dt))
    
    A=np.sqrt(VarUt)
    B=CovUtlt/A
    C=np.sqrt(Varlt-B**2)
    
    Acor=A/np.log(2)
    r1ccor=r1c/np.log(2)
    
    dU=dt/np.log(2)*ml
    
    """The following function will be recursively executed once for every cell at every timestep and update the population until tmax time has passed"""
    
    """
    ti is the time index
    u is the size \ln(m) of the cell
    udd is the size at division \ln(m_d)
    x is the growth rate
    nmax is the maximum number of generations
    tmax is the simulation time
    """
    def FOU(Vtab,ti,u,udd,x,n,nmax,tsteps):      
        """All time-series info is stored in Vtab, 
        Vtab[0] is the total cell count
        Vtab[1] is the total cell mass
        Vtab[2] is the collective amplitude
        Vtab[3] and Vtab[4] are unusued"""  
        for i in range(5):
            Vtab[ti,i] = Vtab[ti,i]+ [1,2**u,2**(u*(1+1j*omega)),2**u*np.exp(x/thx),np.exp((1+1j*omega)*x/thx)*2**(u*(1+1j*omega))][i]

        
        """check whether the cell still fits within the cell array and simulation time"""
        if ti<tsteps-1 and n<nmax:
            
            """we update the cell size and state for +dt time"""
            
            X=np.random.normal()
            Y=np.random.normal()
            
            u=u+dU+r1ccor*x+Acor*X
            x=r1*x+B*X+C*Y
           
            """if the cell's current size is below division size, increase time-index by one and repeat"""
            if u< udd+1:
                return FOU(Vtab,ti+1,u,udd,x,n,nmax,tsteps)
                """otherwise, division happens and we create two new cells."""
            else:     
                xvb=svb
                ubb1= u-1 + np.log(1-xvb)/np.log(2)
                ubb2= u-1 + np.log(1+xvb)/np.log(2)
                
                udd1=beta*ubb1+seta*np.random.normal()/np.log(2)
                udd2=beta*ubb2+seta*np.random.normal()/np.log(2)
                """upon division, two new functions FOU are called upon"""
                Vtab=   FOU(Vtab,ti+1,ubb1,udd1,x,n+1,nmax,tsteps)
                Vtab=FOU(Vtab,ti+1,ubb2,udd2,x,n+1,nmax,tsteps)

        return Vtab
    """this function initiates the population"simulation """
    def FOU_exec(q):

        with open('output'+str(seed_arg)+'.txt','a') as ff:
                ff.write('count:'+str(q))
                ff.write('\n')

        Vtab=np.zeros([tsteps,5],dtype=complex)
        """ulist contains all the log cell sizes of the initial population"""
        for u00 in ulist:
            udd1=ub/np.log(2)+seta*np.random.normal()
            Vtab=FOU(Vtab,0,u00,udd1,0,0,nmax,tsteps)

        return Vtab
    startcalc=time.time()
    
    V_array_new=np.array([FOU_exec(q) for q in range(count)])
    #savestr='data_count='+str(count)+'_tmax='+str(tmax)+'_dt='+str(dt)+'_gam='+str(gam)+'_z='+str(z)+'_onecell='+str(onecell)+'_u0='+str(u0)+'_ub='+str(ub)+'_seta='+str(seta)+'_beta='+str(beta)+'_svb='+str(svb)+'.npy'    
    #np.save(savestr, [V_array_new,count,tmax,gam,z,ml,dt,u0,ub,seta,beta,onecell,svb])
    #np.save('data_recent.npy', [V_array_new,count,tmax,gam,z,ml,dt,u0,ub,seta,beta,onecell,svb])
    endcalc=time.time()
    print(endcalc-startcalc)
    
    """this line tracks progress in a text file while the simulation is running"""
    with open('output'+str(args[0])+str(args[1])+str(args[2])+str(args[3])+str(args[4])+'.txt','a') as ff:
            ff.write('duration:'+str(endcalc-startcalc))
            ff.write('\n')
            
    output=[V_array_new,count,tmax,gam,z,ml,dt,u0,ub,seta,beta,onecell,svb]

    return output

"""use the following function if you want to run a number of simulations with fixed parameters and save the population data"""
"""This function was used to generate figs 1,2,3"""

def save_OUsim(ml=np.log(2),gam=1,z=0.003,count=1,tmax=12,nmax=24,onecell=True,u0=0,ub=0,seta=0.1,beta=0.5,svb=0):
    output=OUsim(ml,gam,z,count,tmax,nmax,onecell,u0,ub,seta,beta,svb)
    np.save('data_recent.npy',np.array(output,dtype=object))
    return



# %% PLotting

"""Change this to the name of any output file generated using save_OUsim"""
temp='data_recent.npy'

"""The following function was used to generate figures 1,2,3"""

"""pt is the plot type, it can be set to either Leff, logLeff, N, or logN for figures 1 and 2
or Psi, A, Psidist or Adist for figure 3
qsel allows one to select a different simulation to show for Leff, logLeff, N or logN if the data contains multiple simulations
fl is the label that appears in the top left corner to distinguish potential subplots
"""

def OUplot(savestr=temp,pt='Leff',qsel=0,fl=''):

    data=np.load(savestr,allow_pickle=True)
    print(data)
    V_array,count,tmax,gam,z,ml,dt,u0,ub,seta,beta,onecell,svb=data
    
    
    dt=0.01
    print(seta,beta)
    
    #print(data[1:])
    
    if onecell==True:
        ulist=np.array([u0])
    else:
        """here ulist is scaled on [0,log(2)]"""
        ulist=np.array([0,0.172834 ,0.442438])
    
    
    #tsteps=len(V_array[0,0])

    tsteps=int(tmax/dt)
    


    
    figsave=str(fl)+'_'+savestr[:-4]+pt+str(qsel)+'.pdf'
    figsave2=str(fl)+'_'+savestr[:-4]+pt+str(qsel)+'.png'
    
    D=z*ml
    

    omega=2*np.pi/np.log(2)
    su=seta*np.sqrt(1/(1-beta**2))
    


    
    rosc=omega**2*D

    mu0=ml+D
    mu1=(1+1j*omega)*ml+(1+1j*omega)**2*D
    rosc=mu0-np.real(mu1)
    
    eps=2-1/(2*z)*(np.sqrt(1+8*z+8*(1-omega**2)*z**2)-1)
    
    

    V1 = np.sum( np.exp(ulist*(1+1j*omega)))
    V0=np.sum( np.exp(ulist))
    
    V1inf2_abs_mean_theory = np.abs(V1**2) +np.sum( np.exp(2*ulist)*( np.exp(-eps*ulist)/(2**(1-eps)-1)-1) )
    Psiinf_root_mean_square = np.exp(3/2*omega**2*z/gam)*np.sqrt(V1inf2_abs_mean_theory )/V0

    Psiinf_abs_mean = np.exp(3/2*omega**2*z/gam)*np.abs(V1/V0)
    
    fig = plt.figure(figsize=(7,4.5))

    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    
    ax.text(-0.07,1.2,fl,fontsize=26,horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,weight='bold')
    
    t_array=np.arange(tsteps)*dt

    
    t_array=np.arange(tsteps)*dt
    t_final=t_array[-1]
    
    
        
    if pt=='Leff' or pt=='logLeff' or pt=='N' or pt=='logN':

        delt=0.1
        ipb=int(delt/dt)
        binsteps=int(tmax/delt)-1
        tmax=t_array[-1]
        
        cutoff=0.001
        
        if rosc<mu0/2:
            show_oscillations=True
            fitlabel='oscillation fit envelope'
        else:
            show_oscillations=False
            fitlabel='ensemble oscillation envelope'
        
        tcut=7
        ticut=int(tcut/dt)
        
        
        for q in [qsel]:
            Leff_array=np.array([ np.log(V_array[q,(i+1)*ipb,0]/V_array[q,i*ipb,0]) for i in range(binsteps)])/delt/ml
            N_array=np.array([ float(V_array[q,(i)*ipb,0]) for i in range(binsteps)])
            t_delt_array= np.array([t_array[i*ipb] for i in range(binsteps)])+delt/2
            
            print(N_array)
            print(len(N_array),len(t_delt_array))
            
            osc_asymp_array=(mu0+2*np.exp(-omega**2*su**2/2)*np.real(np.exp(1j*omega*su**2)*(np.exp(-(mu1)*delt/2)-np.exp((mu1)*delt/2))/delt/(1+1j*omega)*V_array[q,-1,2]/V_array[q,-1,1]*np.exp((mu1-mu0)*(t_array-tmax))))/ml
            osc_asymp_array=(mu0+2*np.exp(-omega**2*su**2/2)*np.real(np.exp(1j*omega*su**2)*(mu1-mu0)/(1+1j*omega)*V_array[q,-1,2]/V_array[q,-1,1]*np.exp((mu1-mu0)*(t_array-tmax))))/ml
            osc_env_asymp_array=(mu0+2*np.abs((np.exp(mu1*delt)-1)/delt/(1+1j*omega)*V_array[q,-1,2]/V_array[q,-1,1])*np.exp(-rosc*(t_array-tmax))*np.exp(-omega**2*su**2/2))/ml

            #def f(t,A,phi):
            #    return (mu0+ A*np.exp(-rosc*t)*np.cos(Om*t+phi))/ml
            
            
            #delticut=int(18*ipb)
            
            #params=curve_fit(f,t_delt_array[delticut:],Leff_array[delticut:],sigma=1/np.sqrt(N_array[delticut:]))
            #Afit,phifit=params[0]
            
            
            
            #print('params'+str(params))
            
            fluct_env_array=(mu0+np.sqrt(mu0*2*np.log(2)/V_array[q,:,1]/delt))/ml
            if show_oscillations==False:

                osc_env_asymp_array=(mu0+2*np.abs((np.exp(mu1*delt)-1)/delt/(1+1j*omega)*V_array[q,0,2]/V_array[q,0,1])*np.exp(-rosc*t_array)*np.exp(-omega**2*su**2/2))/ml
            
            if pt=='N':
                ax.tick_params(axis='both', which='major', labelsize=22)
                ax.set_xlabel('time '+ r'$ t/\tau_{div}$',fontsize=22)
                ax.set_ylabel(r'cell count $N(t)$',fontsize=22)
                ax.plot(t_array,V_array[qsel,:,0])
                
            if pt=='logN':
                ax.tick_params(axis='both', which='major', labelsize=22)
                ax.set_xlabel('time '+ r'$ t/\tau_{div}$',fontsize=22)
                ax.set_ylabel(r'log cell count $\log_{10}(N(t))$',fontsize=22)
                ax.plot(t_array,np.log(V_array[qsel,:,0])/np.log(10))
            
            if pt=='Leff':
                
                ax.tick_params(axis='both', which='major', labelsize=22)
                ax.set_ylim([0.4,2])
                ax.set_xlim([0,tmax-delt])
                ax.set_xlabel('time '+ r'$ t/\tau_{div}$',fontsize=22)
                ax.set_ylabel('cell count growth '+r'$\Lambda(t)/\bar\lambda$',fontsize=20)
                ax.bar(t_delt_array,Leff_array,color='#1f77b4',width=delt,label='simulation')
                ax.plot(t_array,np.zeros(tsteps)+mu0/ml,color='green',linewidth=2,alpha=0.5,label='asymptotic growth rate')
                #ax.plot(t_array,f(t_array,Afit,phifit),color='red',linewidth=2,alpha=0.5)
                #ax.plot(t_array,osc_env_array,color='black',label='oscillation envelope',linewidth=1)
                ax.plot(t_array,fluct_env_array,color='red',label='fluctuation envelope',linewidth=1)
                
                
                if show_oscillations==True:
                    ax.plot(t_array[ticut:],osc_asymp_array[ticut:],color='black',label='oscillation fit',linewidth=1)
               
                ax.legend(fontsize=14) 
                
                plt.rcParams["figure.autolayout"] = True
               
            log_Leff_array=(np.log(Leff_array-mu0/ml)-np.log(cutoff))*(Leff_array-mu0/ml>cutoff)+np.log(cutoff)

            log_osc_env_asymp_array=np.log(osc_env_asymp_array-mu0/ml)
            log_fluct_env_array=np.log(fluct_env_array-mu0/ml)
            
            #print(log_Leff_array)    
            
            ybot=-5
            
            if pt=='logLeff':
                ax.tick_params(axis='both', which='major', labelsize=22)
                ax.bar(t_delt_array,log_Leff_array-ybot,bottom=ybot,color='#1f77b4',width=delt,label='simulation')
                ax.plot(t_array,log_osc_env_asymp_array,color='black',label=fitlabel,linewidth=1)
                ax.plot(t_array,log_fluct_env_array,color='red',label='fluctuation evelope',linewidth=1)
                
                ax.set_ylim([ybot,2])
                ax.set_xlim([0,tmax-delt])
                ax.legend()
                
                ax.set_xlabel('time '+ r'$ t/\tau_{div}$',fontsize=22)
                ax.set_ylabel(r'log deviation',fontsize=20)
                
                plt.rcParams["figure.autolayout"] = True
                
                ax.legend(fontsize=14) 
           
            

        
    if pt=='Psi' or pt=='A':
        Apref=1
        if pt=='A':
            Apref=2* np.abs(mu1/mu0/(1+1j*omega))
        if onecell==True:
            ymax=1.4*Apref
        else:
            ymax=1*Apref
        
        for q in range(count):
            #ax.set_yticks([-0.2,0,0.2])
            #ax.set_xticks([0,2,4,6])
            ax.tick_params(axis='both', which='major', labelsize=22)
            ax.set_xlim([0,tmax])
            ax.set_ylabel("relative amplitude",fontsize=22)
            ax.set_xlabel(r"time $t/\tau_{div}$",fontsize=22)
            ax.set_ylim([0,ymax])
            lw=1/(1+count/50)
            ax.plot(t_array,Apref*np.abs(V_array[q,:,2])/V_array[q,:,1],color='#1f77b4',alpha=lw)
        ax.plot([],[],color='#1f77b4',alpha=1,label=r'simulations')
        ax.plot(t_array,Apref*np.sqrt(np.mean(np.abs(V_array[:,:,2]/V_array[:,:,1])**2,axis=0)),color='black',linewidth=1.5,label=r'average amplitude')
        ax.plot(t_array,Apref*np.abs(np.mean(V_array[:,:,2]/V_array[:,:,1],axis=0)),color='brown',linewidth=1.5,label=r'ensemble amplitude')
        #ax.plot(t_array,Adcor*np.abs(np.mean(2/(1+1j*omega)*V_array[:,:,2]/V_array[:,:,1],axis=0)),color='red',linewidth=1.5,label=r'amplitude of ensemble average')
        
        ax.plot(t_array,Apref*Psiinf_root_mean_square*np.exp(-t_array*rosc),'--',color='darkgreen',label=r'$\bar A e^{-r t}$')
        ax.plot(t_array,Apref*Psiinf_abs_mean*np.exp(-t_array*rosc),'--',color='red',label=r'$A_{ens} e^{-r t}$')
        ax.legend(fontsize=14)
        
        
        
    if pt=='Psidist' or pt=='Adist':
    
        Apref=1
        if pt=='Adist':
            Apref=2* np.abs(mu1/mu0/(1+1j*omega))
            
            
            
        
        ax.set_xlabel(r"amplitude prefactor $A$",fontsize=22)
        ax.hist(Apref*np.abs(V_array[:,-1,2]/V_array[:,-1,1])*np.exp(rosc*t_final),color=u'#1f77b4',bins=40,density=True,label=r'simulations')
        

        ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.set_ylabel("relative abundance",fontsize=20)
        
        mw=3
        
        Psiinf_ens_est=np.abs(np.mean(V_array[:,-1,2]/V_array[:,-1,1],axis=0))*np.exp(rosc*t_final)
        #print(Psiinf_ens_est)
        Psiinf_root_mean_square_est=np.sqrt(np.mean(np.abs(V_array[:,-1,2]/V_array[:,-1,1])**2,axis=0))*np.exp(rosc*t_final)
        
        #Psiinf_root_mean_square_est=np.mean(np.abs(V_array[:,-1,2]/V_array[:,-1,1]),axis=0)*np.exp(rosc*t_final)
        if onecell==True:
            xmax=2.5*np.max(Apref*Psiinf_root_mean_square_est)
        else:
            xmax=3*np.max(Apref*Psiinf_root_mean_square_est)
        ax.set_xlim([-0.05,xmax])
        ymax=3.5/xmax
        ax.set_ylim([0,ymax])
        
        ax.plot([Apref*Psiinf_ens_est,Apref*Psiinf_ens_est],[0,ymax],'-',color='brown',label=r'simulated $A_{ens}$',linewidth=mw)
        
        
        
        ax.plot([Apref*Psiinf_abs_mean,Apref*Psiinf_abs_mean],[0,ymax],'--',color='red',label=r'theory $A_{ens}$',linewidth=mw)

        
        ax.plot([Apref*Psiinf_root_mean_square_est,Apref*Psiinf_root_mean_square_est],[0,ymax],'-',color='black',label=r'simulated $\bar A$',linewidth=mw)
        ax.plot([Apref*Psiinf_root_mean_square,Apref*Psiinf_root_mean_square],[0,ymax],'--',color='darkgreen',label=r'theory $\bar A $',linewidth=mw)
        
        
        
        
        
        if onecell==False:
            x_array=np.arange(100)/100*xmax
            ax.plot(x_array,x_array*2/(Apref*Psiinf_root_mean_square)**2*np.exp(-x_array**2/(Apref*Psiinf_root_mean_square)**2),'--',color='black',label=r'$\chi_2$ distribution')
       
        ax.legend(fontsize=16)
        
          

    
    fig.savefig(figsave)
    fig.savefig(figsave2)
    fig.savefig('recent'+pt+'.pdf')
    fig.savefig('recent'+pt+'.png')

"""This function is used to call multiple simulations if arguments are given to this python file",
this was only used to generate figure 4"""

def sim():
    savestr='recent'+str(args)+'.npy'

    onecell0=True



    
    
    gam0=1/(tau_ratio_arg*np.log(2))
    z0=eps_arg/(8*(np.pi/np.log(2))**2)
    beta0=0.5
    u00=0
    beta0=0.5
    ub0=0
    seta0=0
    svb0=0
    

            
    with open('output'+str(args)+'.txt','a') as ff:
            ff.write('')
            ff.write('\n')
    
    output=np.array([OUsim(gam=gam0,z=z0,count=count_arg,tmax=tmax_arg,u0=u00,ub=ub0,seta=seta0,beta=beta0,onecell=onecell0,svb=svb0)],dtype=object)
    print(output)
    np.save(savestr,output)
    return
            

"""This function is used to generate figure 4"""
"""the plottype can be set by changing p='ratio', p='Aens' or p='Abar'"""
"""the letter or text in the top left of the plot can be changed by setting l='text'"""
    
    
def plot(p='ratio',l='a'):
    

    
    """This section goes through all data files starting with 'recent[' in the same folder as this python file, 
    and compiles them into one big array"""
    
    data_list=[]

    for file in os.listdir('.'):
        if file[:7]=='recent[':
            #print('yes')
            data_new=np.load(file,allow_pickle=True)[0]
            data_list=data_list+[data_new]
            #print(file)
    #print(len(data_list))
    
    
    lib={}
    
    for data in data_list:
        gam=data[3]
        z=data[4]
        
        if gam not in lib:
            lib[gam]={}
        else:
            if z not in lib[gam]:
                lib[gam][z]=data
            else:
                lib[gam][z][0]=np.concatenate((lib[gam][z][0],data[0]))
                lib[gam][z][1]+= data[1]

    
    fig = plt.figure(figsize=(7,7))
    ax= fig.add_subplot(111)
    

    
    ax.text(-0.11,1.1,l,fontsize=32,horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,weight='bold')
    om=2*np.pi/np.log(2)
    d_eps=0.01
    
    
    
    if p=='ratio':
        eps_max=1.3
    
        ax.set_ylim([0,1.1])
        ax.set_xlim([0,eps_max])
        ax.set_ylabel(r'$A_{ens}/\bar A$',fontsize=17)
        
    if p=='Abar':
        eps_max=1.3
    
        ax.set_ylim([0,16])
        ax.set_xlim([0,eps_max])
        ax.set_ylabel(r'$\bar A$',fontsize=21)
    
    if p=='Aens':
        eps_max=1.3
    
        #ax.set_ylim([1.5,5.5])
        ax.set_ylim([0,16])
        ax.set_xlim([0,eps_max])
        ax.set_ylabel(r'$A_{ens}$',fontsize=21)
        

    eps_array=np.arange(int(eps_max/d_eps))*d_eps
    eps1_array=np.arange(int((eps_max-1)/d_eps))*d_eps+1
    
    
    clist=plt.rcParams['axes.prop_cycle'].by_key()['color']
    mlist=["P",'D','s']
    
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.set_xlabel(r'$\epsilon$',fontsize=22)
    ax.set_xticks(np.array([0.5,1.0]))
    
    
    for i_tau in range(len(lib)):
        gam=list(lib)[i_tau]
        showlabel=1
        for z in lib[gam]:
            
            

            V_array,count,tmax,gam,z,ml,dt,u0,ub,seta,beta,onecell,svb=lib[gam][z]
            
            #print(gam,z,len(V_array))
            su=seta/np.sqrt(1-beta**2)
            
            eps=z*2*om**2
            r=om**2*z*ml
            
            if onecell==True:
                
                ulist=np.array([0])
            else:
                ulist=np.array([0,0.2493467,0.6383031])
            
            t_offset=0
            ti_offset=int(t_offset/dt)
            tmax_corrected=tmax-t_offset
            
            
            
            Psi_tilde_abs2_array=np.abs(V_array[:,-1-ti_offset,2]/V_array[:,-1-ti_offset,1])**2*np.exp(2*r*tmax_corrected)
            Psi_tilde_array=V_array[:,-1-ti_offset,2]/V_array[:,-1-ti_offset,1]*np.exp(r*tmax_corrected)
            
            Avg_Psi_tilde_abs2=np.mean(Psi_tilde_abs2_array)
            Avg_A_bar=2*np.exp(-om**2/2*su**2)*np.sqrt(Avg_Psi_tilde_abs2)
            
            Avg_Psi_tilde=np.abs(np.mean(Psi_tilde_array))
            Avg_A_ens=2*np.exp(-om**2/2*su**2)*Avg_Psi_tilde
            
        
            print('eps='+str(eps)+'    A_ens/A_bar='+str(Avg_A_ens/Avg_A_bar))
            #ax.errorbar(eps,1/Avg_A_bar,yerr=[[Avg_A_bar_recip_max],[Avg_A_bar_recip_min]],marker=mlist[i_tau],markersize=8,color=clist[i_tau])
            
            
            if p=='ratio':
                ax.scatter(eps,Avg_A_ens/Avg_A_bar,s=100,marker=mlist[i_tau],linewidth=2,facecolors='none',edgecolors=clist[i_tau],label=['',r'$\tau_{cor}/\tau_{div}=$'+str(1/(np.log(2)*gam))][showlabel])
            if p=='Abar':
                ax.scatter(eps,Avg_A_bar,s=100,marker=mlist[i_tau],linewidth=2,facecolors='none',edgecolors=clist[i_tau],label=['',r'$\tau_{cor}/\tau_{div}=$'+str(1/(np.log(2)*gam))][showlabel])
            if p=='Aens':
                ax.scatter(eps,Avg_A_ens,s=100,marker=mlist[i_tau],linewidth=2,facecolors='none',edgecolors=clist[i_tau],label=['',r'$\tau_{cor}/\tau_{div}=$'+str(1/(np.log(2)*gam))][showlabel])

            showlabel=0

        E_Psi_tilde_abs2=np.array([(np.abs(np.sum(2**((1+1j*om)*ulist)))**2 + np.sum(4**ulist*(2**(-eps0*ulist)/(2**(1-eps0)-1)-1)))/np.sum(4**ulist) for eps0 in eps_array])
        E_Psi_tilde_abs=np.array([np.abs(np.sum(2**((1+1j*om)*ulist))) for eps0 in eps_array])

        A_bar_theory=2*np.exp(-om**2/2*su**2+3/4*eps_array/gam) * np.sqrt(E_Psi_tilde_abs2)
        A_ens_theory=2*np.exp(-om**2/2*su**2+3/4*eps_array/gam) * E_Psi_tilde_abs
        
        if p=='Abar':
            ax.plot(eps_array,A_bar_theory,color=clist[i_tau],linewidth=3)

        if p=='Aens':
            ax.plot(eps_array,A_ens_theory,color=clist[i_tau],linewidth=3)

        
        ratio_theory=np.array([np.sqrt(2**(1-eps0)-1) for eps0 in eps_array])
        
    if p=='ratio':
        ax.plot(eps_array,ratio_theory,color='black',linewidth=3)
        ax.plot(eps1_array,np.zeros([len(eps1_array)]),color='black',linewidth=6)
    ax.plot([],[],color='black',linewidth=3,label='Theory')
    
        
        
        
    
    ax.legend(fontsize=18)
    fig.savefig('plot'+str(p)+'.pdf')
    fig.savefig('plot'+str(p)+'.png')
    
"""This activates sim() if arguments are given to the python file"""    
        
if run_sim==True:
    sim()

    
    
    

# %%


