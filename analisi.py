#import pyerrors as pe
import numpy as np
from obs import Obs


def gamma_analysis_work_rep(work):
    '''
    Gamma analysis for Variational F, delta F and ESS
    work shape : (meas,bs) assuming autocorrelations on repeated measures
    '''
    replicas_name = [f'ennsemble1|r0{i+1}' for i in range(work.shape[1])]
    #Loss
    obs1=Obs([work[:,i] for i in range(work.shape[1])], replicas_name)
    loss=np.mean(obs1)
    loss.gamma_method()

    #betaF
    w0=np.mean(work)
    obs1= Obs([np.exp(-work[:,i]+w0) for i in range(work.shape[1])], replicas_name)
    betaF=-np.log(np.mean(obs1))+w0
    betaF.gamma_method()

    #ESS
    obs2= Obs([np.exp(2*(-work[:,i]+w0)) for i in range(work.shape[1])], replicas_name)
    ESS=np.mean(obs1)**2/np.mean(obs2)
    ESS.gamma_method()

    return loss, betaF, ESS

def gamma_analysis_Z(work):
    '''
    direct Jarzynski, of course highly instable for large W
    '''
    replicas_name = [f'ennsemble1|r0{i+1}' for i in range(work.shape[1])]
    #Loss
    #Z
    obs1= Obs([np.exp(-work[:,i]) for i in range(work.shape[1])], replicas_name)
    Z=np.mean(obs1)
    Z.gamma_method()

    return Z


def compute_Binder(m_list):
    replicas_name = [f'ensemble1_{i+1}' for i in range(m_list.shape[1])]
    
    # calcolo magnetizzazione (media spaziale su L^3)
    #m = np.mean(phi_list, axis=(2,3,4))  # shape: (nmeas, BatchSize)
    # m^2 e m^4 matrices
    #m2 = m_list**2
    #m4 = m_list**4
    # costruisco le Obs
    m2_Obs0 = Obs([m_list[:, i]**2 for i in range(m_list.shape[1])], replicas_name)
    m2_Obs = np.mean(m2_Obs0)
    m2_Obs.gamma_method()

    m4_Obs0 = Obs([m_list[:, i]**4 for i in range(m_list.shape[1])], replicas_name)
    m4_Obs = np.mean(m4_Obs0)

    m4_Obs.gamma_method()

    return m2_Obs**2 / m4_Obs
