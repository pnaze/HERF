import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj, ellipk, roots_legendre

beta = lambda0 = kappa0 = 1.0
t = np.linspace(0, 20, 801)

def fields(N=1200, M=1024, nh=14):
    x, w = roots_legendre(N)
    E = 35*(x+1)/beta
    w = 35*w/beta
    nu = (lambda0**2+4*kappa0*E)**0.25
    m = (1-lambda0/nu**2)/2
    b2 = 4*E/(nu**2+lambda0)
    K = ellipk(m)
    omega = np.pi*nu/K
    W = w*K/nu*np.exp(-beta*E)
    W /= W.sum()
    phi = 2*np.pi*np.arange(M)/M
    cn = ellipj(K[:,None]*phi/np.pi, m[:,None])[1]
    q2 = b2[:,None]*cn**2
    psi, C = [], []
    for r in (2,4):
        A = q2**(r//2)/r
        a = np.fft.rfft(A,axis=1).real/M
        C.append(beta*np.dot(W,a[:,0]**2))
        g = np.zeros_like(t)
        for n in range(1,nh+1):
            g += np.cos(n*t[:,None]*omega) @ (2*beta*W*a[:,n]**2)
        psi.append(g)
    return np.array(psi), np.array(C)

def canonical(C,N=240,substeps=10):
    x,w = roots_legendre(N)
    qmax = 4*(beta*kappa0)**(-0.25)
    pmax = 9/np.sqrt(beta)
    q,p = np.meshgrid(qmax*(x+1)/2,pmax*x,indexing='ij')
    H = p*p/2 + lambda0*q*q/2 + kappa0*q**4/4
    W = w[:,None]*w[None,:]*np.exp(-beta*H)
    W /= W.sum()
    A02 = W*q*q/2
    A04 = W*q**4/4
    g = np.empty((2,len(t)))
    dt = (t[1]-t[0])/substeps
    for j in range(len(t)):
        q2 = q*q
        g[0,j] = beta*np.sum(A02*q2/2)-C[0]
        g[1,j] = beta*np.sum(A04*q2*q2/4)-C[1]
        if j == len(t)-1:
            break
        for _ in range(substeps):
            p -= dt/2*q*(lambda0+kappa0*q*q)
            q += dt*p
            p -= dt/2*q*(lambda0+kappa0*q*q)
    return g

if __name__ == '__main__':
    pred,C = fields()
    brute = canonical(C)
    print('C:',C)
    print('Initial values:',pred[:,0],brute[:,0])
    print('Maximum discrepancies:',np.max(np.abs(pred-brute),axis=1))
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix',
                         'font.size':10,'axes.linewidth':.75,'pdf.fonttype':42})
    fig,axs = plt.subplots(2,1,figsize=(3.4,4.35),sharex=True)
    for i,ax in enumerate(axs):
        r = (2,4)[i]
        ax.plot(t,pred[i],color='black',lw=1.25,label='Hamiltonian fields')
        ax.plot(t[::12],brute[i,::12],ls='none',marker='o',ms=3.1,
                mew=.85,mfc='none',mec='#bd351b',label='Canonical average')
        ax.axhline(0,color='.65',lw=.65,ls='--',zorder=0)
        ax.set_ylabel(r'$\Psi_0^{(%d)}(t)$'%r)
        ax.text(.03,.92,(' (a)' if i==0 else ' (b)')+r'  $r=%d$'%r,
                transform=ax.transAxes,ha='left',va='top')
        ax.tick_params(direction='in',top=True,right=True)
        ax.set_xlim(0,20)
    axs[0].legend(frameon=False,fontsize=8,loc='upper right')
    axs[-1].set_xlabel(r'$t$')
    fig.tight_layout(pad=.65,h_pad=.8)
    fig.savefig('duffing_relaxation_comparison.pdf',bbox_inches='tight')
