# This code computes lower and upper bounds on the asymptotic independence ratio of random regular graphs.
# It is a supplementary material for the following paper:
#   "Boosted second moment method in random regular graphs"
#     by Balázs Gerencsér and Viktor Harangi
#   https://arxiv.org/abs/2510.12600

import mpmath
mpmath.mp.dps=50

def h(x):
    return -x*log(x) if x>0 else 0

def H_fun(x):
    return h(x)+h(1-x)

# one independent set (first moment): vertex entropy 
def fm_vtx(al):
    return h(al)+h(1-al)

# one independent set (first moment): edge entropy 
def fm_edge(al):
    return 2*h(al)+h(1-2*al)

# one independent set (first moment): exp.rate = d/2*fm_edge(al)-(d-1)*fm_vtx(al)
def fm_entropy(d,al):
    return h(al)+d/2*h(1-2*al)-(d-1)*h(1-al)

# two independent sets (second moment): vertex entropy 
def sm_vtx(al,be):
    return h(be)+2*h(al-be)+h(1-2*al+be)

# auxiliary function: optimal ga for sm_edge
def ga_fun(al,be):
    return al-0.5+sqrt((al-0.5)^2+(al-be)^2)

# two independent sets (second moment): edge entropy 
def sm_edge(al,be):    
    ga=ga_fun(al,be)
    return 2*h(be)+2*h(ga)+4*h(al-be-ga)+h(1-4*al+2*be+2*ga)

# two independent sets (second moment): exp.rate 
def sm_entropy(d,al,be):
    vtx=h(be)+2*h(al-be)+h(1-2*al+be)
    return 0.5*d*sm_edge(al,be)-(d-1)*sm_vtx(al,be)

#
# BOUNDS for the asymptotic independence ratio alpha*_d of random d-regular graphs

# first moment bound for alpha*_d (Bollobas bound)
def alpha_fm(d,float_res=True):
    fun=lambda al: fm_entropy(d,al)
    #al=find_root(fun,1./d,0.49)
    al_fm=mpmath.findroot(fun,(1.25*log(d)/d,1.5*log(d)/d))
    return float(al_fm) if float_res else al_fm

# lower bound for alpha*_d  (Shearer1983)
def shearer_or(d):
    return (d*log(d+0.)-d+1)/(d-1)^2

# lower bound for alpha*_d  (Shearer1991)
def shearer(d):
    assert d>=2
    if d<=2:
        return 79./151
    else:
        return (1+d*(d-1)*shearer(d-1))/(d^2+1)

# lower bound for alpha*_d  (Wormald1995,Thm 4)    
def wormald_simple(d):
    return 0.5-0.5/(d-1.)^((2./(d-2)))

# formula for alpha*_d as in Ding-Sly-Sun ('á la frozen configuration')
def ding_sly_sun(d, show_plot=False, float_res=True):
    la=lambda q: q*(1-(1-q)^(d-1))/(1-q)^d
    al=lambda q: q*(1-q+d*q/2/la(q))/(1-q^2*(1-1/la(q)))
    fun= lambda q: -log(1-q*(1-1/la(q)))-(d/2-1)*log(1-q^2*(1-1/la(q)))-al(q)*log(la(q))
    a=1.5*log(d)/d
    b=1.75*log(d)/d
    #q0=find_root(f,a,b)
    q0=mpmath.findroot(fun,(1.5*a,1.75*a))
    if show_plot:
        plot(f,a,b).show()
        print("root=",q0)
    return float(al(q0)) if float_res else al(q0)

# 1-RSB upper bound for alpha*_d  'á la interpolatin method'
# should give the same value as ding_sly_sun
def one_rsb(d,show_plot=False):
    def fun(q):
        be=q*( (1-q)^(-d)-(1-q)^(-1) )
        enum=log(1+(be-1)*(1-q)^d)-d/2*log(1-q^2*(1-1/be))
        return enum/log(be)
    a=1.*log(d)/d
    b=2.*log(d)/d
    res=find_local_minimum(fun,a,b)
    if show_plot:
        plot(fun,a,b).show()
        print("minimum attained at:",res[1])
    return res[0]

# finds  max {x : func(x) is True}
def max_True(func,x0,x1,thr=1e-8):
    assert x1>x0

    if not func(x0):
        return None
    if func(x1):
        return x1

    diff=x1-x0
    while x1-x0>thr:
        diff=diff/2
        x=x0+diff
        if func(x):
            x0=x
        else:
            x1=x
    return x0    

# finds the largest alpha for which second moment method works
# --> the Markovian independent set of density alpha is typical
def alpha_sm(d,show_plot=False):
    def sm_works(al):
        fun=lambda be: sm_entropy(d,al,be)
        #val1=find_local_maximum(fun,0,0.99*al^2)[0]
        val2=fun(al^2)
        val3=find_local_maximum(fun,1.01*al^2,al)[0]
        #return val1<=val2 and val3<=val2
        return val3<=val2
    
    al_1rsb=ding_sly_sun(d)
    al_sm=max_True(sm_works,0.75*al_1rsb,al_1rsb)
        
    if show_plot:
        f_sm=lambda be: sm_entropy(d,al_sm,be)
        indep_val=f_sm(al_sm^2)
        def diff(be):
            val=f_sm(be)
            val_max=indep_val-0.125/d
            return val if val>val_max else val_max

        fig=plot(diff,0,al_sm)
        fig+=plot(indep_val,0,al_sm,color='red')
        fig+=point((al_sm^2,diff(al_sm^2)),color='red')
        fig.show()
        
    return al_sm

# augments Markovian independent set of density 'al' and returns new density
def augmented_alpha(d,al):
    p=(1-2*al)/(1-al)
    q=p*(1-p^(d-1))
    assert (d-1)*p^(d-1)<1  #so that full-zero components are a.s. finite
    full_zero=(1-al)*p^d
    comp_one=(1-al)*q^d    
    #survive=p^(d-1)
    #die=1-survive
    #comp_one=(1-al)*p^d*die^d
    #comp_two=d*(1-al)*p^d*survive*die^(2*d-2)
    return al+(full_zero+comp_one)/2
