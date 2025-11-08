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
    return al-1/2+sqrt((al-1/2)^2+(al-be)^2)

# two independent sets (second moment): edge entropy 
def sm_edge(al,be):    
    ga=ga_fun(al,be)
    return 2*h(be)+2*h(ga)+4*h(al-be-ga)+h(1-4*al+2*be+2*ga)

# two independent sets (second moment): exp.rate 
def sm_entropy(d,al,be):
    vtx=h(be)+2*h(al-be)+h(1-2*al+be)
    return d/2*sm_edge(al,be)-(d-1)*sm_vtx(al,be)

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
    return (d*log(float(d))-d+1)/(d-1)^2

# lower bound for alpha*_d  (Shearer1991)
def shearer(d):
    assert d>=2
    if d<=2:
        return 79./151
    else:
        return (1+d*(d-1)*shearer(d-1))/(d^2+1)

# lower bound for alpha*_d  (Wormald1995,Thm 4)    
def wormald_simple(d):
    return 0.5-0.5/(d-1)^((2./(d-2)))

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
    a=1*log(float(d))/d
    b=2*log(float(d))/d
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
    while diff>thr:
        diff=diff/2
        x=x0+diff
        if func(x):
            x0=x
        else:
            x1=x
    return x0    

# gives an upper bound for a concave function
def max_upper_bound(func,a0,a1,div=8):
    assert a1>a0
    diff=(a1-a0)/div
    val0=func(a0)
    val1=func(a1)
    slope0=(val0-func(a0-diff))*div
    slope1=(val1-func(a1+diff))*div
    assert slope0>=0
    assert slope1>=0
    assert val1<=val0+slope0
    assert val0<=val1+slope1    
    return (slope0*slope1+slope0*val1+slope1*val0)/(slope0+slope1)

# finds maximum of a concave function using ternary search                
def findmax_ternary_search(func,a0,a3,thr=1e-12):
    while True:
        diff=(a3-a0)/3
        if diff<thr:
            return max_upper_bound(func,a0,a3),(a0+a3)/2
        a1=a0+diff
        a2=a1+diff        
        if func(a1) >= func(a2):
            a3=a2
        else:
            a0=a1

# finds maximum of a concave function                
def findmax_quaternary_search(func,a0,a4,thr=1e-12):
    diff=(a4-a0)/2
    a2=a0+diff
    f0,f2,f4=func(a0),func(a2),func(a4)

    while diff>thr:
        diff=diff/2
        a1=a0+diff
        a3=a2+diff
        f1,f3=func(a1),func(a3)
        if f2>=f1 and f2>=f3:
            a0,f0=a1,f1
            a4,f4=a3,f3
        elif f1>=f3:
            a4,f4=a2,f2
            a2,f2=a1,f1
        else:
            a0,f0=a2,f2
            a2,f2=a3,f3

    return max_upper_bound(func,a0,a4),(a0+a4)/2


# checks if the second moment method works for the pair d and al=alpha
# --> the Markovian independent set of density alpha is typical
def alpha_sm_check(d,al,verbose=False,bits=300):
    RR=RealField(bits)
    al_prec=RR(al)
    fun=lambda be: sm_entropy(d,al_prec,RR(be))
    fun_float=lambda be: float(fun(be))
    
    x0=al_prec^2
    f0=fun(x0)
    
    #if x0 is not a local maximum, then SMM surely fails:
    if fun(1.001*x0)>f0:
        if verbose:
            print("The second moment method FAILS as the independent coupling does not belong to a local maximum.")
        return False    

    be_start=al/2
    if d<15:
        be_start=al/3
    if d<6:
        be_start=1.1*al^2
    if d==3:
        be_start=1.01*al^2
        
    f1,x1=find_local_maximum(fun_float,be_start,al,tol=1e-12,maxfun=5000)
    
    fdiff=f0-f1
    res=False
    if fdiff>1e-6*al:
        res=True
    elif fdiff>=0:
        # finding x1,f1 more precisely
        eps=1e-3
        xminus=(1-eps)*x1
        xplus=min( (1+eps)*x1,(x1+al)/2)
        #f1,x1=findmax_ternary_search(fun,RR(xminus),RR(xplus),thr=1e-12*al^1.5)
        f1,x1=findmax_quaternary_search(fun,RR(xminus),RR(xplus),thr=1e-12*al^1.5)
        res=f1<f0

    if verbose:
        def clipped(be):
            val=fun(be)
            return val if val>f0/2 else None

        print("plot of f_d,al(be) on [0,al] (clipped below the value at be=al)")
        print("the local maxima are indicated by (red and black) dots")
        set_verbose(-1)
        fig=plot(clipped,0,al)
        set_verbose(0)
        #fig+=plot(f0,0,al,color='red')
        fig+=point((x0,f0),color='red')#,size=16)
        fig+=point((x1,f1),color='black')#,size=16)
        fig.show()
        
        print(f"first local max at beta={x0}")
        fig=plot(fun,0.99*x0,1.01*x0)
        fig+=point((x0,f0),color='red')
        fig.show()
        
        print(f"second local max at beta={x1}")        
        rad=min(0.01*al^2,(al-x1)/2)
        fig=plot(fun,x1-rad,x1+rad)
        #fig+=plot(f0,x1-rad,x1+rad,color='red')
        fig+=point((x1,f1),color='black')
        fig.show()

        print(f"{f0} (first local max value)")
        print(f"{f1} (second local max value - upper bound)")
        if res:
            print("The first local max (corresponding to the independent coupling) is the global max so the second moment method WORKS and we have a typical Markovian independent set of density:")
            print(al)
        else:
            print("The second moment method FAILS as the first local max is not the global max.")

    return res


# finds the largest alpha for which second moment method works
# --> the Markovian independent set of density alpha is typical
def alpha_sm(d):#,show_plot=False):
    al_1rsb=ding_sly_sun(d)
    al_start=al_1rsb-al_1rsb^1.5/4 if d>3 else 0.414189
    #al_start=0.75*al_1rsb
    #thr=1e-8*al_1rsb
    thr=1e-8*al_1rsb^2.5
    al_sm=max_True(lambda al:alpha_sm_check(d,al),al_start,al_1rsb,thr=thr)        
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
