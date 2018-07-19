import matplotlib.pylab as plt
import numpy as np
import scipy.special as special
import scipy.constants as constants
import scipy.integrate as integrate

################################################################################
hbarc = 0.1973269664036767; # in GeV nm 
#c = constants.c*100.0 # cm/s 
c = 3e10 # cm/s 
rhoDM = 0.3;  # in GeV/cm^3
#rhoDM = 0.4;  # Chris says this might be pointing toward higher values.

mDM = 8.0; # in GeV 
#mDM = 7.0; # in GeV 
#mDM = 8.8; # in GeV 

mn = 0.938; # mass of nucleon in GeV 
mU = 0.9315; # amu in GeV 

#sigma_n = 1E-40; # in cm^2 
#sigma_n = 5E-40; # in cm^2 
#sigma_n = 0.5e-40; # in cm^2 
Na = 6.022E26; # Avogadro's # in mol/kg 
#Na = constants.N_A*1000.0 # Avogadro's # in mol/kg 
M_PI = np.pi

################################################################################
# double zeroVec[]={0,0,0};
# Atomic masses of targets
################################################################################
AGe = 72.6
ANa = 23
AXe = 131.6
################################################################################

################################################################################
def dot(v1, v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

################################################################################
# this function will normalize first vector and put in second vector
################################################################################
def normalize(v):
    vnorm = np.zeros(3)
    for i,vx in enumerate(v):
        vnorm[i] = vx/np.sqrt(dot(v,v))

    return vnorm


################################################################################
# Conversion
################################################################################
def mT(A):
    return A*mU


def mu_n(m):
    return (m*mn)/(m + mn)


def mu(A, m):
    return (mT(A)*m)/(mT(A)+m)

################################################################################

################################################################################
# Quenching factor
################################################################################
def quench_keVr_to_keVee(x):

    # For Ge, according to Phil Barbeau
    Z = 32
    A = AGe

    #k1 = 0.133*(Z**(2.0/3.))*(A**(-0.5))
    # This k1 below is what Phil gets from his fits.
    k1 = 0.169
    #print k1
    epsilon = 11.5*x*(Z**(-7./3.))
    g_epsilon = 3.*(epsilon**0.15) + 0.7*(epsilon**0.6) + epsilon

    y = (x*k1*g_epsilon)/(1.+(k1*g_epsilon))

    # From Chris Kelso for Ge (CDMS calculations using CoGeNT measurements?)
    #y = 0.2*(x**(1.12))
    return y

def quench_keVee_to_keVr(x):
    # x is Eee
    # Using Phil's code doing a numerical approximation to the inverse.
    y = 0.1683244348504411 + 4.840977386120165*x - 0.6566353866147424*np.power(x,2) + \
       0.17748072066644058*np.power(x,3) - 0.021903569551527707*np.power(x,4)
    # From Chris Kelso for Ge (CDMS calculations using CoGeNT measurements?)
    #y = (x/0.2)**(1.0/1.12)
    return y

def quench_dEr_dEee(Er):
    # dR/dEee
    # This is to account for going from dEr to dEee

    # From Kelso's taking the derivative from Phil's function.
    k1=0.169 # best fit as given by Phil  on Aug. 1, 2013

    dEr_dEee = np.power(1. + 1.2865404292244196*np.power(Er,0.15)*k1 + \
               0.023675915133928603*np.power(Er,0.6)*k1 + \
               0.0035373759945778916*Er*k1,2)/(np.power(Er,0.15)*k1*(1.4795214936080825 + \
               0.037881464214285766*np.power(Er,0.45) + 0.007074751989155783*np.power(Er,0.85) + \
               1.6551862760289537*np.power(Er,0.15)*k1 + 0.06092004403737088*np.power(Er,0.6)*k1 \
               + 0.009101954460784798*Er*k1 + 0.0005605489574289895*np.power(Er,1.05)*k1 + \
               0.0001675012276888449*np.power(Er,1.45)*k1 + \
               0.000012513028927015927*np.power(Er,1.85)*k1))

   
    # Old way from Chris Kelso.
    # Eee
    #dEr_dEee = ((5.0**(1.0/1.12))/1.12)*(Eee**(-0.12/1.12))

    return dEr_dEee

################################################################################
#minimum speed DM must have to produce recoil of energy Er
# Er is assumed to be in keV, and A is atomic mass of target, m is DM mass
# result is in km/s
################################################################################
def vmin(Er, A, m):
    #return c*np.sqrt(Er*mT(A)/2)*1e-8/mu(A,m)
    ret = c*np.sqrt(Er*mT(A)/2)*1e-8/mu(A,m)
    #print ret
    if type(ret)==np.ndarray:
        # Remove nans
        ret[ret!=ret] = 0
    return ret

################################################################################
#Earth orbit parameters
################################################################################
t1 = 0.218*365 #  
e1 = np.array([0.9931,0.1170,-0.01032]) # Define Earth's orbit in Galactic coordinates.
e2= np.array([-0.0670, 0.4927, -0.8676]) # Define Earth's orbit in Galactic coordinates.
vorb = 30
omega = 2*M_PI/365.0


################################################################################
#SHM parameters      #
################################################################################
vo = 220.0 # Chris thinks this might be 235.
vesc = 544.0
# galactic velocity of sun, comes from peculiar velocity and local standard of rest 
vsVec = np.array([10,13+vo,7])
vs = np.sqrt(dot(vsVec,vsVec))
vsVecHat = np.array([vsVec[0]/vs,vsVec[1]/vs,vsVec[2]/vs])
z = vesc/vo
Nesc = special.erf(z)-2*z*np.exp(-z*z)/np.sqrt(M_PI)



################################################################################
# this function will put the total velocity of the earth (observer) in galactic coordinates in the passed vector at the given time
################################################################################
def vObs_t(vObs, t):
    for i in range(0,3):
        # vObs is velocity of the observer...Earth.
        #print type(vObs[i])
        #print vsVec[i]
        #print vorb
        #print omega
        #print t
        #print t1
        #print e1[i]
        #print e2[i]
        #print  (vorb*(np.cos(omega*(t-t1))*e1[i]+np.sin(omega*(t-t1))*e2[i])) 
        #print  (vorb*(np.cos(omega*(t-t1))*e1[i]+np.sin(omega*(t-t1))*e2[i]))
        #print  type((vorb*(np.cos(omega*(t-t1))*e1[i]+np.sin(omega*(t-t1))*e2[i])))
        # Max stream vector will be opposite this value!
        vObs[i] = vsVec[i]+vorb*(np.cos(omega*(t-t1))*e1[i]+np.sin(omega*(t-t1))*e2[i])
        #print vObs


################################################################################
# this function will put only the orbital velocity of the earth in galactic coordinates in the passed vector at the given time
################################################################################
def vE_t(vE, t):
    for i in range(0,3):
        vE[i] = vorb*(np.cos(omega*(t-t1))*e1[i]+np.sin(omega*(t-t1))*e2[i])

################################################################################
# this function is the projection of the observer's velocity onto the stream
# the veoctr that is passes must be the unit vecotr pointing along the stream
# since the earth moves, this function is time dependent
################################################################################
def alpha(vstr, t):
    #vObs = np.zeros(3)
    #vObs_t(vObs,t)
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))
    vObs_t(vObs,t)

    return dot(vObs,vstr)


################################################################################
# this function will give the magnitude of the stream along the given direction
# that will have a cut-off at the given recoil energy for target atomic number A
# at a time t and for dark matter mass of m 
################################################################################
def vstr(vstrHat, A, t, m, Er):
    #vObs = np.zeros(3)
    #vObs_t(vObs,t)
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))
    vObs_t(vObs,t)

    vObsSqrd = dot(vObs,vObs)
    return alpha(vstrHat, t)+np.sqrt(alpha(vstrHat, t)*alpha(vstrHat, t)+vmin(Er,A,m)*vmin(Er,A,m)-vObsSqrd)


################################################################################
# this function gives the speed of the stream in the frame of the earth
################################################################################
def vstrEarth(vstr, t):
    #vObs = np.zeros(3)
    #vObs_t(vObs,t)
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))
    vObs_t(vObs,t)

    return np.sqrt(dot(vstr,vstr)+dot(vObs,vObs)-2*dot(vstr,vObs))


################################################################################
# returns the energy cut-off for a given stream in keV
#t is in days, vstr in km/s, m is DM mass in GeV
################################################################################
def EcutOff(vstr,t,A,m):
    return 2.0e16*mu(A,m)*mu(A,m)*vstrEarth(vstr,t)*vstrEarth(vstr,t)/(c*c*mT(A))

#  return c*sqrt(Er*mT(A)/2)*1E-8/mu(A,m);

################################################################################
# This function finds the time when the stream will have its' max phase
################################################################################
def tc(vstr):
    b1 = 0.0
    b2 = 0.0
    b = 0.0
    t = 0.0
    vSunRel = np.array([vsVec[0]-vstr[0],vsVec[1]-vstr[1],vsVec[2]-vstr[2]])
    vSunRelHat = np.zeros(3)
    vSunRelHat = normalize(vSunRel)
    #print "vSunRelHat: ",vSunRelHat
    b1 = dot(e1,vSunRelHat)
    b2 = dot(e2,vSunRelHat)
    b = np.sqrt(b1*b1+b2*b2)
    t = np.arccos(b1/b)/omega

    if(b2<0):
        t=2*M_PI/omega-t

    return t+t1

################################################################################
# Velocity distribution integral for a stream
# vstrE must be stream speed in frame of earth
# vo is the dispersion of the stream i.e. 
# f(v)~exp(-(v-vstrE)^2/v0^2)
################################################################################
def gStream(vmin, vstrE, v0):
    return (special.erf((vmin+vstrE)/v0)-special.erf((vmin-vstrE)/v0))/(2*vstrE)

################################################################################
# Same as above but if the stream has zero dispersion (i.e. delta function)
################################################################################
def gStreamZeroDispersion(vmin, vstrE):
    if(vmin>vstrE):
        return 0
    else:
        return 1/vstrE


################################################################################
#Thse functions are the velocities integrals for the SHM
################################################################################
def glow(Er, t, A, m):
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))
    vObs_t(vObs,t)

    y = np.sqrt(dot(vObs,vObs))/vo
    x = vmin(Er,A,m)/vo

    #  printf("\nve(t)=%f",vo*y);

    return (special.erf(x+y)-special.erf(x-y)-4*y*np.exp(-z*z)/np.sqrt(M_PI))/(2*Nesc*vo*y);


################################################################################
def ghigh(Er, t, A, m):
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))
    vObs_t(vObs,t)

    y = np.sqrt(dot(vObs,vObs))/vo
    x = vmin(Er,A,m)/vo

    return (special.erf(z)-special.erf(x-y)+2*(x-y-z)*np.exp(-z*z)/np.sqrt(M_PI))/(2*Nesc*vo*y)

################################################################################
def gSHM(Er, t, A, m):
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))

    vObs_t(vObs,t)

    y = np.sqrt(dot(vObs,vObs))/vo
    x = vmin(Er,A,m)/vo

    if type(Er)==np.ndarray:
        #print "HERE!"
        nvals = len(Er)
        ret = np.zeros(nvals)
        #print len(Er)
        test = glow(Er,t,A,m)[x<z-y]
        #print type(test)
        #print test.shape
        #print ret.shape
        #print ret[x<z-y].shape
        ret[x<z-y] = glow(Er,t,A,m)[x<z-y]
        ret[x<z+y] = ghigh(Er,t,A,m)[x<z+y]

        return ret

    elif type(t)==np.ndarray:
        nvals = len(t)
        ret = np.zeros(nvals)
        ret[x<z-y] = glow(Er,t,A,m)[x<z-y]
        ret[x<z+y] = ghigh(Er,t,A,m)[x<z+y]

        return ret

    else:
        if(x<z-y):
            return glow(Er,t,A,m)
        elif(x<y+z):
            return ghigh(Er,t,A,m)
        else:
            return 0


################################################################################
# This works ok for overall rates, but will not work well if doing a modulation analysis
################################################################################
def gDebris(vmin,vflow,t):
    vObs = np.zeros(3)
    if type(t)==np.ndarray:
        vObs = np.zeros((3,len(t)))
    vObs_t(vObs,t)

    vobs = np.sqrt(dot(vObs,vObs))

    if type(vmin)==np.ndarray:
        nvals = len(vmin)
        ret = np.zeros(nvals)
        ret[vmin<np.abs(vflow-vobs)] = (vflow+vobs-np.abs(vflow-vobs))/(2*vflow*vobs)
        ret[vmin<vflow+vobs] = ((vflow+vobs-vmin)/(2*vflow*vobs))[vmin<vflow+vobs]

        ret[ret<0] = 0 # Do we need this?

        return ret

    elif type(t)==np.ndarray:
        nvals = len(t)
        ret = np.zeros(nvals)
        ret[vmin<np.abs(vflow-vobs)] = (vflow+vobs-np.abs(vflow-vobs))/(2*vflow*vobs)
        ret[vmin<vflow+vobs] = (vflow+vobs-vmin)/(2*vflow*vobs)

        ret[ret<0] = 0 # Do we need this?

        return ret

    else:
        if(vmin<np.abs(vflow-vobs)):
            return (vflow+vobs-np.abs(vflow-vobs))/(2*vflow*vobs)
        elif(vmin<vflow+vobs):
            return (vflow+vobs-vmin)/(2*vflow*vobs)
        else:
            return 0

################################################################################
# Form Factor Functions
################################################################################
a=0.523
s=0.9

################################################################################
def cf(A):
    return 1.23*np.power(A,1.0/3.0)-0.6



################################################################################
def q(Er, A):
    return np.sqrt(2*mT(A)*Er*1.0E-6)

################################################################################
def R1(A):
    return np.sqrt(cf(A)*cf(A)+7*M_PI*M_PI*a*a/3-5*s*s)

################################################################################
def Z1(Er,A):
    return q(Er,A)*R1(A)/hbarc

################################################################################
def Zs(Er, A):
    return q(Er,A)*s/hbarc

################################################################################
def j1(x):
    return (np.sin(x)-x*np.cos(x))/(x*x)

################################################################################
def F(Er,A):
    return 3*j1(Z1(Er,A))*np.exp(-Zs(Er,A)*Zs(Er,A)/2)/Z1(Er,A);


################################################################################
#All of the particle physics in the spectrum that is not the velocity distribution integral
#This will give the spectra in counts per kg per keV per day 
################################################################################
def spectraParticle(Er,A,m,sigma_n):
    #return Na*c*c*rhoDM*A*A*sigma_n*F(Er,A)*F(Er,A)*1.0e-11*24*3600/(2*m*mu_n(m)*mu_n(m))
    #ret = Na*c*c*rhoDM*A*A*sigma_n*F(Er,A)*F(Er,A)*1.0e-11*24.*3600./(2*m*mu_n(m)*mu_n(m))
    # Chris says I needs mU here for units.
    ret = mU*Na*c*c*rhoDM*A*A*sigma_n*F(Er,A)*F(Er,A)*1.0e-11*24.*3600./(2*m*mu_n(m)*mu_n(m))
    if type(ret)==np.ndarray:
        # Remove nans
        ret[ret!=ret] = 0
    return ret


################################################################################
# The spectrum for a stream
################################################################################
def dRdErStream(Er, t, A, vstr, v0,m,sigma_n):
    return spectraParticle(Er,A,m,sigma_n)*gStream(vmin(Er,A,m),vstrEarth(vstr,t),v0)


################################################################################
# The SHM spectrum
################################################################################
def dRdErSHM(Er, t, A, m,sigma_n):
    #print "gSHM: ",1000.0*gSHM(Er,t,A,m)
    #print "spectraParticle: ",spectraParticle(Er,A,m,sigma_n)
    return spectraParticle(Er,A,m,sigma_n)*gSHM(Er,t,A,m)


################################################################################
# The debris spectrum from Lisanti's paper
################################################################################
def dRdErDebris(Er,t,A,m,vflow,sigma_n):
    return spectraParticle(Er,A,m,sigma_n)*gDebris(vmin(Er,A,m),vflow,t)


################################################################################
# WIMP signal
################################################################################
def wimp_Er_day(Er,day=1,target_atom=AGe,massDM=10.0,sigma_n=1e-42,quenching_factor=None,efficiency=None,model='shm',vDeb1=340,vSag=220,v0Sag=25,num_wimps=None):

    # Er is the recoil energy.

    if not (model=='shm' or model=='stream' or model=='debris'):
        print("Not correct model for plotting WIMP PDF!")
        print("Model: ",model)
        exit(-1)

    # This is specifically for CoGeNT data taking. Starts at Dec 3. 
    #day = (day+338)%365.0

    #xkeVr = quench_keVee_to_keVr(x)
    #xkeVr = x

    if model=='shm':
        dR = dRdErSHM(Er,day,AGe,mDM,sigma_n)
    elif model=='debris':
        dR = dRdErDebris(Er,day,AGe,mDM,vDeb1,sigma_n)
    elif model=='stream':
        #The Sagitarius stream may intersect the solar system
        #vSag=300
        #v0Sag=100
        vSagHat = np.array([0,0.233,-0.970])
        # IS THIS THE MAX???
        #vSagHat = np.array([0.07247722,0.99486114,-0.07069913])
        vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
        streamVel = vSagVec
        streamVelWidth = v0Sag
        dR = dRdErStream(Er,day,AGe,streamVel,streamVelWidth,mDM,sigma_n)

    # Allow for an efficiency as a function of energy.
    ############# THIS MIGHT NEED TO BE A FUNCTION OF dEee!!!!!!!!
    eff = 1.0
    if efficiency!=None:
        eff = efficiency(Er)

    dR *= eff
    
    # Need this because we've converted from one function to another.
    #### dR/dEee = dR/dEr dEr
    ####dEr_dEee = ((5.0**(1.0/1.12))/1.12)*(x**(-0.12/1.12))
    ####print "dEr_dEee: ",dEr_dEee
    #### Pass in for dEee (ionization energy)
    #dEr_dEee = quench_dEr_dEee(x)
    #dR *= dEr_dEee
    Eee = None
    dR_dEee = None
    if quenching_factor is not None:
        Eee,dEr_dEee = quenching_factor(Er)
        dR_dEee = dR*dEr_dEee
        

    # Normalize
    if num_wimps is not None:
        dR /= num_wimps

    return dR,dR_dEee,Eee


################################################################################
# WIMP signal
################################################################################
def wimp(org_day,x,AGe,mDM,sigma_n,efficiency=None,model='shm',vDeb1=340,vSag=220,v0Sag=25,num_wimps=None):

    # x is the equivalent energy, *not* the recoil energy. Should probably change this. 

    if not (model=='shm' or model=='stream' or model=='debris'):
        print("Not correct model for plotting WIMP PDF!")
        print("Model: ",model)
        exit(-1)

    # This is specifically for CoGeNT data taking. Starts at Dec 3. 
    y = (org_day+338)%365.0
    #y = org_day
    xkeVr = quench_keVee_to_keVr(x)
    #xkeVr = x

    if model=='shm':
        dR = dRdErSHM(xkeVr,y,AGe,mDM,sigma_n)
    elif model=='debris':
        dR = dRdErDebris(xkeVr,y,AGe,mDM,vDeb1,sigma_n)
    elif model=='stream':
        #The Sagitarius stream may intersect the solar system
        #vSag=300
        #v0Sag=100
        vSagHat = np.array([0,0.233,-0.970])
        # IS THIS THE MAX???
        #vSagHat = np.array([0.07247722,0.99486114,-0.07069913])
        vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
        streamVel = vSagVec
        streamVelWidth = v0Sag
        dR = dRdErStream(xkeVr,y,AGe,streamVel,streamVelWidth,mDM,sigma_n)

    ############# THIS MIGHT NEED TO BE A FUNCTION OF dEee!!!!!!!!
    eff = 1.0
    if efficiency!=None:
        eff = efficiency(x)

    dR *= eff
    
    # Need this because we've converted from one function to another.
    # dR/dEee = dR/dEr dEr
    #dEr_dEee = ((5.0**(1.0/1.12))/1.12)*(x**(-0.12/1.12))
    #print "dEr_dEee: ",dEr_dEee
    # Pass in for dEee (ionization energy)
    dEr_dEee = quench_dEr_dEee(x)
    dR *= dEr_dEee

    # Normalize
    if num_wimps is not None:
        dR /= num_wimps

    return dR


################################################################################
# Number of expected WIMPS
################################################################################
def expected_number_of_WIMPs(wimp_model='shm',loE=0.0,hiE=5.0,loDay=0,hiDay=365,target_atom=65,target_mass=1.0,massDM=10,sigma_n=2e-42,efficiency=None):

    dblqtol = 1.0

    nwimps = integrate.dblquad(wimp,loE,hiE,lambda x: loDay,lambda x:hiDay, args=(target_atom,massDM,sigma_n,efficiency,wimp_model),epsabs=dblqtol)[0]*(target_mass)

    return nwimps



################################################################################
# Plot WIMP signal
################################################################################
def plot_wimp_energy(x,target_atom=AGe,massDM=10.0,sigma_n=1e-42,time_range=[1,365],model='shm'):

    if not (model=='shm' or model=='stream' or model=='debris'):
        print("Not correct model for plotting WIMP PDF!")
        print("Model: ",model)
        exit(-1)

    # For debris flow. (340 m/s)
    vDeb1 = 340

    # Assume that the energy passed in is in keVee
    keVr = quench_keVee_to_keVr(x)

    n = 0
    for org_day in range(int(time_range[0]),int(time_range[1]),1):

        day = (org_day+338)%365.0 #- 151

        if model=='shm':
            n += dRdErSHM(keVr,day,target_atom,massDM,sigma_n)
        elif model=='debris':
            n += dRdErDebris(keVr,day,target_atom,massDM,vDeb1,sigma_n)
        elif model=='stream':
            #The Sagitarius stream may intersect the solar system
            vSag=300
            v0Sag=100
            vSagHat = np.array([0,0.233,-0.970])
            vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
            streamVel = vSagVec
            streamVelWidth = v0Sag
            n += dRdErStream(keVr,day,target_atom,streamVel,streamVelWidth,massDM,sigma_n)

    return n

################################################################################

def plot_wimp_day(org_day,target_atom=AGe,massDM=10.0,sigma_n=1e-42,e_range=[0.5,3.2],model='shm',vSag=300,v0Sag=100,vDeb1=340):

    if not (model=='shm' or model=='stream' or model=='debris'):
        print("Not correct model for plotting WIMP PDF!")
        print("Model: ",model)
        exit(-1)

    # For debris flow. (340 m/s)
    #vDeb1 = 340

    n = 0
    day = (org_day+338)%365.0 #- 151
    
    if type(day)==np.ndarray:
        
        n = np.zeros(len(day))
        for i,d in enumerate(day):
            
            x = np.linspace(e_range[0],e_range[1],100)
            keVr = quench_keVee_to_keVr(x)

            if model=='shm':
                n[i] = (dRdErSHM(keVr,d,AGe,mDM,sigma_n)).sum()
            elif model=='debris':
                n[i] = (dRdErDebris(keVr,d,AGe,mDM,vDeb1,sigma_n)).sum()
            elif model=='stream':
                #The Sagitarius stream may intersect the solar system
                #vSag=300
                #v0Sag=100
                vSagHat = np.array([0,0.233,-0.970])
                vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
                streamVel = vSagVec
                streamVelWidth = v0Sag
                n[i] = (dRdErStream(keVr,d,AGe,streamVel,streamVelWidth,mDM,sigma_n)).sum()
    else:
        for x in np.linspace(e_range[0],e_range[1],100):

            keVr = quench_keVee_to_keVr(x)

            if model=='shm':
                n += dRdErSHM(keVr,day,AGe,mDM,sigma_n)
            elif model=='debris':
                n += dRdErDebris(keVr,day,AGe,mDM,vDeb1,sigma_n)
            elif model=='stream':
                #The Sagitarius stream may intersect the solar system
                #vSag=300
                #v0Sag=100
                vSagHat = np.array([0,0.233,-0.970])
                vSagVec = np.array([vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]])
                streamVel = vSagVec
                streamVelWidth = v0Sag
                n += dRdErStream(keVr,day,AGe,streamVel,streamVelWidth,mDM,sigma_n)
                
    return n

