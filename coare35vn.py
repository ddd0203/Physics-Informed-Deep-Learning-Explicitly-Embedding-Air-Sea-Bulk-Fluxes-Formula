"""
Functions for COARE model bulk flux calculations.
Translated and vectorized from J Edson/ C Fairall MATLAB scripts.
Execute '%run coare35vn.py' from the iPython command line for test run with
'test_35_data.txt' input data file.
Byron Blomquist, CU/CIRES, NOAA/ESRL/PSD3
Ludovic Bariteau, CU/CIRES, NOAA/ESRL/PSD3
v1: May 2015
v2: July 2020. Fixed some typos and changed syntax for python 3.7 compatibility.
"""

import numpy as np
import meteo
import flux_util
import os

def coare35vn(u, t, rh, ts, P=1015, Rs=150, Rl=370, zu=18, zt=18, zq=18, lat=45,
             zi=600, rain=None, cp=None, sigH=None, jcool=1):
    """
    usage: A = coare35vn(u, t, rh, ts)  -  include other kwargs as desired
    Vectorized version of COARE 3 code (Fairall et al, 2003) with modification
    based on the CLIMODE, MBL and CBLAST experiments (Edson et al., 2013).
    """

    # be sure array inputs are ndarray floats
    u = np.copy(np.asarray(u, dtype=float))
    t = np.copy(np.asarray(t, dtype=float))
    rh = np.copy(np.asarray(rh, dtype=float))
    ts = np.copy(np.asarray(ts, dtype=float))

    # these default to 1 element arrays
    P = np.copy(np.asarray(P, dtype=float))
    Rs = np.copy(np.asarray(Rs, dtype=float))
    Rl = np.copy(np.asarray(Rl, dtype=float))
    zi = np.copy(np.asarray(zi, dtype=float))
    lat = np.copy(np.asarray(lat, dtype=float))

    # check for mandatory input variable consistency
    length = u.size
    if not np.all([t.size == length, rh.size == length, ts.size == length]):
        raise ValueError('coare35vn: u, t, rh, ts arrays of different length')

    # format optional array inputs
    if P.size != length and P.size != 1:
        raise ValueError('coare35vn: P array of different length')
    elif P.size == 1:
        P = P * np.ones(length)

    if Rl.size != length and Rl.size != 1:
        raise ValueError('coare35vn: Rl array of different length')
    elif Rl.size == 1:
        Rl = Rl * np.ones(length)

    if Rs.size != length and Rs.size != 1:
        raise ValueError('coare35vn: Rs array of different length')
    elif Rs.size == 1:
        Rs = Rs * np.ones(length)

    if zi.size != length and zi.size != 1:
        raise ValueError('coare35vn: zi array of different length')
    elif zi.size == 1:
        zi = zi * np.ones(length)

    if lat.size != length and lat.size != 1:
        raise ValueError('coare35vn: lat array of different length')
    elif lat.size == 1:
        lat = lat * np.ones(length)

    if rain is not None:
        rain = np.asarray(rain, dtype=float)
        if rain.size != length:
            raise ValueError('coare35vn: rain array of different length')

    if cp is not None:
        waveage_flag = True
        cp = np.copy(np.asarray(cp, dtype=float))
        if cp.size != length:
            raise ValueError('coare35vn: cp array of different length')
        elif cp.size == 1:
            cp = cp * np.ones(length)
    else:
        waveage_flag = False
        cp = np.nan * np.ones(length)

    if sigH is not None:
        seastate_flag = True
        sigH = np.copy(np.asarray(sigH, dtype=float))
        if sigH.size != length:
            raise ValueError('coare35vn: sigH array of different length')
        elif sigH.size == 1:
            sigH = sigH * np.ones(length)
    else:
        seastate_flag = False
        sigH = np.nan * np.ones(length)

    if waveage_flag and seastate_flag:
        print('Using seastate dependent parameterization')
    if waveage_flag and not seastate_flag:
        print('Using waveage dependent parameterization')

    # check jcool
    if jcool != 0:
        jcool = 1   # all input other than 0 defaults to jcool=1

    # check sensor heights
    test = [type(zu) is int or type(zu) is float]
    test.append(type(zt) is int or type(zt) is float)
    test.append(type(zq) is int or type(zq) is float)
    if not np.all(test):
        raise ValueError('coare35vn: zu, zt, zq, should be constants')
    zu = zu * np.ones(length)
    zt = zt * np.ones(length)
    zq = zq * np.ones(length)

    # input variable u is surface relative wind speed
    us = np.zeros(length)

    # convert rh to specific humidity
    Qs = meteo.qsea(ts, P) / 1000.0  # surface water specific humidity (kg/kg)
    Q, Pv = meteo.qair(t, P, rh)    # specific hum. and partial Pv (mb)
    Q /= 1000.0                   # Q (kg/kg)

    # set constants
    zref = 10.          # ref height, m
    Beta = 1.2
    von  = 0.4          # von Karman const
    fdg  = 1.00         # Turbulent Prandtl number
    tdk  = 273.16
    grav = meteo.grv(lat)

    # air constants
    Rgas = 287.1
    Le   = (2.501 - 0.00237*ts) * 1e6
    cpa  = 1004.67
    cpv  = cpa * (1 + 0.84*Q)
    rhoa = P*100. / (Rgas * (t + tdk) * (1 + 0.61*Q))
    rhodry = (P - Pv)*100. / (Rgas * (t + tdk))
    visa = 1.326e-5 * (1 + 6.542e-3*t + 8.301e-6*t**2 - 4.84e-9*t**3)

    # cool skin constants
    Al   = 2.1e-5 * (ts + 3.2)**0.79
    be   = 0.026
    cpw  = 4000.
    rhow = 1022.
    visw = 1.e-6
    tcw  = 0.6
    bigc = 16. * grav * cpw * (rhow * visw)**3 / (tcw**2 * rhoa**2)
    wetc = 0.622 * Le * Qs / (Rgas * (ts + tdk)**2)

    # net radiation fluxes
    Rns = 0.945 * Rs        # albedo correction
    Rnl = 0.97 * (5.67e-8 * (ts - 0.3*jcool + tdk)**4 - Rl) # initial value

    #####     BEGIN BULK LOOP     #####

    # first guess
    du = u - us
    dt = ts - t - 0.0098*zt
    dq = Qs - Q
    ta = t + tdk
    ug = 0.5
    dter = 0.3
    ut = np.sqrt(du**2 + ug**2)
    u10 = ut * np.log(10/1e-4) / np.log(zu/1e-4)
    usr = 0.035 * u10
    zo10 = 0.011 * usr**2 / grav + 0.11*visa / usr
    Cd10 = (von / np.log(10/zo10))**2
    Ch10 = 0.00115
    Ct10 = Ch10 / np.sqrt(Cd10)
    zot10 = 10 / np.exp(von/Ct10)
    Cd = (von / np.log(zu/zo10))**2
    Ct = von / np.log(zt/zot10)
    CC = von * Ct/Cd
    Ribcu = -zu / zi / 0.004 / Beta**3
    Ribu = -grav * zu/ta * ((dt - dter*jcool) + 0.61*ta*dq) / ut**2
    zetu = CC * Ribu * (1 + 27/9 * Ribu/CC)

    k50 = flux_util.find(zetu > 50)   # stable with thin M-O length relative to zu

    k = flux_util.find(Ribu < 0)
    if Ribcu.size == 1:
        zetu[k] = CC[k] * Ribu[k] / (1 + Ribu[k] / Ribcu)
    else:
        zetu[k] = CC[k] * Ribu[k] / (1 + Ribu[k] / Ribcu[k])

    L10 = zu / zetu
    gf = ut / du
    usr = ut * von / (np.log(zu/zo10) - meteo.psiu_40(zu/L10))
    tsr = -(dt - dter*jcool)*von*fdg / (np.log(zt/zot10) - meteo.psit_26(zt/L10))
    qsr = -(dq - wetc*dter*jcool)*von*fdg / (np.log(zq/zot10) - meteo.psit_26(zq/L10))
    tkt = 0.001 * np.ones(length)

    # The following gives the new formulation for the Charnock variable
    charnC = 0.011 * np.ones(length)
    umax = 19
    a1 = 0.0017
    a2 = -0.0050

    charnC = a1 * u10 + a2
    j = flux_util.find(u10 > umax)
    charnC[j] = a1 * umax + a2

    A_coef = 0.114   # wave-age dependent coefficients
    B_coef = 0.622

    Ad = 0.091  # Sea-state/wave-age dependent coefficients
    Bd = 2.0

    charnW = A_coef * (usr/cp)**B_coef
    zoS = sigH * Ad * (usr/cp)**Bd
    charnS = zoS * grav / usr / usr

    charn = 0.011 * np.ones(length)
    k = flux_util.find(ut > 10)
    charn[k] = 0.011 + (ut[k] - 10) / (18 - 10)*(0.018 - 0.011)
    k = flux_util.find(ut > 18)
    charn[k] = 0.018

    # begin bulk loop
    nits = 10   # number of iterations
    for i in range(nits):
        zet = von*grav*zu / ta*(tsr + 0.61*ta*qsr) / (usr**2)
        if waveage_flag:
            if seastate_flag:
                charn = charnS
            else:
                charn = charnW
        else:
            charn = charnC

        L = zu / zet
        zo = charn*usr**2/grav + 0.11*visa/usr  # surface roughness
        rr = zo*usr/visa

        # thermal roughness lengths
        zoq = np.minimum(1.6e-4, 5.8e-5/rr**0.72)
        zot = zoq
        cdhf = von / (np.log(zu/zo) - meteo.psiu_26(zu/L))
        cqhf = von*fdg / (np.log(zq/zoq) - meteo.psit_26(zq/L))
        cthf = von*fdg / (np.log(zt/zot) - meteo.psit_26(zt/L))
        usr = ut*cdhf
        qsr = -(dq - wetc*dter*jcool)*cqhf
        tsr = -(dt - dter*jcool)*cthf
        tvsr = tsr + 0.61*ta*qsr
        tssr = tsr + 0.51*ta*qsr
        Bf = -grav / ta*usr*tvsr
        ug = 0.2 * np.ones(length)

        k = flux_util.find(Bf > 0)
        if zi.size == 1:
            ug[k] = Beta*(Bf[k]*zi)**0.333
        else:
            ug[k] = Beta*(Bf[k]*zi[k])**0.333

        ut = np.sqrt(du**2 + ug**2)
        gf = ut/du
        hsb = -rhoa*cpa*usr*tsr
        hlb = -rhoa*Le*usr*qsr
        qout = Rnl + hsb + hlb
        dels = Rns * (0.065 + 11*tkt - 6.6e-5/tkt*(1 - np.exp(-tkt/8.0e-4)))
        qcol = qout - dels
        alq = Al*qcol + be*hlb*cpw/Le
        xlamx = 6.0 * np.ones(length)
        tkt = np.minimum(0.01, xlamx*visw/(np.sqrt(rhoa/rhow)*usr))
        k = flux_util.find(alq > 0)
        xlamx[k] = 6/(1 + (bigc[k]*alq[k]/usr[k]**4)**0.75)**0.333
        tkt[k] = xlamx[k]*visw / (np.sqrt(rhoa[k]/rhow)*usr[k])
        dter = qcol*tkt/tcw
        dqer = wetc*dter
        Rnl = 0.97*(5.67e-8*(ts - dter*jcool + tdk)**4 - Rl)   # update dter

        # save first iteration solution for case of zetu>50
        if i == 0:
            usr50 = usr[k50]
            tsr50 = tsr[k50]
            qsr50 = qsr[k50]
            L50 = L[k50]
            zet50 = zet[k50]
            dter50 = dter[k50]
            dqer50 = dqer[k50]
            tkt50 = tkt[k50]

        u10N = usr/von/gf*np.log(10/zo)
        charnC = a1*u10N + a2
        k = flux_util.find(u10N > umax)
        charnC[k] = a1*umax + a2
        charnW = A_coef*(usr/cp)**B_coef
        zoS = sigH*Ad*(usr/cp)**Bd - 0.11*visa/usr
        charnS = zoS*grav/usr/usr
    # end bulk loop

    # insert first iteration solution for case with zetu > 50
    usr[k50] = usr50
    tsr[k50] = tsr50
    qsr[k50] = qsr50
    L[k50] = L50
    zet[k50] = zet50
    dter[k50] = dter50
    dqer[k50] = dqer50
    tkt[k50] = tkt50

    # compute fluxes
    tau = rhoa*usr*usr/gf
    hsb = -rhoa*cpa*usr*tsr
    hlb = -rhoa*Le*usr*qsr
    hbb = -rhoa*cpa*usr*tvsr
    hsbb = -rhoa*cpa*usr*tssr
    wbar = 1.61*hlb/Le/(1+1.61*Q)/rhoa + hsb/rhoa/cpa/ta
    hlwebb = rhoa*wbar*Q*Le
    Evap = 1000*hlb/Le/1000*3600 # mm/hour

    # compute transfer coeffs relative to ut @ meas. ht
    Cd = tau/rhoa/ut/np.maximum(0.1, du)
    Ch = -usr*tsr/ut/(dt - dter*jcool)
    Ce = -usr*qsr/(dq - dqer*jcool)/ut

    # compute 10-m neutral coeff relative to ut
    Cdn_10 = 1000*von**2 / np.log(10/zo)**2
    Chn_10 = 1000*von**2 * fdg/np.log(10/zo) / np.log(10/zot)
    Cen_10 = 1000*von**2 * fdg/np.log(10/zo) / np.log(10/zoq)

    # compute the stability functions
    zrf_u = 10      # User defined reference heights
    zrf_t = 10
    zrf_q = 10
    psi = meteo.psiu_26(zu/L)
    psi10 = meteo.psiu_26(10/L)
    psirf = meteo.psiu_26(zrf_u/L)
    psiT = meteo.psit_26(zt/L)
    psi10T = meteo.psit_26(10/L)
    psirfT = meteo.psit_26(zrf_t/L)
    psirfQ = meteo.psit_26(zrf_q/L)
    gf = ut/du

    # Determine wind speeds relative to ocean surface
    S = ut
    U = du
    S10 = S + usr/von*(np.log(10/zu) - psi10 + psi)
    U10 = S10/gf
    # or U10 = U + usr/von/gf*(np.log(10/zu) - psi10 + psi)
    Urf = U + usr/von/gf*(np.log(zrf_u/zu) - psirf + psi)
    UN = U + psi*usr/von/gf
    U10N = U10 + psi10*usr/von/gf
    UrfN = Urf + psirf*usr/von/gf

    UN2 = usr/von/gf * np.log(zu/zo)
    U10N2 = usr/von/gf * np.log(10/zo)
    UrfN2 = usr/von/gf * np.log(zrf_u/zo)

    # rain heat flux
    if rain is None:
        RF = np.zeros(usr.size)
    else:
        dwat = 2.11e-5*((t + tdk)/tdk)**1.94
        dtmp = (1 + 3.309e-3*t - 1.44e-6*t**2) * 0.02411/(rhoa*cpa)
        dqs_dt = Q*Le / (Rgas*(t + tdk)**2)
        alfac = 1/(1 + 0.622*(dqs_dt*Le*dwat)/(cpa*dtmp))
        RF = rain*alfac*cpw*((ts-t-dter*jcool)+(Qs-Q-dqer*jcool)*Le/cpa)/3600

    lapse = grav/cpa
    SST = ts - dter*jcool

    T = t
    T10 = T + tsr/von*(np.log(10/zt) - psi10T + psiT) + lapse*(zt - 10)
    Trf = T + tsr/von*(np.log(zrf_t/zt) - psirfT + psiT) + lapse*(zt - zrf_t)
    TN = T + psiT*tsr/von
    T10N = T10 + psi10T*tsr/von
    TrfN = Trf + psirfT*tsr/von

    TN2 = SST + tsr/von * np.log(zt/zot) - lapse*zt
    T10N2 = SST + tsr/von * np.log(10/zot) - lapse*10
    TrfN2 = SST + tsr/von * np.log(zrf_t/zot) - lapse*zrf_t

    dqer = wetc*dter*jcool
    SSQ = Qs - dqer
    SSQ = SSQ*1000
    Q = Q*1000
    qsr = qsr*1000
    Q10 = Q + qsr/von*(np.log(10/zq) - psi10T + psiT)
    Qrf = Q + qsr/von*(np.log(zrf_q/zq) - psirfQ + psiT)
    QN = Q + psiT*qsr/von/np.sqrt(gf)
    Q10N = Q10 + psi10T*qsr/von
    QrfN = Qrf + psirfQ*qsr/von

    QN2 = SSQ + qsr/von * np.log(zq/zoq)
    Q10N2 = SSQ + qsr/von * np.log(10/zoq)
    QrfN2 = SSQ + qsr/von * np.log(zrf_q/zoq)
    RHrf = meteo.rhcalc(Trf, P, Qrf/1000)
    RH10 = meteo.rhcalc(T10, P, Q10/1000)

    # basic default output...
    #list1 = [usr, tau, hsb, hlb, hlwebb, tsr, qsr, zot, zoq, Cd, Ch, Ce, L, zet]
    #list2 = [dter, dqer, tkt, RF, Cdn_10, Chn_10, Cen_10, Rns, Rnl]
    list1 = [tau, hsb, hlb, Rns, Rnl, Evap, rain]
    out = tuple(list1)
    A = np.column_stack(out)
    return A

if __name__ == '__main__':
    path = './'
    fil = 'test_35_data.txt'
    cols = 15
    data, varNames = flux_util.load_txt_file(path, fil, cols)

    u = data[:, 0]
    ta = data[:, 2]
    rh = data[:, 4]
    Pa = data[:, 6]
    ts = data[:, 7]
    rs = data[:, 8]
    rl = data[:, 9]
    Lat = data[:, 10]
    ZI = data[:, 11]
    Rain = data[:, 12]

    A = coare35vn(u, ta, rh, ts, P=Pa, Rs=rs, Rl=rl, zu=16, zt=16, zq=16,
                  lat=Lat, zi=ZI, rain=Rain, jcool=1)

    fnameA = os.path.join(path, 'test_35_output_py_082020.txt')
    A_hdr = 'usr\ttau\thsb\thlb\thlwebb\ttsr\tqsr\tzot\tzoq\tCd\t'
    A_hdr += 'Ch\tCe\tL\tzet\tdter\tdqer\ttkt\tRF\tCdn_10\tChn_10\tCen_10'
    # np.savetxt(fnameA,A,fmt='%.18e',delimiter='\t',header=A_hdr)