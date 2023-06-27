"""
    ---------------------------------------------------------------------------
    Copyright (C) 2020 Francesco Darugna <fd@geopp.de>  Geo++ GmbH,
                       Jannes B. Wübbena <jw@geopp.de>  Geo++ GmbH.
    A list of all the historical SSRZ Python Demonstrator contributors in
    CREDITS.info
    The first author has received funding from the European Union's
    Horizon 2020 research and innovation programme under the
    Marie Sklodowska-Curie Grant Agreement No 722023.
    ---------------------------------------------------------------------------

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    #--------------------------------------------------------------------------
    
    Description:

    The module uses Millbert's subroutines from the SOFTS software to compute
    the solid Earth tides impact on the user location in the North, East, and
    Up components. The original Fortran subroutines have been translated
    to Python. The output has been validated against the output of the
    SOFTS SW that can be found at:
    https://geodesyworld.github.io/SOFTS/solid.htm.
"""
import numpy as np

# constants to be consistent with the Fortran code
pi = 4.e0 * np.arctan(1.e0)
pi2 = pi + pi
rad = 180.e0 / pi
# grs80
a = 6378137.e0
e2 = 6.69438002290341574957e-03
mjd0 = 0.0
LEApflag = False


def compute_solid_tides(date_time, lat: float, lon: float,
                        zero_tide: bool = False) -> float:
    """ The methods uses Millbert's code to compute solid
        Earth tides. The code was translated to Python
        from the Millbert's source code available at:
        https://geodesyworld.github.io/SOFTS/solid.htm
        Input:
        - date_time : Gregorian calendare date (GPS time)
        - lat       : ellipsoidal latitude [deg]
        - lon       : ellipsoidal longitude [deg]
        - hgt       : ellipsoidal height [m]
        - zero_tide : flag to consider zero tide system for
                      solid Earth tides.
                      Default is a "conventional tide free" system,
                      in conformance with the IERS Conventions.
        Output:
        - tides_neu : solid Earth tides effect in the North, East
                      and Up components.  
    """
    year = date_time.year
    month = date_time.month
    day = date_time.day
    ihr = date_time.hour
    imn = date_time.minute
    sec = date_time.second

    # query section
    if (year < 1901 or year > 2099):
        print('Error: Year must be between 1901-2099')

    if (month < 1 or month > 12):
        print('Error: Month must be between 1-12')

    if (day < 1 or day > 31):
        print('Error: Day must be in 1-31')

    if (lat < -90.e0 or lat > 90.e0):
        print('Error: Latitude must be in -90-90')

    if (lon < -360.e0 or lon > 360.e0):
        print('Error: Longitude must be in -360-360')

    # position of observing point (positive East)
    if (lon < 0.e0):
        lon = lon + 360.e0
    if (lon >= 360.e0):
        lon = lon - 360.e0

    gla0 = lat / rad
    glo0 = lon / rad
    eht0 = 0.e0
    [x0, y0, z0] = geoxyz(gla0, glo0, eht0)
    xsta = np.zeros(3)
    xsta[0] = x0
    xsta[1] = y0
    xsta[2] = z0

    # here comes the sun  (and the moon)  (go, tide#)

    [mjd, fmjd] = civmjd(year, month, day, ihr, imn, sec)
    setje0(year, month, day)
    # Initialize output
    tides_neu = np.zeros(3)
    # -------------------------- Compute tides ----------------------------
    lflag = False  # *** false means flag not raised
    [rsun, lflag] = sunxyz(mjd, fmjd)  # *** mjd/fmjd in UTC
    [rmoon, lflag] = moonxyz(mjd, fmjd)  # *** mjd/fmjd in UTC
    [etide, lflag] = detide(xsta, mjd, fmjd,
                            rsun, rmoon, zero_tide)  # *** mjd/fmjd in UTC
    xt = etide[0]
    yt = etide[1]
    zt = etide[2]

    # determine local geodetic horizon components (topocentric)

    [ut, vt, wt] = rge(gla0, glo0, xt, yt, zt)  # *** tide vector

    [year, month, idy,
     ihr, imn, sec] = mjdciv(mjd, fmjd + 0.001e0 / 86400.e0)

    # Save out variables
    tides_neu[0] = ut
    tides_neu[1] = vt
    tides_neu[2] = wt

    return tides_neu


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def detide(xsta, Mjd, Fmjd, Xsun, Xmon, zero_tide):
    """ 
    #** computation of tidal corrections of station displacements caused
    #**    by lunar and solar gravitational attraction
    #** UTC version

    #** step 1 (here general degree 2 and 3 corrections +
    #**         call st1idiu + call st1isem + call st1l1)
    #**   + step 2 (call step2diu + call step2lon + call step2idiu)
    #** it has been decided that the step 3 un-correction for permanent tide
    #** would *not* be applied in order to avoid jump in the reference frame
    #** (this step 3 must added in order to get the mean tide station position
    #** and to be conformed with the iag resolution.)

    #** inputs
    #**   xsta[i],i=1,2,3   -- geocentric position of the station (ITRF/ECEF)
    #**   xsun[i],i=1,2,3   -- geoc. position of the sun (ECEF)
    #**   xmon[i],i=1,2,3   -- geoc. position of the moon (ECEF)
    #**   mjd,fmjd          -- modified julian day (and fraction) (in UTC time)
    #**   zero_tide         -- flag to consider a zero tide" system for the geopotential 

    #***old calling sequence*****************************************************
    #**   dmjd               -- time in mean julian date (including day fraction)
    #**   fhr=hr+zmin/60.+sec/3600.   -- hr in the day

    #** outputs
    #**   dxtide[i],i=1,2,3  -- displacement vector (ITRF)
    #**   lflag              -- leap second table limit flag, false:flag not raised

    #** author iers 1996 :  v. dehant, s. mathews and j. gipson
    #**    (test between two subroutines)
    #** author iers 2000 :  v. dehant, c. bruyninx and s. mathews
    #**    (test in the bernese program by c. bruyninx)

    #** created:  96/03/23 (see above)
    #** modified from dehanttideinelMJD.f by Dennis Milbert 2006sep10
    #** bug fix regarding fhr (changed calling sequence, too)
    #** modified to reflect table 7.5a and b IERS Conventions 2003
    #** modified to use TT time system to call step 2 functions
    #** sign correction by V.Dehant to match eq.16b, p.81, Conventions
    #** applied by Dennis Milbert 2007may05
    #** UTC version by Dennis Milbert 2018june01
    #
    # Translated to Python by Francesco Darugna 2023 June 27
    """
    global LEApflag
    dxtide = np.zeros(3)
    xcorsta = np.zeros(3)
    # ** nominal second degree and third degree love numbers and shida numbers
    # data
    h20 = 0.6078e0
    l20 = 0.0847e0
    h3 = 0.292e0
    l3 = 0.015e0

    # ** internal support for new calling sequence
    # ** first, convert UTC time into TT time (and, bring leapflag into variable)
    tsecutc = Fmjd * 86400.e0  # *** UTC time (sec of day)
    tsectt = UTC2TTT(tsecutc)  # *** TT  time (sec of day)
    fmjdtt = tsectt/86400.e0  # *** TT  time (fract. day)
    Lflag = LEApflag

    dmjdtt = Mjd + fmjdtt  # *** float MJD in TT
    # ** commented line was live code in dehanttideinelMJD.f
    # ** changed on the suggestion of Dr. Don Kim, UNB -- 09mar21
    # ** Julian date for 2000 January 1 00:00:00.0 UT is  JD 2451544.5
    # ** MJD         for 2000 January 1 00:00:00.0 UT is MJD   51544.0
    # **** t=(dmjdtt-51545.e0)/36525.e0                #*** days to centuries, TT
    t = (dmjdtt - 51544.e0) / 36525.e0  # *** days to centuries, TT
    fhr = (dmjdtt - int(dmjdtt)) * 24.e0  # *** hours in the day, TT

    # ** scalar product of station vector with sun/moon vector
    [scs, rsta, rsun] = SPROD(xsta, Xsun)
    [scm, rsta, rmon] = SPROD(xsta, Xmon)
    scsun = scs/rsta/rsun
    scmon = scm/rsta/rmon

    # ** computation of new h2 and l2
    cosphi = np.sqrt(xsta[0]*xsta[0]+xsta[1]*xsta[1])/rsta
    h2 = h20 - 0.0006e0*(1.e0-3.e0/2.e0*cosphi*cosphi)
    l2 = l20 + 0.0002e0*(1.e0-3.e0/2.e0*cosphi*cosphi)
    # ** p2-term
    p2sun = 3.e0*(h2/2.e0-l2)*scsun*scsun - h2/2.e0
    p2mon = 3.e0*(h2/2.e0-l2)*scmon*scmon - h2/2.e0

    # ** p3-term
    p3sun = 5.e0/2.e0*(h3-3.e0*l3)*scsun**3 + 3.e0/2.e0*(l3-h3)*scsun
    p3mon = 5.e0/2.e0*(h3-3.e0*l3)*scmon**3 + 3.e0/2.e0*(l3-h3)*scmon

    # ** term in direction of sun/moon vector

    x2sun = 3.e0*l2*scsun
    x2mon = 3.e0*l2*scmon
    x3sun = 3.e0*l3/2.e0*(5.e0*scsun*scsun-1.e0)
    x3mon = 3.e0*l3/2.e0*(5.e0*scmon*scmon-1.e0)

    # ** factors for sun/moon
    mass_ratio_sun = 332945.943062e0
    mass_ratio_moon = 0.012300034e0
    re = 6378136.55e0
    fac2sun = mass_ratio_sun*re*(re/rsun)**3
    fac2mon = mass_ratio_moon*re*(re/rmon)**3
    fac3sun = fac2sun*(re/rsun)
    fac3mon = fac2mon*(re/rmon)

    # ** total displacement
    for i in range(0, 3):
        dxtide[i] = (fac2sun*(x2sun*Xsun[i]/rsun+p2sun*xsta[i]/rsta)
                     + fac2mon*(x2mon*Xmon[i]/rmon+p2mon*xsta[i]/rsta)
                     + fac3sun*(x3sun*Xsun[i]/rsun+p3sun*xsta[i]/rsta)
                     + fac3mon*(x3mon*Xmon[i]/rmon+p3mon*xsta[i]/rsta))
    xcorsta = np.zeros(3)
    # ** corrections for the out-of-phase part of love numbers
    # **     (part h_2^(0)i and l_2^(0)i )

    # ** first, for the diurnal band

    xcorsta = ST1IDIU(xsta, Xsun, Xmon, fac2sun, fac2mon)
    dxtide[0] = dxtide[0] + xcorsta[0]
    dxtide[1] = dxtide[1] + xcorsta[1]
    dxtide[2] = dxtide[2] + xcorsta[2]

    # ** second, for the semi-diurnal band

    xcorsta = ST1ISEM(xsta, Xsun, Xmon, fac2sun, fac2mon)
    dxtide[0] = dxtide[0] + xcorsta[0]
    dxtide[1] = dxtide[1] + xcorsta[1]
    dxtide[2] = dxtide[2] + xcorsta[2]

    # ** corrections for the latitude dependence of love numbers (part l^[0] )

    xcorsta = ST1L1(xsta, Xsun, Xmon, fac2sun, fac2mon)
    dxtide[0] = dxtide[0] + xcorsta[0]
    dxtide[1] = dxtide[1] + xcorsta[1]
    dxtide[2] = dxtide[2] + xcorsta[2]

    # ** consider corrections for step 2
    # ** corrections for the diurnal band:

    # **  first, we need to know the date converted in julian centuries

    # **  this is now handled at top of code   (also convert to TT time system)
    # **** t=(dmjd-51545.)/36525.
    # **** fhr=dmjd-int(dmjd)             #*** this is/was a buggy line (day vs. hr)

    # **  second, the diurnal band corrections,
    # **   (in-phase and out-of-phase frequency dependence):

    xcorsta = STEP2DIU(xsta, fhr, t)
    dxtide[0] = dxtide[0] + xcorsta[0]
    dxtide[1] = dxtide[1] + xcorsta[1]
    dxtide[2] = dxtide[2] + xcorsta[2]

    # **  corrections for the long-period band,
    # **   (in-phase and out-of-phase frequency dependence):

    xcorsta = STEP2LON(xsta, fhr, t)
    dxtide[0] = dxtide[0] + xcorsta[0]
    dxtide[1] = dxtide[1] + xcorsta[1]
    dxtide[2] = dxtide[2] + xcorsta[2]

    # ** consider corrections for step 3
    # -----------------------------------------------------------------------
    # The code below is commented to prevent restoring deformation
    # due to permanent tide.  All the code above removes
    # total tidal deformation with conventional Love numbers.
    # The code above realizes a conventional tide free crust (i.e. ITRF).
    # This does NOT conform to Resolution 16 of the 18th General Assembly
    # of the IAG (1983).  This resolution has not been implemented by
    # the space geodesy community in general (c.f. IERS Conventions 2003).
    # -----------------------------------------------------------------------

    # ** uncorrect for the permanent tide  (only if you want mean tide system)
    if (zero_tide):
        PI = 3.141592654
        sinphi = xsta[2]/rsta
        cosphi = np.sqrt(xsta[0]**2+xsta[1]**2)/rsta
        cosla = xsta[0]/cosphi/rsta
        sinla = xsta[1]/cosphi/rsta
        dr = -np.sqrt(5./4./PI)*h2*0.31460*(3./2.*sinphi**2-0.5)
        dn = -np.sqrt(5./4./PI)*l2*0.31460*3.*cosphi*sinphi
        dxtide[0] = dxtide[0]-dr*cosla*cosphi+dn*cosla*sinphi
        dxtide[1] = dxtide[1]-dr*sinla*cosphi+dn*sinla*sinphi
        dxtide[2] = dxtide[2]-dr*sinphi-dn*cosphi

    return dxtide, Lflag

# -----------------------------------------------------------------------


def ST1L1(xsta, Xsun, Xmon, Fac2sun, Fac2mon):
    """ 
    #** this subroutine gives the corrections induced by the latitude dependence
    #** given by l^[0] in mahtews et al (1991)

    #**  input: xsta,xsun,xmon,fac3sun,fac3mon
    #** output: xcorsta
    """
    Xcorsta = np.zeros(3)
    # DATA
    l1d = 0.0012e0
    l1sd = 0.0024e0

    rsta = ENORM8(xsta)
    sinphi = xsta[2]/rsta
    cosphi = np.sqrt(xsta[0]**2+xsta[1]**2)/rsta
    sinla = xsta[1]/cosphi/rsta
    cosla = xsta[0]/cosphi/rsta
    rmon = ENORM8(Xmon)
    rsun = ENORM8(Xsun)

    # ** for the diurnal band

    l1 = l1d
    dnsun = (-l1*sinphi**2*Fac2sun*Xsun[2]
             * (Xsun[0]*cosla+Xsun[1]*sinla)/rsun**2)
    dnmon = (-l1*sinphi**2*Fac2mon*Xmon[2]
             * (Xmon[0]*cosla+Xmon[1]*sinla)/rmon**2)
    desun = (l1*sinphi*(cosphi**2-sinphi**2)*Fac2sun*Xsun[2]
             * (Xsun[0]*sinla-Xsun[1]*cosla)/rsun**2)
    demon = (l1*sinphi*(cosphi**2-sinphi**2)*Fac2mon*Xmon[2]
             * (Xmon[0]*sinla-Xmon[1]*cosla)/rmon**2)
    de = 3.e0*(desun+demon)
    dn = 3.e0*(dnsun+dnmon)
    Xcorsta[0] = -de*sinla - dn*sinphi*cosla
    Xcorsta[1] = de*cosla - dn*sinphi*sinla
    Xcorsta[2] = dn*cosphi

    # ** for the semi-diurnal band

    l1 = l1sd
    costwola = cosla**2 - sinla**2
    sintwola = 2.e0*cosla*sinla
    dnsun = -l1/2.e0*sinphi*cosphi*Fac2sun*((Xsun[0]**2-Xsun[1]**2)
                                            * costwola+2.e0*Xsun[0]*Xsun[1]*sintwola)/rsun**2
    dnmon = -l1/2.e0*sinphi*cosphi*Fac2mon*((Xmon[0]**2-Xmon[1]**2)
                                            * costwola+2.e0*Xmon[0]*Xmon[1]*sintwola)/rmon**2
    desun = -l1/2.e0*sinphi**2*cosphi*Fac2sun*((Xsun[0]**2-Xsun[1]**2)
                                               * sintwola-2.e0*Xsun[0]*Xsun[1]*costwola)/rsun**2
    demon = -l1/2.e0*sinphi**2*cosphi*Fac2mon*((Xmon[0]**2-Xmon[1]**2)
                                               * sintwola-2.e0*Xmon[0]*Xmon[1]*costwola)/rmon**2
    de = 3.e0*(desun+demon)
    dn = 3.e0*(dnsun+dnmon)
    Xcorsta[0] = Xcorsta[0] - de*sinla - dn*sinphi*cosla
    Xcorsta[1] = Xcorsta[1] + de*cosla - dn*sinphi*sinla
    Xcorsta[2] = Xcorsta[2] + dn*cosphi
    return Xcorsta
# -----------------------------------------------------------------------


def STEP2DIU(xsta, Fhr, T):
    """
    #** last change:  vd   17 may 00   1:20 pm
    #** these are the subroutines for the step2 of the tidal corrections.
    #** they are called to account for the frequency dependence
    #** of the love numbers.
    """
    Xcorsta = np.zeros(3)
    deg2rad = 0.017453292519943295769e0

    # ** note, following table is derived from dehanttideinelMJD.f (2000oct30 16:10)
    # ** has minor differences from that of dehanttideinel.f (2000apr17 14:10)
    # ** D.M. edited to strictly follow published table 7.5a (2006aug08 13:46)

    # ** cf. table 7.5a of IERS conventions 2003 (TN.32, pg.82)
    # ** columns are s,h,p,N',ps, dR(ip),dR(op),dT(ip),dT(op)
    # ** units of mm

    # ****-----------------------------------------------------------------------
    # ***** -2., 0., 1., 0., 0.,-0.08,-0.05, 0.01,-0.02,      #*** original entry
    # ****-----------------------------------------------------------------------
    # ****-----------------------------------------------------------------------
    # ***** -1., 0., 0.,-1., 0.,-0.10,-0.05, 0.0 ,-0.02,      #*** original entry
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # ***** -1., 0., 0., 0., 0.,-0.51,-0.26,-0.02,-0.12,      #*** original entry
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # ****-----------------------------------------------------------------------
    # *****  0., 0., 1., 0., 0., 0.06, 0.02, 0.0 , 0.01,      #*** original entry
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # *****  1.,-2., 0., 0., 0.,-1.23,-0.05, 0.06,-0.06,      #*** original entry
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # *****  1., 0., 0., 0., 0.,12.02,-0.45,-0.66, 0.17,      #*** original entry
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # *****  1., 0., 0., 1., 0., 1.73,-0.07,-0.10, 0.02,      #*** original entry
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # ****-----------------------------------------------------------------------
    # *****  1., 1., 0., 0.,-1.,-0.50, 0.0 , 0.03, 0.0,       #*** original entry
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # ****-----------------------------------------------------------------------
    # *****  0., 1., 0., 1.,-1.,-0.01, 0.0 , 0.0 , 0.0,       #*** original entry
    # *** table 7.5a
    # ****-----------------------------------------------------------------------
    # ****-----------------------------------------------------------------------
    # *****  1., 2., 0., 0., 0.,-0.12, 0.01, 0.01, 0.0,       #*** original entry
    # *** v.dehant 2007
    # ****-----------------------------------------------------------------------
    _datdi = [[-3.,  0.,  2.,  0., 0., -0.01, -0.01,  0.0,  0.0],
              [-3.,  2.,  0.,  0., 0., -0.01, -0.01,  0.0,  0.0],
              [-2.,  0.,  1., -1., 0., -0.02, -0.01,  0.0,  0.0],
              [-2.,  0.,  1.,  0., 0., -0.08,  0.00,  0.01,  0.01],
              [-2.,  2., -1.,  0., 0., -0.02, -0.01,  0.0,  0.0],
              [-1.,  0.,  0., -1., 0., -0.10,  0.00,  0.00,  0.00],
              [-1.,  0.,  0.,  0., 0., -0.51,  0.00, -0.02,  0.03],
              [-1.,  2.,  0.,  0., 0.,  0.01,  0.0,  0.0,  0.0],
              [0., -2.,  1.,  0., 0.,  0.01,  0.0,  0.0,  0.0],
              [0.,  0., -1.,  0., 0.,  0.02,  0.01,  0.0,  0.0],
              [0.,  0.,  1.,  0., 0.,  0.06,  0.00,  0.00,  0.00],
              [0.,  0.,  1.,  1., 0.,  0.01,  0.0,  0.0,  0.0],
              [0.,  2., -1.,  0., 0.,  0.01,  0.0,  0.0,  0.0],
              [1., -3.,  0.,  0., 1., -0.06,  0.00,  0.00,  0.00],
              [1., -2.,  0.,  1., 0.,  0.01,  0.0,  0.0,  0.0],
              [1., -2.,  0.,  0., 0., -1.23, -0.07,  0.06,  0.01],
              [1., -1.,  0.,  0., -1.,  0.02,  0.0,  0.0,  0.0],
              [1., -1.,  0.,  0., 1.,  0.04,  0.0,  0.0,  0.0],
              [1., 0.,  0., -1., 0., -0.22,  0.01,  0.01,  0.00],
              [1., 0.,  0.,  0., 0., 12.00, -0.78, -0.67, -0.03],
              [1., 0.,  0.,  1., 0.,  1.73, -0.12, -0.10,  0.00],
              [1., 0.,  0.,  2., 0., -0.04,  0.0,  0.0,  0.0],
              [1., 1.,  0.,  0., -1., -0.50, -0.01,  0.03,  0.00],
              [1., 1.,  0.,  0., 1.,  0.01,  0.0,  0.0,  0.0],
              [1., 1.,  0.,  1., -1., -0.01, 0.0,  0.0,  0.0],
              [1., 2., -2.,  0., 0., -0.01, 0.0,  0.0,  0.0],
              [1., 2.,  0.,  0., 0., -0.11, 0.01,  0.01,  0.00],
              [2., -2.,  1.,  0., 0., -0.01, 0.0,  0.0,  0.0],
              [2., 0., -1.,  0., 0., -0.02, 0.02,  0.0,  0.01],
              [3., 0.,  0.,  0., 0.,  0.0, 0.01, 0.0,  0.01],
              [3., 0.,  0.,  1., 0.,  0.0, 0.01, 0.0,  0.0]]  # *** table 7.5a

    # Make the correct shape of the data
    datdi = np.transpose(np.array(_datdi))
    s = (218.31664563e0 + 481267.88194e0*T - 0.0014663889e0*T*T +
         0.00000185139e0*T**3)
    tau = (Fhr*15.e0 + 280.4606184e0 + 36000.7700536e0*T +
           0.00038793e0*T*T - 0.0000000258e0*T**3 - s)
    pr = (1.396971278*T + 0.000308889*T*T + 0.000000021*T**3 +
          0.000000007*T**4)
    s = s + pr
    h = (280.46645e0 + 36000.7697489e0*T + 0.00030322222e0*T*T +
         0.000000020*T**3 - 0.00000000654*T**4)
    p = (83.35324312e0 + 4069.01363525e0*T - 0.01032172222e0*T*T -
         0.0000124991e0*T**3 + 0.00000005263e0*T**4)
    zns = (234.95544499e0 + 1934.13626197e0*T - 0.00207561111e0*T*T -
           0.00000213944e0*T**3 + 0.00000001650e0*T**4)
    ps = (282.93734098e0 + 1.71945766667e0*T + 0.00045688889e0*T*T -
          0.00000001778e0*T**3 - 0.00000000334e0*T**4)

# ** reduce angles to between 0 and 360

    s = np.mod(s, 360.e0)
    tau = np.mod(tau, 360.e0)
    h = np.mod(h, 360.e0)
    p = np.mod(p, 360.e0)
    zns = np.mod(zns, 360.e0)
    ps = np.mod(ps, 360.e0)

    rsta = np.sqrt(xsta[0]**2+xsta[1]**2+xsta[2]**2)
    sinphi = xsta[2]/rsta
    cosphi = np.sqrt(xsta[0]**2+xsta[1]**2)/rsta

    cosla = xsta[0]/cosphi/rsta
    sinla = xsta[1]/cosphi/rsta
    zla = np.arctan2(xsta[1], xsta[0])
    for i in range(0, 3):
        Xcorsta[i] = 0.e0

    for j in range(0, 31):
        thetaf = ((tau+datdi[0, j]*s+datdi[1, j]*h+datdi[2, j]*p+datdi[3, j]
                   * zns+datdi[4, j]*ps)*deg2rad)
        dr = (datdi[5, j]*2.e0*sinphi*cosphi*np.sin(thetaf+zla) + datdi[6, j]
              * 2.e0*sinphi*cosphi*np.cos(thetaf+zla))
        dn = (datdi[7, j]*(cosphi**2-sinphi**2)*np.sin(thetaf+zla)
              + datdi[8, j]*(cosphi**2-sinphi**2)*np.cos(thetaf+zla))
    # **** following correction by V.Dehant to match eq.16b, p.81, 2003 Conventions
    # ****   de=datdi[7, j]*sinphi*cos(thetaf+zla)+
        de = (datdi[7, j]*sinphi*np.cos(thetaf+zla) - datdi[8, j]
              * sinphi*np.sin(thetaf+zla))
        Xcorsta[0] = (Xcorsta[0] + dr*cosla*cosphi - de*sinla -
                      dn*sinphi*cosla)
        Xcorsta[1] = (Xcorsta[1] + dr*sinla*cosphi + de*cosla -
                      dn*sinphi*sinla)
        Xcorsta[2] = Xcorsta[2] + dr*sinphi + dn*cosphi

    for i in range(0, 3):
        Xcorsta[i] = Xcorsta[i]/1000.e0

    return Xcorsta


# -----------------------------------------------------------------------
def STEP2LON(xsta, Fhr, T):
    """
        #** cf. table 7.5b of IERS conventions 2003 (TN.32, pg.82)
        #** columns are s,h,p,N',ps, dR(ip),dT(ip),dR(op),dT(op)
        #** IERS cols.= s,h,p,N',ps, dR(ip),dR(op),dT(ip),dT(op)
        #** units of mm
    """
    Xcorsta = np.zeros(3)
    deg2rad = 0.017453292519943295769e0

    _datdi = [[0, 0,  0,  1, 0,  0.47,  0.23,  0.16,  0.07],
              [0, 2,  0,  0, 0, -0.20, -0.12, -0.11, -0.05],
              [1, 0, -1,  0, 0, -0.11, -0.08, -0.09, -0.04],
              [2, 0,  0,  0, 0, -0.13, -0.11, -0.15, -0.07],
              [2, 0,  0,  1, 0, -0.05, -0.05, -0.06, -0.03]]
    # Correct data shape
    datdi = np.transpose(np.array(_datdi))
    s = (218.31664563e0 + 481267.88194e0*T - 0.0014663889e0*T*T +
         0.00000185139e0*T**3)
    pr = (1.396971278*T + 0.000308889*T*T + 0.000000021*T**3 +
          0.000000007*T**4)
    s = s + pr
    h = (280.46645e0 + 36000.7697489e0*T + 0.00030322222e0*T*T +
         0.000000020*T**3 - 0.00000000654*T**4)
    p = (83.35324312e0 + 4069.01363525e0*T - 0.01032172222e0*T*T -
         0.0000124991e0*T**3 + 0.00000005263e0*T**4)
    zns = (234.95544499e0 + 1934.13626197e0*T - 0.00207561111e0*T*T -
           0.00000213944e0*T**3 + 0.00000001650e0*T**4)
    ps = (282.93734098e0 + 1.71945766667e0*T + 0.00045688889e0*T*T -
          0.00000001778e0*T**3 - 0.00000000334e0*T**4)
    rsta = np.sqrt(xsta[0]**2+xsta[1]**2+xsta[2]**2)
    sinphi = xsta[2]/rsta
    cosphi = np.sqrt(xsta[0]**2+xsta[1]**2)/rsta
    cosla = xsta[0]/cosphi/rsta
    sinla = xsta[1]/cosphi/rsta

    # ** reduce angles to between 0 and 360

    s = np.mod(s, 360.e0)
    # **** tau=dmod(tau,360.e0)       #*** tau not used here--09jul28
    h = np.mod(h, 360.e0)
    p = np.mod(p, 360.e0)
    zns = np.mod(zns, 360.e0)
    ps = np.mod(ps, 360.e0)

    dr_tot = 0.e0
    dn_tot = 0.e0
    for i in range(0, 3):
        Xcorsta[i] = 0.e0

    # **             1 2 3 4   5   6      7      8      9
    # ** columns are s,h,p,N',ps, dR(ip),dT(ip),dR(op),dT(op)

    for j in range(0, 5):
        thetaf = ((datdi[0, j]*s+datdi[1, j]*h+datdi[2, j]*p+datdi[3, j]
                   * zns+datdi[4, j]*ps)*deg2rad)
        dr = (datdi[5, j]*(3.e0*sinphi**2-1.e0)/2.*np.cos(thetaf)
              + datdi[7, j]*(3.e0*sinphi**2-1.e0)/2.*np.sin(thetaf))
        dn = (datdi[6, j]*(cosphi*sinphi*2.e0)*np.cos(thetaf) + datdi[8, j]
              * (cosphi*sinphi*2.e0)*np.sin(thetaf))
        de = 0.e0
        dr_tot = dr_tot + dr
        dn_tot = dn_tot + dn
        Xcorsta[0] = (Xcorsta[0] + dr*cosla*cosphi - de*sinla -
                      dn*sinphi*cosla)
        Xcorsta[1] = (Xcorsta[1] + dr*sinla*cosphi + de*cosla -
                      dn*sinphi*sinla)
        Xcorsta[2] = Xcorsta[2] + dr*sinphi + dn*cosphi

    for i in range(0, 3):
        Xcorsta[i] = Xcorsta[i]/1000.e0

    return Xcorsta

# -----------------------------------------------------------------------


def ST1IDIU(xsta, Xsun, Xmon, Fac2sun, Fac2mon):
    """ 
    #** this subroutine gives the out-of-phase corrections induced by
    #** mantle inelasticity in the diurnal band

    #**  input: xsta,xsun,xmon,fac2sun,fac2mon
    #** output: xcorsta
    """
    Xcorsta = np.zeros(3)
    dhi = - 0.0025e0
    dli = - 0.0007e0

    rsta = ENORM8(xsta)
    sinphi = xsta[2]/rsta
    cosphi = np.sqrt(xsta[0]**2+xsta[1]**2)/rsta
    cos2phi = cosphi**2 - sinphi**2
    sinla = xsta[1]/cosphi/rsta
    cosla = xsta[0]/cosphi/rsta
    rmon = ENORM8(Xmon)
    rsun = ENORM8(Xsun)
    drsun = (-3.e0*dhi*sinphi*cosphi*Fac2sun*Xsun[2]
             * (Xsun[0]*sinla-Xsun[1]*cosla)/rsun**2)
    drmon = (-3.e0*dhi*sinphi*cosphi*Fac2mon*Xmon[2]
             * (Xmon[0]*sinla-Xmon[1]*cosla)/rmon**2)
    dnsun = (-3.e0*dli*cos2phi*Fac2sun*Xsun[2]
             * (Xsun[0]*sinla-Xsun[1]*cosla)/rsun**2)
    dnmon = (-3.e0*dli*cos2phi*Fac2mon*Xmon[2]
             * (Xmon[0]*sinla-Xmon[1]*cosla)/rmon**2)
    desun = (-3.e0*dli*sinphi*Fac2sun*Xsun[2]
             * (Xsun[0]*cosla+Xsun[1]*sinla)/rsun**2)
    demon = (-3.e0*dli*sinphi*Fac2mon*Xmon[2]
             * (Xmon[0]*cosla+Xmon[1]*sinla)/rmon**2)
    dr = drsun + drmon
    dn = dnsun + dnmon
    de = desun + demon
    Xcorsta[0] = dr*cosla*cosphi - de*sinla - dn*sinphi*cosla
    Xcorsta[1] = dr*sinla*cosphi + de*cosla - dn*sinphi*sinla
    Xcorsta[2] = dr*sinphi + dn*cosphi

    return Xcorsta

# -----------------------------------------------------------------------


def ST1ISEM(xsta, Xsun, Xmon, Fac2sun, Fac2mon):
    """ 
    #** this subroutine gives the out-of-phase corrections induced by
    #** mantle inelasticity in the diurnal band

    #**  input: xsta,xsun,xmon,fac2sun,fac2mon
    #** output: xcorsta
    """
    Xcorsta = np.zeros(3)
    dhi = - 0.0022e0
    dli = - 0.0007e0

    rsta = ENORM8(xsta)
    sinphi = xsta[2]/rsta
    cosphi = np.sqrt(xsta[0]**2+xsta[1]**2)/rsta
    sinla = xsta[1]/cosphi/rsta
    cosla = xsta[0]/cosphi/rsta
    costwola = cosla**2 - sinla**2
    sintwola = 2.e0*cosla*sinla
    rmon = ENORM8(Xmon)
    rsun = ENORM8(Xsun)
    drsun = -3.e0/4.e0*dhi*cosphi**2*Fac2sun*((Xsun[0]**2-Xsun[1]**2)
                                              * sintwola-2.*Xsun[0]*Xsun[1]*costwola)/rsun**2
    drmon = -3.e0/4.e0*dhi*cosphi**2*Fac2mon*((Xmon[0]**2-Xmon[1]**2)
                                              * sintwola-2.*Xmon[0]*Xmon[1]*costwola)/rmon**2
    dnsun = 1.5e0*dli*sinphi*cosphi*Fac2sun*((Xsun[0]**2-Xsun[1]**2)
                                             * sintwola-2.e0*Xsun[0]*Xsun[1]*costwola)/rsun**2
    dnmon = 1.5e0*dli*sinphi*cosphi*Fac2mon*((Xmon[0]**2-Xmon[1]**2)
                                             * sintwola-2.e0*Xmon[0]*Xmon[1]*costwola)/rmon**2
    desun = -3.e0/2.e0*dli*cosphi*Fac2sun*((Xsun[0]**2-Xsun[1]**2)
                                           * costwola+2.*Xsun[0]*Xsun[1]*sintwola)/rsun**2
    demon = -3.e0/2.e0*dli*cosphi*Fac2mon*((Xmon[0]**2-Xmon[1]**2)
                                           * costwola+2.e0*Xmon[0]*Xmon[1]*sintwola)/rmon**2
    dr = drsun + drmon
    dn = dnsun + dnmon
    de = desun + demon
    Xcorsta[0] = dr*cosla*cosphi - de*sinla - dn*sinphi*cosla
    Xcorsta[1] = dr*sinla*cosphi + de*cosla - dn*sinphi*sinla
    Xcorsta[2] = dr*sinphi + dn*cosphi

    return Xcorsta
# -----------------------------------------------------------------------


def SPROD(X, Y):
    """ 
    #**  computation of the scalar-product of two vectors and their norms

    #**  input:   x[i],i=1,2,3  -- components of vector x
    #**           y[i],i=1,2,3  -- components of vector y
    #**  output:  scal          -- scalar product of x and y
    #**           r1,r2         -- lengths of the two vectors x and y
    """
    R1 = np.sqrt(X[0]*X[0]+X[1]*X[1]+X[2]*X[2])
    R2 = np.sqrt(Y[0]*Y[0]+Y[1]*Y[1]+Y[2]*Y[2])
    Scal = X[0]*Y[0] + X[1]*Y[1] + X[2]*Y[2]
    return Scal, R1, R2
# -----------------------------------------------------------------------


def ENORM8(A):
    """
    #** compute euclidian norm of a vector (of length 3)
    """
    return np.sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2])

# -----------------------------------------------------------------------


def ZERO_VEC8(V):
    """
    #** initialize a vector (of length 3) to zero
    """
    V[0] = 0.e0
    V[1] = 0.e0
    V[2] = 0.e0
    return V

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def moonxyz(Mjd, Fmjd):
    """ 
    #** get low-precision, geocentric coordinates for moon (ECEF)
    #** UTC version

    #** input:  mjd/fmjd, is Modified Julian Date (and fractional) in UTC time
    #** output: rm, is geocentric lunar position vector [m] in ECEF
    #**         lflag  -- leap second table limit flag,  false:flag not raised
    #** 1."satellite orbits: models, methods, applications" montenbruck  gill(2000)
    #** section 3.3.2, pg. 72-73
    #** 2."astronomy on the personal computer, 4th ed." montenbruck  pfleger (2005)
    #** section 3.2, pg. 38-39  routine MiniMoon
    #
    # Translated to Python by Francesco Darugna 2023 June 27
    """
    Rm = np.zeros(3)
    # ** use TT for lunar ephemerides

    tsecutc = Fmjd*86400.e0  # *** UTC time (sec of day)
    tsectt = UTC2TTT(tsecutc)  # *** TT  time (sec of day)
    fmjdtt = tsectt/86400.e0  # *** TT  time (fract. day)
    Lflag = LEApflag

    # ** julian centuries since 1.5 january 2000 (J2000)
    # **   (note: also low precision use of mjd --> tjd)

    tjdtt = Mjd + fmjdtt + 2400000.5e0  # *** Julian Date, TT
    t = (tjdtt-2451545.e0)/36525.e0  # *** julian centuries, TT

    # ** el0 -- mean longitude of Moon (deg)
    # ** el  -- mean anomaly of Moon (deg)
    # ** elp -- mean anomaly of Sun  (deg)
    # ** f   -- mean angular distance of Moon from ascending node (deg)
    # ** d   -- difference between mean longitudes of Sun and Moon (deg)

    # ** equations 3.47, p.72

    el0 = 218.31617e0 + 481267.88088e0*t - 1.3972*t
    el = 134.96292e0 + 477198.86753e0*t
    elp = 357.52543e0 + 35999.04944e0*t
    f = 93.27283e0 + 483202.01873e0*t
    d = 297.85027e0 + 445267.11135e0*t

    # ** longitude w.r.t. equinox and ecliptic of year 2000

    selond = (el0
              + 22640.e0/3600.e0*np.sin(el/rad)
              + 769.e0/3600.e0*np.sin((el+el)/rad)
              - 4586.e0/3600.e0*np.sin((el-d-d)/rad)
              + 2370.e0/3600.e0*np.sin((d+d)/rad)
              - 668.e0/3600.e0*np.sin((elp)/rad)
              - 412.e0/3600.e0*np.sin((f+f)/rad)
              - 212.e0/3600.e0*np.sin((el+el-d-d)/rad)
              - 206.e0/3600.e0*np.sin((el+elp-d-d)/rad)
              + 192.e0/3600.e0*np.sin((el+d+d)/rad)
              - 165.e0/3600.e0*np.sin((elp-d-d)/rad)
              + 148.e0/3600.e0*np.sin((el-elp)/rad)
              - 125.e0/3600.e0*np.sin(d/rad)
              - 110.e0/3600.e0*np.sin((el+elp)/rad)
              - 55.e0/3600.e0*np.sin((f+f-d-d)/rad))  # *** eq 3.48, p.72

    # ** latitude w.r.t. equinox and ecliptic of year 2000

    q = (412.e0/3600.e0*np.sin((f+f)/rad)
         + 541.e0/3600.e0*np.sin((elp)/rad))  # *** temporary term

    selatd = (+18520.e0/3600.e0*np.sin((f+selond-el0+q)/rad)
              - 526.e0/3600.e0*np.sin((f-d-d)/rad)
              + 44.e0/3600.e0*np.sin((el+f-d-d)/rad)
              - 31.e0/3600.e0*np.sin((-el+f-d-d)/rad)
              - 25.e0/3600.e0*np.sin((-el-el+f)/rad)
              - 23.e0/3600.e0*np.sin((elp+f-d-d)/rad)
              + 21.e0/3600.e0*np.sin((-el+f)/rad)
              + 11.e0/3600.e0*np.sin((-elp+f-d-d)/rad))  # *** eq 3.49, p.72

    # ** distance from Earth center to Moon (m)

    rse = (385000.e0*1000.e0 - 20905.e0*1000.e0*np.cos((el)/rad)
           - 3699.e0*1000.e0*np.cos((d+d-el)/rad)
           - 2956.e0*1000.e0*np.cos((d+d)/rad)
           - 570.e0*1000.e0*np.cos((el+el)/rad)
           + 246.e0*1000.e0*np.cos((el+el-d-d)/rad)
           - 205.e0*1000.e0*np.cos((elp-d-d)/rad)
           - 171.e0*1000.e0*np.cos((el+d+d)/rad)
           - 152.e0*1000.e0*np.cos((el+elp-d-d)/rad))  # *** eq 3.50, p.72

    # ** convert spherical ecliptic coordinates to equatorial cartesian

    # ** precession of equinox wrt. J2000   (p.71)

    selond = selond + 1.3972e0*t  # *** degrees

    # ** position vector of moon (mean equinox  ecliptic of J2000) (EME2000, ICRF)
    # **                         (plus long. advance due to precession -- eq. above)

    oblir = 23.43929111e0/rad  # *** obliquity of the J2000 ecliptic

    sselat = np.sin(selatd/rad)
    cselat = np.cos(selatd/rad)
    sselon = np.sin(selond/rad)
    cselon = np.cos(selond/rad)

    t1 = rse*cselon*cselat  # *** meters          #*** eq. 3.51, p.72
    t2 = rse*sselon*cselat  # *** meters          #*** eq. 3.51, p.72
    t3 = rse*sselat  # *** meters          #*** eq. 3.51, p.72

    [rm1, rm2, rm3] = ROT1(-oblir, t1, t2, t3)  # *** eq. 3.51, p.72

    # ** convert position vector of moon to ECEF  (ignore polar motion/LOD)
    ghar = GETGHAR(Mjd, Fmjd)  # *** sec 2.3.1,p.33
    [Rm[0],
     Rm[1], Rm[2]] = ROT3(ghar, rm1, rm2, rm3)  # *** eq. 2.89, p.37

    return [Rm, Lflag]


# *******************************************************************************
def GETGHAR(Mjd, Fmjd):
    """ 
    #** convert mjd/fmjd in UTC time to Greenwich hour angle (in radians)

    #** "satellite orbits: models, methods, applications" montenbruck  gill(2000)
    #** section 2.3.1, pg. 33
    #
    # Translated to Python by Francesco Darugna 2023 June 27
    """
    # ** need UTC to get sidereal time ("astronomy on the personal computer", 4th ed)
    # **                               (pg.43, montenbruck  pfleger, springer, 2005)

    tsecutc = Fmjd*86400.e0  # *** UTC time (sec of day)
    fmjdutc = tsecutc/86400.e0  # *** UTC time (fract. day)

    # **** d = MJD - 51544.5e0                               #*** footnote
    d = (Mjd-51544) + (fmjdutc-0.5e0)  # *** days since J2000

    # ** greenwich hour angle for J2000  (12:00:00 on 1 Jan 2000)

    # **** ghad = 100.46061837504e0 + 360.9856473662862e0*d  #*** eq. 2.85 (+digits)
    ghad = 280.46061837504e0 + 360.9856473662862e0*d  # *** corrn.   (+digits)

    # *** normalize to 0-360 and convert to radians

    i = int(ghad/360.e0)
    ghar = (ghad-i*360.e0)/rad
    while (ghar > pi2):
        ghar = ghar - pi2

    while (ghar < 0.e0):
        ghar = ghar + pi2

    return ghar

# *******************************************************************************


def sunxyz(Mjd, Fmjd):
    """ 
    #** get low-precision, geocentric coordinates for sun (ECEF)

    #** input, mjd/fmjd, is Modified Julian Date (and fractional) in UTC time
    #** output, rs, is geocentric solar position vector [m] in ECEF
    #**         lflag  -- leap second table limit flag,  false:flag not raised
    #** 1."satellite orbits: models, methods, applications" montenbruck  gill(2000)
    #** section 3.3.2, pg. 70-71
    #** 2."astronomy on the personal computer, 4th ed." montenbruck  pfleger (2005)
    #** section 3.2, pg. 39  routine MiniSun
    #
    # Translated to Python by Francesco Darugna 2023 June 27
    """

    Rs = np.zeros(3)

    # ** mean elements for year 2000, sun ecliptic orbit wrt. Earth

    obe = 23.43929111e0/rad  # *** obliquity of the J2000 ecliptic
    sobe = np.sin(obe)
    cobe = np.cos(obe)
    opod = 282.9400e0  # *** RAAN + arg.peri.  (deg.)

    # ** use TT for solar ephemerides
    tsecutc = Fmjd*86400.e0  # *** UTC time (sec of day)
    tsectt = UTC2TTT(tsecutc)  # *** TT  time (sec of day)
    fmjdtt = tsectt/86400.e0  # *** TT  time (fract. day)
    Lflag = LEApflag

    # ** julian centuries since 1.5 january 2000 (J2000)
    # **   (note: also low precision use of mjd --> tjd)

    tjdtt = Mjd + fmjdtt + 2400000.5e0  # *** Julian Date, TT
    t = (tjdtt-2451545.e0)/36525.e0  # *** julian centuries, TT
    emdeg = 357.5256e0 + 35999.049e0*t  # *** degrees
    em = emdeg/rad  # *** radians
    em2 = em + em  # *** radians

    # ** series expansions in mean anomaly, em   (eq. 3.43, p.71)

    r = (149.619e0-2.499e0*np.cos(em)-0.021e0*np.cos(em2))*1.e9  # *** m.
    slond = opod + emdeg + (6892.e0*np.sin(em)+72.e0*np.sin(em2))/3600.e0

    # ** precession of equinox wrt. J2000   (p.71)

    slond = slond + 1.3972e0*t  # *** degrees

    # ** position vector of sun (mean equinox  ecliptic of J2000) (EME2000, ICRF)
    # **                        (plus long. advance due to precession -- eq. above)

    slon = slond/rad  # *** radians
    sslon = np.sin(slon)
    cslon = np.cos(slon)

    rs1 = r*cslon  # *** meters             #*** eq. 3.46, p.71
    rs2 = r*sslon*cobe  # *** meters             #*** eq. 3.46, p.71
    rs3 = r*sslon*sobe  # *** meters             #*** eq. 3.46, p.71

    # ** convert position vector of sun to ECEF  (ignore polar motion/LOD)

    ghar = GETGHAR(Mjd, Fmjd)  # *** sec 2.3.1,p.33
    Rs[0], Rs[1], Rs[2] = ROT3(ghar, rs1, rs2, rs3)  # *** eq. 2.89, p.37

    return Rs, Lflag
# *******************************************************************************


def LHSAAZ(U, V, W):
    """ 
    #** determine range,azimuth,vertical angle from local horizon coord.
    """
    s2 = U*U + V*V
    r2 = s2 + W*W

    s = np.sqrt(s2)
    Ra = np.sqrt(r2)

    Az = np.arctan2(V, U)
    Va = np.arctan2(W, s)

    return Ra, Az, Va


# -----------------------------------------------------------------------
def geoxyz(Gla, Glo, Eht):
    """ 
    #** convert geodetic lat, long, ellip ht. to x,y,z
    """

    sla = np.sin(Gla)
    cla = np.cos(Gla)
    w2 = 1.e0 - e2*sla*sla
    w = np.sqrt(w2)
    en = a/w

    X = (en+Eht)*cla*np.cos(Glo)
    Y = (en+Eht)*cla*np.sin(Glo)
    Z = (en*(1.e0-e2)+Eht)*sla

    return X, Y, Z

# -----------------------------------------------------------------------


def rge(Gla, Glo, X, Y, Z):
    """
    #** given a rectangular cartesian system (x,y,z)
    #** compute a geodetic h cartesian sys   (u,v,w)
    #
    # Translated to Python by Francesco Darugna 2023 June 27
    """
    sb = np.sin(Gla)
    cb = np.cos(Gla)
    sl = np.sin(Glo)
    cl = np.cos(Glo)

    U = -sb * cl * X - sb * sl * Y + cb * Z
    V = -sl * X + cl * Y
    W = cb * cl * X + cb * sl * Y + sb * Z
    return [U, V, W]
# -----------------------------------------------------------------------


def ROT1(Theta, X, Y, Z):
    """
    #** rotate coordinate axes about 1 axis by angle of theta radians
    #** x,y,z transformed into u,v,w
    #
    # Translated to Python by Francesco Darugna 2023 June 27
    """
    s = np.sin(Theta)
    c = np.cos(Theta)

    U = X
    V = c * Y + s * Z
    W = c * Z - s * Y
    return [U, V, W]


def ROT3(Theta, X, Y, Z):
    """
    #** rotate coordinate axes about 3 axis by angle of theta radians
    #** x,y,z transformed into u,v,w
    """
    s = np.sin(Theta)
    c = np.cos(Theta)

    U = c*X + s*Y
    V = c*Y - s*X
    W = Z

    return [U, V, W]
# ***********************************************************************
# ** time conversion ****************************************************
# ***********************************************************************


def setje0(Iyr, Imo, Idy):
    """ 
    #** set the integer part of a modified julian date as epoch, mjd0
    #** the modified julian day is derived from civil time as in civmjd()
    #** allows single number expression of time in seconds w.r.t. mjd0
    """
    global mjd0
    if (Iyr < 1900):
        return
    if (Imo <= 2):
        y = Iyr - 1
        m = Imo + 12
    else:
        y = Iyr
        m = Imo

    it1 = int(365.25e0 * y)
    it2 = int(30.6001e0 * (m + 1))
    mjd = int(it1 + it2 + Idy - 679019)

# ** now set the epoch for future time computations

    mjd0 = int(mjd)
    return mjd0


def CIVJTS(Iyr, Imo, Idy, Ihr, Imn, Sec, Tsec):
    """
    #** convert civil date to time in seconds past mjd epoch, mjd0
    #** requires initialization of mjd0 by setje0()

    #** imo in range 1-12, idy in range 1-31
    #** only valid in range mar-1900 thru feb-2100     (leap year protocols)
    #** ref: hofmann-wellenhof, 2nd ed., pg 34-35
    #** adapted from civmjd()
    """
    if (Iyr < 1900):
        return

    if (Imo <= 2):
        y = Iyr - 1
        m = Imo + 12
    else:
        y = Iyr
        m = Imo

    it1 = 365.25e0*y
    it2 = 30.6001e0*(m+1)
    mjd = it1 + it2 + Idy - 679019

    Tsec = (mjd - mjd0) * 86400.e0 + 3600 * Ihr + 60 * Imn + Sec
    return Tsec


def JTSCIV(Tsec):
    """
    #** convert time in seconds past mjd0 epoch into civil date
    #** requires initialization of mjd0 by setje0()

    #** imo in range 1-12, idy in range 1-31
    #** only valid in range mar-1900 thru feb-2100
    #** ref: hofmann-wellenhof, 2nd ed., pg 34-35
    #** adapted from mjdciv()
    """
    mjd = mjd0 + Tsec/86400.e0
# ** the following equation preserves significant digits
    fmjd = np.mod(Tsec, 86400.e0) / 86400.e0

    rjd = mjd + fmjd + 2400000.5e0
    ia = (rjd+0.5e0)
    ib = ia + 1537
    ic = (ib-122.1e0)/365.25e0
    id = 365.25e0*ic
    ie = (ib-id)/30.6001e0

# ** the fractional part of a julian day is (fractional mjd + 0.5)
# ** therefore, fractional part of julian day + 0.5 is (fractional mjd)

    it1 = ie*30.6001e0
    Idy = ib - id - it1 + fmjd
    it2 = ie/14.e0
    Imo = ie - 1 - 12*it2
    it3 = (7+Imo)/10.e0
    Iyr = ic - 4715 - it3

    tmp = fmjd*24.e0
    Ihr = tmp
    tmp = (tmp-Ihr)*60.e0
    Imn = tmp
    Sec = (tmp-Imn)*60.e0

    return Iyr, Imo, Idy, Ihr, Imn, Sec

# ***********************************************************************


def civmjd(Iyr, Imo, Idy, Ihr, Imn, Sec):
    """
    #** convert civil date to modified julian date

    #** imo in range 1-12, idy in range 1-31
    #** only valid in range mar-1900 thru feb-2100     (leap year protocols)
    #** ref: hofmann-wellenhof, 2nd ed., pg 34-35
    #** operation confirmed against table 3.3 values on pg.34
    """
    if (Iyr < 1900):
        return

    if (Imo <= 2):
        y = Iyr - 1
        m = Imo + 12
    else:
        y = Iyr
        m = Imo

    it1 = int(365.25e0*y)
    it2 = int(30.6001e0*(m+1))
    Mjd = it1 + it2 + Idy - 679019

    Fmjd = (3600*Ihr+60*Imn+Sec)/86400.e0
    return [Mjd, Fmjd]


def mjdciv(Mjd, Fmjd):
    """
    #** convert modified julian date to civil date

    #** imo in range 1-12, idy in range 1-31
    #** only valid in range mar-1900 thru feb-2100
    #** ref: hofmann-wellenhof, 2nd ed., pg 34-35
    #** operation confirmed for leap years (incl. year 2000)
    """
# *** Start of declarations inserted by SPAG
    rjd = Mjd + Fmjd + 2400000.5e0
    ia = int(rjd+0.5e0)
    ib = int(ia + 1537)
    ic = int((ib-122.1e0)/365.25e0)
    id = int(365.25e0*ic)
    ie = int((ib-id)/30.6001e0)

# ** the fractional part of a julian day is fractional mjd + 0.5
# ** therefore, fractional part of julian day + 0.5 is fractional mjd

    it1 = int(ie*30.6001e0)
    Idy = int(ib - id - it1 + Fmjd)
    it2 = int(ie/14.e0)
    Imo = int(ie - 1 - 12*it2)
    it3 = int((7+Imo)/10.e0)
    Iyr = int(ic - 4715 - it3)

    tmp = Fmjd*24.e0
    Ihr = int(tmp)
    tmp = (tmp-Ihr)*60.e0
    Imn = int(tmp)
    Sec = (tmp-Imn)*60.e0
    return [Iyr, Imo, Idy, Ihr, Imn, Sec]

# ***********************************************************************
# ** new supplemental time functions ************************************
# ***********************************************************************


def UTC2TTT(Tutc):
    """ 
    #** convert utc (sec) to terrestrial time (sec)
    #*--UTC2TTT1254
    #*** Start of declarations inserted by SPAG
    #*** End of declarations inserted by SPAG
    """
    ttai = UTC2TAI(Tutc)
    return TAI2TT(ttai)

# -----------------------------------------------------------------------


def GPS2TTT(Tgps):
    """
    #** convert gps time (sec) to terrestrial time (sec)
    #*--GPS2TTT1270
    #*** Start of declarations inserted by SPAG
    #*** End of declarations inserted by SPAG
    """
    ttai = GPS2TAI(Tgps)
    return TAI2TT(ttai)

# -----------------------------------------------------------------------


def UTC2TAI(Tutc):
    """ 
    #** convert utc (sec) to tai (sec)
    #*--UTC2TAI1286
    #*** Start of declarations inserted by SPAG
    #*** End of declarations inserted by SPAG
    """
    return Tutc - GETUTCMTAI(Tutc)

# -----------------------------------------------------------------------


def GETUTCMTAI(Tsec):
    """ get utc - tai (s)

    #**** "Julian Date Converter"
    #**** http://aa.usno.navy.mil/data/docs/JulianDate.php

    #*--GETUTCMTAI1304
    #*** Start of declarations inserted by SPAG
    #*** End of declarations inserted by SPAG

    #**** parameter(MJDUPPER=58299)    #*** upper limit, leap second table, 2018jun30

    """
    global LEApflag
    # ** clone for tests (and do any rollover)
    MJDUPPER = 58664  # *** upper limit, leap second table, 2019jun30
    MJDLOWER = 41317  # *** lower limit, leap second table, 1972jan01
    ttsec = Tsec
    mjd0t = mjd0

    while (ttsec >= 86400.e0):
        ttsec = ttsec - 86400.e0
        mjd0t = mjd0t + 1

    while (ttsec < 0.e0):
        ttsec = ttsec + 86400.e0
        mjd0t = mjd0t - 1

# ** test upper table limit         (upper limit set by bulletin C memos)

    if (mjd0t > MJDUPPER):
        LEApflag = True  # *** true means flag *IS* raised
        return -37.e0  # *** return the upper table value

# ** test lower table limit

    if (mjd0t < MJDLOWER):
        LEApflag = True  # *** true means flag *IS* raised
        return -10.e0  # *** return the lower table value

    # **** http://maia.usno.navy.mil/ser7/tai-utc.dat
    # ** 1972 JAN  1 =JD 2441317.5  TAI-UTC=  10.0s
    # ** 1972 JUL  1 =JD 2441499.5  TAI-UTC=  11.0s
    # ** 1973 JAN  1 =JD 2441683.5  TAI-UTC=  12.0s
    # ** 1974 JAN  1 =JD 2442048.5  TAI-UTC=  13.0s
    # ** 1975 JAN  1 =JD 2442413.5  TAI-UTC=  14.0s
    # ** 1976 JAN  1 =JD 2442778.5  TAI-UTC=  15.0s
    # ** 1977 JAN  1 =JD 2443144.5  TAI-UTC=  16.0s
    # ** 1978 JAN  1 =JD 2443509.5  TAI-UTC=  17.0s
    # ** 1979 JAN  1 =JD 2443874.5  TAI-UTC=  18.0s
    # ** 1980 JAN  1 =JD 2444239.5  TAI-UTC=  19.0s
    # ** 1981 JUL  1 =JD 2444786.5  TAI-UTC=  20.0s
    # ** 1982 JUL  1 =JD 2445151.5  TAI-UTC=  21.0s
    # ** 1983 JUL  1 =JD 2445516.5  TAI-UTC=  22.0s
    # ** 1985 JUL  1 =JD 2446247.5  TAI-UTC=  23.0s
    # ** 1988 JAN  1 =JD 2447161.5  TAI-UTC=  24.0s
    # ** 1990 JAN  1 =JD 2447892.5  TAI-UTC=  25.0s
    # ** 1991 JAN  1 =JD 2448257.5  TAI-UTC=  26.0s
    # ** 1992 JUL  1 =JD 2448804.5  TAI-UTC=  27.0s
    # ** 1993 JUL  1 =JD 2449169.5  TAI-UTC=  28.0s
    # ** 1994 JUL  1 =JD 2449534.5  TAI-UTC=  29.0s
    # ** 1996 JAN  1 =JD 2450083.5  TAI-UTC=  30.0s
    # ** 1997 JUL  1 =JD 2450630.5  TAI-UTC=  31.0s
    # ** 1999 JAN  1 =JD 2451179.5  TAI-UTC=  32.0s
    # ** 2006 JAN  1 =JD 2453736.5  TAI-UTC=  33.0s
    # ** 2009 JAN  1 =JD 2454832.5  TAI-UTC=  34.0s
    # ** 2012 JUL  1 =JD 2456109.5  TAI-UTC=  35.0s
    # ** 2015 JUL  1 =JD 2457204.5  TAI-UTC=  36.0s
    # ** 2017 JAN  1 =JD 2457754.5  TAI-UTC=  37.0s
    # **** other leap second references at:
    # **** http://hpiers.obspm.fr/eoppc/bul/bulc/Leap_Second_History.dat
    # **** http://hpiers.obspm.fr/eoppc/bul/bulc/bulletinc.dat

    # ** test against newest leaps first

    if (mjd0t >= 57754):  # *** 2017 JAN 1 = 57754
        tai_utc = 37.e0
    elif (mjd0t >= 57204):  # *** 2015 JUL 1 = 57204
        tai_utc = 36.e0
    elif (mjd0t >= 56109):  # *** 2012 JUL 1 = 56109
        tai_utc = 35.e0
    elif (mjd0t >= 54832):  # *** 2009 JAN 1 = 54832
        tai_utc = 34.e0
    elif (mjd0t >= 53736):  # *** 2006 JAN 1 = 53736
        tai_utc = 33.e0
    elif (mjd0t >= 51179):  # *** 1999 JAN 1 = 51179
        tai_utc = 32.e0
    elif (mjd0t >= 50630):  # *** 1997 JUL 1 = 50630
        tai_utc = 31.e0
    elif (mjd0t >= 50083):  # *** 1996 JAN 1 = 50083
        tai_utc = 30.e0
    elif (mjd0t >= 49534):  # *** 1994 JUL 1 = 49534
        tai_utc = 29.e0
    elif (mjd0t >= 49169):  # *** 1993 JUL 1 = 49169
        tai_utc = 28.e0
    elif (mjd0t >= 48804):  # *** 1992 JUL 1 = 48804
        tai_utc = 27.e0
    elif (mjd0t >= 48257):  # *** 1991 JAN 1 = 48257
        tai_utc = 26.e0
    elif (mjd0t >= 47892):  # *** 1990 JAN 1 = 47892
        tai_utc = 25.e0
    elif (mjd0t >= 47161):  # *** 1988 JAN 1 = 47161
        tai_utc = 24.e0
    elif (mjd0t >= 46247):  # *** 1985 JUL 1 = 46247
        tai_utc = 23.e0
    elif (mjd0t >= 45516):  # *** 1983 JUL 1 = 45516
        tai_utc = 22.e0
    elif (mjd0t >= 45151):  # *** 1982 JUL 1 = 45151
        tai_utc = 21.e0
    elif (mjd0t >= 44786):  # *** 1981 JUL 1 = 44786
        tai_utc = 20.e0
    elif (mjd0t >= 44239):  # *** 1980 JAN 1 = 44239
        tai_utc = 19.e0
    elif (mjd0t >= 43874):  # *** 1979 JAN 1 = 43874
        tai_utc = 18.e0
    elif (mjd0t >= 43509):  # *** 1978 JAN 1 = 43509
        tai_utc = 17.e0
    elif (mjd0t >= 43144):  # *** 1977 JAN 1 = 43144
        tai_utc = 16.e0
    elif (mjd0t >= 42778):  # *** 1976 JAN 1 = 42778
        tai_utc = 15.e0
    elif (mjd0t >= 42413):  # *** 1975 JAN 1 = 42413
        tai_utc = 14.e0
    elif (mjd0t >= 42048):  # *** 1974 JAN 1 = 42048
        tai_utc = 13.e0
    elif (mjd0t >= 41683):  # *** 1973 JAN 1 = 41683
        tai_utc = 12.e0
    elif (mjd0t >= 41499):  # *** 1972 JUL 1 = 41499
        tai_utc = 11.e0
    elif (mjd0t >= 41317):  # *** 1972 JAN 1 = 41317
        tai_utc = 10.e0

# ** should never, ever get here

    else:
        print('FATAL ERROR --')
        print('fell thru tests in gpsleap()')
        tai_utc = np.nan

# ** return utc - tai (in seconds)

    return -tai_utc

# -----------------------------------------------------------------------


def TAI2TT(Ttai):
    """
    #** convert tai (sec) to terrestrial time (sec)
    Ref : http://tycho.usno.navy.mil/systime.html
    """
    return Ttai + 32.184e0

# -----------------------------------------------------------------------


def GPS2TAI(Tgps):

    # ** convert gps time (sec) to tai (sec)
 # *--GPS2TAI1481
    # *** Start of declarations inserted by SPAG

    # *** End of declarations inserted by SPAG

    # **** http://leapsecond.com/java/gpsclock.htm
    # **** http://tycho.usno.navy.mil/leapsec.html
    return Tgps + 19.e0