# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:12:16 2022

@author: cperschke
"""

import numpy as np
import matplotlib.pyplot as plt
import tropo_model as unb3
import tropo_model_geopp_schueler as geopp

class tropo_diff:
    def __init__(self, type, ztd_diff, llh, doy):
        self.type=type
        self.ztd_diff=ztd_diff
        self.llh=llh
        self.doy=doy
      

if __name__ =='__main__':

    llh=np.array([47.49396132866128,3.9152497972469678,312.00])
    doy=271.2505787
    
    mdl_unb3=unb3.get_model_troposphere(llh, doy)
    mdl_geopp=geopp.get_model_troposphere(llh, doy)


    print(mdl_unb3)
    print(mdl_geopp)



"""    
    filename="ztd_model_rtca.txt"
    file=open(filename, "w")

    str=('height[m],' 
         + 'lat_dry[deg],doy_dry,dZTDmodel_dry,'
         + 'lat_wet[deg],doy_wet,dZTDmodel_wet\n')
    file.write(str)
    #print(str)

    tropo_diff_wet=tropo_diff('wet', 0, [0,0,0], 0)
    tropo_diff_dry=tropo_diff('dry', 0, [0,0,0], 0)    


    h=np.array([])
    d=np.array([])
    w=np.array([])
    


    for h_m in range(0, 4000,50):
        
        max_diff_ztd_dry=0
        max_diff_ztd_wet=0
    
        for lat_deg in range(-90,90, 10):

            for doy_offset in np.array([0,(325.25/2.)]):
                llh=(lat_deg, 0, h_m)
            
                if lat_deg>0.0: #northern hemisphere
                    doy = doy_offset + 28.0
                else:
                    doy = doy_offset + 211
        
                mdl_unb3=unb3.get_model_troposphere(llh, doy)
                mdl_geopp=geopp.get_model_troposphere(llh, doy)
        
                diff_dry = np.abs(mdl_unb3[0]-mdl_geopp[0])
                diff_wet = np.abs(mdl_unb3[1]-mdl_geopp[1])
        
                if diff_dry > max_diff_ztd_dry:
                    max_diff_ztd_dry=diff_dry
                    tropo_diff_dry=tropo_diff('dry', diff_dry, llh, doy_offset)
                
                if diff_wet > max_diff_ztd_wet:
                    max_diff_ztd_wet = diff_wet
                    tropo_diff_wet=tropo_diff('wet', diff_wet, llh, doy_offset)

        h=np.append(h,h_m)
        d=np.append(d,tropo_diff_dry.ztd_diff)
        w=np.append(w,tropo_diff_wet.ztd_diff)
        

        str=('{:4.0f}'.format(tropo_diff_dry.llh[2]) + ',' +
             '{:4.1f}'.format(tropo_diff_dry.llh[0]) + ',' +
             '{:4.1f}'.format(tropo_diff_dry.doy) + ',' +
             '{:.8f}'.format(tropo_diff_dry.ztd_diff) + ',' +
             '{:4.1f}'.format(tropo_diff_wet.llh[0]) + ',' +
             '{:4.1f}'.format(tropo_diff_wet.doy) + ',' +
             '{:.8f}'.format(tropo_diff_wet.ztd_diff))
        file.write(str)
        file.write('\n')
        #print(str)

    file.close()          
        
    plt.plot(h,d)
    plt.xlabel('height [m]')
    plt.ylabel('max dZTD_model dry [m]')
    plt.show()

    plt.plot(h,w)
    plt.xlabel('height [m]')
    plt.ylabel('max dZTD_model wet [m]')  
    plt.show()
"""    