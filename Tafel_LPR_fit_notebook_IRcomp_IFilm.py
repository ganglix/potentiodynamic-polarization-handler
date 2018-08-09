# version: V1.2 plot with notebook interactively, try pd.read_xlsx and csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

print '''import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score'''



def BVeq(E,Eeq,i0,Ba,Bc): # Butler volmer
    ia = i0*10**((E-Eeq)/Ba)
    ic = -i0*10**((E-Eeq)/Bc)
    inet = ia+ic
    return inet

def Feq(E,Eeq,i0,Ba,Bc,Va,Vc): # Z.T. Chang etal 2008,

    iF= (
        (E>=Eeq)*i0*(np.abs(E-Eeq)/Va*10**(-(E-Eeq)/Ba))**0.5 # Va>0
         +(E<Eeq)*-i0*(-np.abs(E-Eeq)/Vc*10**(-(E-Eeq)/Bc))**0.5 # Vc<0
         )

    return iF

def BVFeq(E,Eeq,i0,Ba,Bc,Va,Vc):
    return BVeq(E,Eeq,i0,Ba,Bc)+Feq(E,Eeq,i0,Ba,Bc,Va,Vc)

class Info:
    """
    filename:.xlsx or .csv
    scantype: 'one_step' or 'two_step'
    """
    def __init__(self,filename,scantype='one_step',area=1,two_step_drift_offset=True, pd_dfIE = None,use_pd_df=False):
        self.filename = filename
        self.scantype = scantype
        self.area = area

        if use_pd_df:
            df = pd_dfIE
            df.columns = ['I','E']
            df['i_density']=df['I']/self.area
            df['i_density_abs'] = df.i_density.abs()
            df.dropna(inplace=True)
            self.data = df

        else:
            
            if self.filename.split('.')[-1]=='xlsx':
                df = pd.read_excel(self.filename,skiprows=1)
            elif self.filename.split('.')[-1]=='csv':
                df = pd.read_csv(self.filename,skiprows=1)
            else: 
                print 'load data error'
                
            if df.shape[1]==4:
                self.scantype ='two_step'
            if self.scantype == 'one_step':
                df.columns = ['I','E']
                Ecorr = df[df.I==df.I.min()].E.values[0]
                df.I[df.E<Ecorr]*=-1 # identify cathodic current
                df['i_density']=df['I']/self.area
                df['i_density_abs'] = df.i_density.abs()
                df.dropna(inplace=True)
                self.data = df

            if self.scantype == 'two_step':
                df.columns = ['Ic','Ec','Ia','Ea']
                OCP_c = df[df.Ic==df.Ic.min()].Ec.values[0]
                OCP_a = df[df.Ia==df.Ia.min()].Ea.values[0]
                if two_step_drift_offset:
                    drift = OCP_a - OCP_c
                else:
                    drift = 0.
                if drift>0.02: print 'Warning: dift=',drift,'V'
                df.Ea = df.Ea-drift
                df.Ec = df.Ec

                df.Ic[df.Ec<OCP_c]*=-1
                df.Ia[df.Ea<OCP_a]*=-1

                df_rev = df[['Ec','Ic']].sort_index(ascending=False).reset_index(drop=True)
                df_IE = pd.DataFrame({'I':df_rev.Ic.append(df.Ia),'E':df_rev.Ec.append(df.Ea)}).reset_index(drop=True)
                df_IE['i_density']=df_IE['I']/self.area
                df_IE['i_density_abs'] = df_IE.i_density.abs()
                df_IE.dropna(inplace=True)
                self.data = df_IE

    def get_filename(self):
        return self.filename
    def get_scantype(self):
        return self.scantype
    def get_area(self):
        return self.area
    def get_data(self):
        return self.data
    def get_quick_Ecorr(self):
        return self.data.E[self.data.i_density_abs==self.data.i_density_abs.min()].values[0]


class Tafit:
    """ Tafit: init obj
        BV_fit: auto fit butler-volmer
        BV_fit_interact: select curve portion and fit"""
    def __init__(self, info):
        #input attributes
        self.info = info
        self.area = info.get_area()
        self.data = info.get_data()

        #output attributes can be added dynamically
        self.Ecorr = None
        self.icorr = None
        self.Ba = None
        self.Bc = None

        self.B = None
        self.Rp = None
        self.icorr_LPR = None
        # init figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        # plot data
        self.line_data_sel, = self.ax.plot([],[],'C0.',markersize=3,label='selceted data')
        self.line__data_dis1, = self.ax.plot([],[],'C1.',markersize=3,label='disgarded data')
        self.line__data_dis2, = self.ax.plot([],[],'C1.',markersize=3)

        self.line_guess, = self.ax.plot([],[],'--g',alpha=0.5,label='Initial Guess')
        # fitted BV equation
        self.line_bv, = self.ax.plot([],[],'-r',label='B_V fit')
        self.line_bvf, = self.ax.plot([],[],'-r',label='B_V_F fit')
    # tangent line
        self.line_tan1, = self.ax.plot([],[],'--r',alpha=0.5)
        self.line_tan2, = self.ax.plot([],[],'--r',alpha=0.5)

    def print_out(self):
        return self.result

    def BV_LPR_manual(self,data_range,df_IE,fname = '',taf_init=200,R=0, auto_zoom=True,grid_on=True,logx=True):
        # mask data
        start=data_range[0]
        stop = data_range[1]

        self.R = R

        I = df_IE.i_density.values
        E = df_IE.E.values - I*self.area*self.R  # IR compensation


        I_select = I[start:stop]
        E_select = E[start:stop]

        # name change in favour of legacy
        iext = I_select
        iext_abs = np.abs(iext)
        E_rebar = E_select
        ind_min = np.where(np.abs(I_select)==np.abs(I_select).min())[0][0]
        OCP = E_select[ind_min]  # open circuit potential

        ############## start of guess parameters from scan#############
        # anodic
        iext_a = iext[iext>0]
        E_rebar_a = E_rebar[iext>0]

        Ba_scan,intercept_a = np.polyfit(np.log10(iext_a)[-taf_init:-1], E_rebar_a[-taf_init:-1], 1) # quick fit slope

        # cathodic
        iext_c = np.abs(iext[iext<0])
        E_rebar_c = E_rebar[iext<0]
        Bc_scan,intercept_c = np.polyfit(np.log10(iext_c)[0:taf_init], E_rebar_c[0:taf_init], 1) # quick fit slope

        icorr = 10**(-(intercept_a - intercept_c)*1.0/(Ba_scan-Bc_scan))

        ############## end of guess parameters from scan###############

        # bound based on guess
        bound = ([OCP-0.001,icorr*0.01,Ba_scan-0.20,Bc_scan-0.20],
                 [OCP+0.001,icorr*100.0,Ba_scan+0.20,Bc_scan+0.20])

        p_guess = [OCP,icorr,Ba_scan,Bc_scan]

        popt, pcov = curve_fit(BVeq,E_rebar, iext, p_guess, bounds=bound) # popt is optimal parameter array
        ############## end of Fitting parameters from scan#############

        # out put
        taf_series = pd.Series(popt ,index=['Ecorr','icorr','Ba','Bc'])

        self.Ecorr = taf_series.Ecorr
        self.icorr = taf_series.icorr
        self.Ba = taf_series.Ba
        self.Bc = taf_series.Bc
        self.B = self.Ba*abs(self.Bc)/(2.303*(self.Ba+abs(self.Bc)))

        # create figure frame
        #fig,ax = plt.subplots(figsize=(8,6))
        # plot fit
        _ = np.linspace(E.min(),E.max(),1000) # spaced temporary E for plotting

        if logx:
            self.line_data_sel.set_data(np.abs(I_select),E_select)       #selceted data
            self.line__data_dis1.set_data(np.abs(I[0:start]),E[0:start]) #disgarded data
            self.line__data_dis2.set_data(np.abs(I[stop:-1]),E[stop:-1]) #disgarded data
            self.line_guess.set_data(np.abs(BVeq(_,OCP,icorr,Ba_scan,Bc_scan)),_) #Initial Guess
            self.line_bv.set_data(np.abs(BVeq(_,*popt)),_)#Fitted from observation
            self.ax.semilogx()

            self.line_tan1.set_data(self.icorr*10**((_-self.Ecorr)/self.Ba),_) # tangent line
            self.line_tan2.set_data(self.icorr*10**((_-self.Ecorr)/self.Bc),_)

        else:
            self.line_data_sel.set_data((I_select),E_select)       #selceted data
            self.line__data_dis1.set_data((I[0:start]),E[0:start]) #disgarded data
            self.line__data_dis2.set_data((I[stop:-1]),E[stop:-1]) #disgarded data
            self.line_guess.set_data((BVeq(_,OCP,icorr,Ba_scan,Bc_scan)),_) #Initial Guess
            self.line_bv.set_data((BVeq(_,*popt)),_)#Fitted from observation
            self.ax.set_xscale('linear')

            self.line_tan1.set_data([],[])
            self.line_tan2.set_data([],[])


        plt.xlabel('Averaged logi [$A/m^2$]')
        plt.ylabel('E_rebar [V]')
        plt.title(str(fname))
        plt.legend(loc='best')

        self.ax.relim()
        self.ax.autoscale_view()

        if auto_zoom:
            if logx:
                plt.xlim(0.1*iext_abs.min(),10*iext_abs.max())
            else:
                plt.xlim(1.1*iext.min(),1.1*iext.max())
            #plt.ylim(E_select.min()*1.05,E_select.max()*1.05)
        self.ax.grid(grid_on,which='both')
        #self.fig.show()
        #self.fig.canvas.draw()
        plt.show()

        print 'range vs Ecorr: {:.3f}~{:.3f} V'.format(E_select[0]-self.Ecorr, E_select[-1]-self.Ecorr)

        print ('Goodness of fit, R2_score:', r2_score(I_select,BVeq(E_select,*popt)),
               'Chi squared:',np.sum((I_select-BVeq(E_select,*popt))**2/BVeq(E_select,*popt))
               )

        print '\r'
        print 'Unit:V, A/m^2\n',taf_series
        print 'guess', p_guess
        print 'bound', bound


        #LPR
        df_LPR = self.data[((self.data.E>OCP-0.02)& (self.data.E<OCP+0.02))]
        df_LPR1 = self.data[((self.data.E>OCP+0.005)& (self.data.E<OCP+0.02))]
        df_LPR2 = self.data[((self.data.E>OCP-0.02)& (self.data.E<OCP-0.005))]

        Rp1,_ = np.polyfit(df_LPR1.I,df_LPR1.E, 1) # quick fit slope
        Rp2,_ = np.polyfit(df_LPR2.I,df_LPR2.E, 1) # quick fit slope
        self.Rp = np.mean([Rp1, Rp2])
        self.icorr_LPR = self.B/self.Rp/self.area
        LPR_series = pd.Series([self.B, self.Rp, self.icorr_LPR] ,index=['B','Rp','icorr_LPR'])

        if False:
            fig1,ax1 = plt.subplots(figsize=(8,6))
            ax1.plot(df_LPR.I,df_LPR.E,'.')
            ax1.plot(df_LPR1.I,df_LPR1.E,'.')
            ax1.plot(df_LPR2.I,df_LPR2.E,'.')
            ax1.set_xlabel('I(A)')
            ax1.set_ylabel('E(V)')
            plt.show()

        print 'Unit:V, Ohm, A/m^2'
        print LPR_series
        self.result = taf_series.append(LPR_series)
        return self


    def BV_LPR_interact(self, anodic_range=0.15,cathodic_range=0.15):
        import ipywidgets as widgets
        layout = widgets.Layout(width = '500px',height='40px')
        df_IE = self.data
        # define good range
        start_val = self.info.get_quick_Ecorr()-cathodic_range
        end_val = self.info.get_quick_Ecorr()+anodic_range

        start = (np.abs(self.info.data['E'].values - (start_val))).argmin()
        end = (np.abs(self.info.data['E'].values - (end_val))).argmin()
        #remove unused line
        try: 
            self.line_bvf.remove()
        except:
            pass
        ####
        widgets.interact(self.BV_LPR_manual,
                         data_range=widgets.IntRangeSlider(min=0, max=len(df_IE), step=1,value=[start,end],continuous_update=False,layout=layout),
                         df_IE = widgets.fixed(df_IE),
                         fname=widgets.fixed(self.info.filename),
                         taf_init=widgets.IntSlider(min=0, max=len(df_IE), step=1.0,value=200, continuous_update=False,layout=layout),
                         R=widgets.FloatSlider(min=-20000, max=20000, step=1.0,value=0, continuous_update=False,layout=layout),
                         auto_zoom = widgets.Checkbox(value=True,description='Auto_zoom',disabled=False),
                         grid_on = widgets.Checkbox(value=True,description='Grid_on',disabled=False),
                         logx = widgets.Checkbox(value=True,description='logx',disabled=False)
                         )

    def BVF_LPR_manual(self,data_range,df_IE,fname = '',taf_init=200,R=0, auto_zoom=True,grid_on=True,logx=True):
        # mask data
        start=data_range[0]
        stop = data_range[1]

        self.R = R

        I = df_IE.i_density.values
        E = df_IE.E.values + I*self.area*self.R  # IR compensation


        I_select = I[start:stop]
        E_select = E[start:stop]

        # name change in favour of legacy
        iext = I_select
        iext_abs = np.abs(iext)
        E_rebar = E_select
        ind_min = np.where(np.abs(I_select)==np.abs(I_select).min())[0][0]
        OCP = E_select[ind_min]  # open circuit potential

        ############## start of guess parameters from scan#############
        # anodic
        iext_a = iext[iext>0]
        E_rebar_a = E_rebar[iext>0]

        Ba_scan,intercept_a = np.polyfit(np.log10(iext_a)[-taf_init:-1], E_rebar_a[-taf_init:-1], 1) # quick fit slope

        # cathodic
        iext_c = np.abs(iext[iext<0])
        E_rebar_c = E_rebar[iext<0]
        Bc_scan,intercept_c = np.polyfit(np.log10(iext_c)[0:taf_init], E_rebar_c[0:taf_init], 1) # quick fit slope

        icorr = 10**(-(intercept_a - intercept_c)*1.0/(Ba_scan-Bc_scan))

        Va_guess = 1000.0 # Va>0
        Vc_guess = -1000.0 # Vc<0

        ############## end of guess parameters from scan###############

        # bound based on guess

        bound = ([OCP-0.001,icorr*0.01,Ba_scan-0.20,Bc_scan-0.20,0.0001,Vc_guess],
                 [OCP+0.001,icorr*100.0,Ba_scan+0.20,Bc_scan+0.20,Va_guess,-0.0001])

        p_guess = [OCP,icorr,Ba_scan,Bc_scan,Va_guess,Vc_guess]

        popt, pcov = curve_fit(BVFeq,E_rebar, iext, p_guess, bounds=bound) # popt is optimal parameter array
        ############## end of Fitting parameters from scan#############

        # out put
        taf_series = pd.Series(popt ,index=['Ecorr','icorr','Ba','Bc','Va','Vc'])

        self.Ecorr = taf_series.Ecorr
        self.icorr = taf_series.icorr
        self.Ba = taf_series.Ba
        self.Bc = taf_series.Bc
        self.Va = taf_series.Va
        self.Vc = taf_series.Vc

        self.B = self.Ba*abs(self.Bc)/(2.303*(self.Ba+abs(self.Bc)))

        # create figure frame
        #fig,ax = plt.subplots(figsize=(8,6))
        # plot fit
        _ = np.linspace(E.min(),E.max(),1000) # spaced temporary E for plotting


        if logx:
            self.line_data_sel.set_data(np.abs(I_select),E_select)       #selceted data
            self.line__data_dis1.set_data(np.abs(I[0:start]),E[0:start]) #disgarded data
            self.line__data_dis2.set_data(np.abs(I[stop:-1]),E[stop:-1]) #disgarded data
            self.line_guess.set_data(np.abs(BVFeq(_,*p_guess)),_) #Initial Guess
            #self.line_bv.set_data(np.abs(BVeq(_,*popt[0:4])),_)#BV Fitted from observation
            self.line_bvf.set_data(np.abs(BVFeq(_,*popt)),_)#BVF Fitted from observation
            self.ax.semilogx()

#             self.line_tan1.set_data(self.icorr*10**((_-self.Ecorr)/self.Ba),_)
#             self.line_tan2.set_data(self.icorr*10**((_-self.Ecorr)/self.Bc),_)

        else:
            self.line_data_sel.set_data((I_select),E_select)       #selceted data
            self.line__data_dis1.set_data((I[0:start]),E[0:start]) #disgarded data
            self.line__data_dis2.set_data((I[stop:-1]),E[stop:-1]) #disgarded data
            self.line_guess.set_data((BVFeq(_,*p_guess)),_) #Initial Guess
            #self.line_bv.set_data((BVeq(_,*popt[0:4])),_)#BV Fitted from observation
            self.line_bvf.set_data((BVFeq(_,*popt)),_)#BVF Fitted from observation            self.ax.set_xscale('linear')
            self.ax.set_xscale('linear')

#             self.line_tan1.set_data([],[])
#             self.line_tan2.set_data([],[])


        plt.xlabel('Averaged logi [$A/m^2$]')
        plt.ylabel('E_rebar [V]')
        plt.title(str(fname))
        plt.legend(loc='best')

        self.ax.relim()
        self.ax.autoscale_view()

        if auto_zoom:
            if logx:
                plt.xlim(0.1*iext_abs.min(),10*iext_abs.max())
            else:
                plt.xlim(1.1*iext.min(),1.1*iext.max())
            #plt.ylim(E_select.min()*1.05,E_select.max()*1.05)
        self.ax.grid(grid_on,which='both')
        #self.fig.show()
        #self.fig.canvas.draw()
        plt.show()

        print 'range vs Ecorr: {:.3f}~{:.3f} V'.format(E_select[0]-self.Ecorr, E_select[-1]-self.Ecorr)

        print ('Goodness of fit, R2_score:', r2_score(I_select,BVFeq(E_select,*popt)),
               'Chi squared:',np.sum((I_select-BVFeq(E_select,*popt))**2/BVFeq(E_select,*popt))
               )

        print '\r'
        print 'Unit:V, A/m^2\n',taf_series
        print 'guess', p_guess
        print 'bound', bound

        #LPR
        df_LPR = self.data[((self.data.E>OCP-0.02)& (self.data.E<OCP+0.02))]
        df_LPR1 = self.data[((self.data.E>OCP+0.005)& (self.data.E<OCP+0.02))]
        df_LPR2 = self.data[((self.data.E>OCP-0.02)& (self.data.E<OCP-0.005))]

        Rp1,_ = np.polyfit(df_LPR1.I,df_LPR1.E, 1) # quick fit slope
        Rp2,_ = np.polyfit(df_LPR2.I,df_LPR2.E, 1) # quick fit slope
        self.Rp = np.mean([Rp1, Rp2])
        self.icorr_LPR = self.B/self.Rp/self.area
        LPR_series = pd.Series([self.B, self.Rp, self.icorr_LPR] ,index=['B','Rp','icorr_LPR'])

        if False:
            fig1,ax1 = plt.subplots(figsize=(8,6))
            ax1.plot(df_LPR.I,df_LPR.E,'.')
            ax1.plot(df_LPR1.I,df_LPR1.E,'.')
            ax1.plot(df_LPR2.I,df_LPR2.E,'.')
            ax1.set_xlabel('I(A)')
            ax1.set_ylabel('E(V)')
            plt.show()

        print 'Unit:V, Ohm, A/m^2'
        print LPR_series
        self.result = taf_series.append(LPR_series)
        return self


    def BVF_LPR_interact(self, anodic_range=0.15,cathodic_range=0.15):
        import ipywidgets as widgets
        layout = widgets.Layout(width = '500px',height='40px')
        df_IE = self.data
        # remove line
        self.line_bv.remove()
        self.line_tan1.remove()
        self.line_tan2.remove()
        ####
        # define good range
        start_val = self.info.get_quick_Ecorr()-cathodic_range
        end_val = self.info.get_quick_Ecorr()+anodic_range

        start = (np.abs(self.info.data['E'].values - (start_val))).argmin()
        end = (np.abs(self.info.data['E'].values - (end_val))).argmin()

        widgets.interact(self.BVF_LPR_manual,
                         data_range=widgets.IntRangeSlider(min=0, max=len(df_IE), step=1,value=[start,end],continuous_update=False,layout=layout),
                         df_IE = widgets.fixed(df_IE),
                         fname=widgets.fixed(self.info.filename),
                         taf_init=widgets.IntSlider(min=0, max=len(df_IE), step=1.0,value=200, continuous_update=False,layout=layout),
                         R=widgets.FloatSlider(min=-20000, max=20000, step=1.0,value=0, continuous_update=False,layout=layout),
                         auto_zoom = widgets.Checkbox(value=True,description='Auto_zoom',disabled=False),
                         grid_on = widgets.Checkbox(value=True,description='Grid_on',disabled=False),
                         logx = widgets.Checkbox(value=True,description='logx',disabled=False)
                         )

    def remove_outlier(self,roughness=10):
        try:
            out_lier_idx=(self.data.index[0:-1][np.abs(np.diff(self.data['i_density']))
                   > roughness *np.abs(np.mean(np.diff(self.data['i_density'])))])
            self.data.iloc[self.data.iloc[out_lier_idx].index]=np.nan
            self.data.dropna(axis='index',inplace=True)
        except:
            print 'Outlier removal Failed!'
        return self

    def remove_outlier_manual(self,data_range):
        out_lier_idx= range(data_range[0],data_range[1])
        self.data.iloc[self.data.iloc[out_lier_idx].index]=np.nan
        self.data.dropna(axis='index',inplace=True)
        return self

    def remove_outlier_interact(self,):
        import ipywidgets as widgets
        layout = widgets.Layout(width = '500px',height='40px')
        widgets.interact(self.remove_outlier_manual,
                         data_range=widgets.IntRangeSlider(min=0, max=0, step=1,value=[0,len(df_IE)],continuous_update=False,layout=layout)
                         )

    def plot_BV_F_components(self, logx=True):
        if logx:
            plt.plot(BVFeq(self.data.E,*self.result.values[0:6]).abs(),
                     self.data.E,
                     label='BVF')

            plt.plot(BVeq(self.data.E,*self.result.values[0:4]).abs(),
                     self.data.E,
                     label='BV_component')

            plt.plot(BVFeq(self.data.E,*self.result.values[0:6]).abs()
                     -BVeq(self.data.E,*self.result.values[0:4]).abs(),
                     self.data.E,
                     label='F_component')

            plt.gca().semilogx()

        else:
            plt.plot(BVFeq(self.data.E,*self.result.values[0:6]),
                     self.data.E,
                     label='BVF')

            plt.plot(BVeq(self.data.E,*self.result.values[0:4]),
                     self.data.E,
                     label='BV_component')

            plt.plot(BVFeq(self.data.E,*self.result.values[0:6])
                     -BVeq(self.data.E,*self.result.values[0:4]),
                     self.data.E,
                     label='F_component')

        plt.legend()

def plot_compare(fobjs,offset_drift=True,logx=True,absolute_value = True, inplace = False):
    '''Info object: fobjs = [Info('filename.xlsx')...]'''
    if offset_drift:
        Ecorrs = np.array([this_obj.get_quick_Ecorr() for this_obj in fobjs])
        drifts = Ecorrs-Ecorrs[0]
    else:
        drifts = np.zeros(len(fobjs))

    fig, this_ax = plt.subplots()
    for this_obj,this_drift in zip(fobjs, drifts):

        #df_modifed = this_obj.get_data().
        this_obj.get_data().E=this_obj.get_data().E - this_drift
        if absolute_value:
            this_obj.get_data().plot(x='i_density_abs', y='E',ax=this_ax, style='.',markersize=1,
                                     logx=logx, label=this_obj.get_filename().split('/')[-1].split('.xlsx')[0])
        else:
            this_obj.get_data().plot(x='i_density', y='E',ax=this_ax, style='.',markersize=1,
                                     logx=logx, label=this_obj.get_filename().split('/')[-1].split('.xlsx')[0])

    plt.grid()
    plt.show()
    if inplace == False:
        # reverse the drifts
        for this_obj,this_drift in zip(fobjs, drifts):
            this_obj.get_data().E=this_obj.get_data().E + this_drift
