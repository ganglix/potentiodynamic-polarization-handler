# DOI 10.5281/zenodo.1342163
# GNU General Public License 3.0
# cite as
# Li, Gang, Evitts, Richard, Boulfiza, Moh, & Li, Alice D.S. (2018, August 11). A customized Python module for interactive curve fitting on potentiodynamic scan data (Version v1.0.1). Zenodo. http://doi.org/10.5281/zenodo.1343975


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import streamlit as st

# Helper functions
# Butler-Volmer Equation
def BVeq(E, Eeq, i0, Ba, Bc):
    """
    E: eletrode potential
    Eeq: equilibrium potential
    i0: exchange current density
    Ba: Tafel slope, anodic
    Bc: Tafel slope, cathodic
    NOTE: B-V equation has the same mathematic form when used in single and mixed
    electrode process; however Eeq and i0 may have different terminology
    """
    ia = i0 * 10 ** ((E - Eeq) / Ba)
    ic = -i0 * 10 ** ((E - Eeq) / Bc)
    inet = ia + ic
    return inet


# Empirical Film growth/dissolution Equation, after Z.T. Chang et al 2008,
def Feq(E, Eeq, i0, Ba, Bc, Va, Vc):
    """
    Return net current density due to Film growth/dissolution
    Va,Vc are empirical parameters, larger the value, less the
    contribution of film change to total current in BVFeq.
    """
    iF = (
        (E >= Eeq)
        * i0
        * (np.abs(E - Eeq) / Va * 10 ** (-(E - Eeq) / Ba)) ** 0.5  # Va>0
        + (E < Eeq)
        * -i0
        * (-np.abs(E - Eeq) / Vc * 10 ** (-(E - Eeq) / Bc)) ** 0.5  # Vc<0
    )

    return iF


def BVFeq(E, Eeq, i0, Ba, Bc, Va, Vc):
    """combined rate of main B-V and Film growth/dissolution"""
    return BVeq(E, Eeq, i0, Ba, Bc) + Feq(E, Eeq, i0, Ba, Bc, Va, Vc)


class Info:
    """
    Info object: store and pre-process all raw current potential data and experiment settings
    filename:.xlsx or .csv
    scantype: 'one_step' or 'two_step', default is 'one step'
    """

    def __init__(
        self,
        filename,
        scantype="one_step",
        two_step_drift_offset=True,
        pd_dfIE=None,
        use_pd_df=False,
        area=1,
    ):
        self.filename = filename
        self.scantype = scantype
        self.area = area

        df = None

        if use_pd_df:
            df = pd_dfIE.copy()
            df.columns = ["I", "E"]
            df["i_density"] = df["I"] / self.area
            df["i_density_abs"] = df.i_density.abs()
            df.dropna(inplace=True)
            self.data = df

        else:

            if self.filename.split(".")[-1] == "xlsx":
                df = pd.read_excel(self.filename, skiprows=1)
            elif self.filename.split(".")[-1] == "csv":
                df = pd.read_csv(self.filename, skiprows=1)
            else:
                print("load data error")

            if df.shape[1] == 4:
                self.scantype = "two_step"
            if self.scantype == "one_step":
                df.columns = ["I", "E"]
                Ecorr = df[df.I == df.I.min()].E.values[0]
                df.I[df.E < Ecorr] *= -1  # identify cathodic current
                df["i_density"] = df["I"] / self.area
                df["i_density_abs"] = df.i_density.abs()
                df.dropna(inplace=True)
                self.data = df

            if self.scantype == "two_step":
                df.columns = ["Ic", "Ec", "Ia", "Ea"]
                OCP_c = df[df.Ic == df.Ic.min()].Ec.values[0]
                OCP_a = df[df.Ia == df.Ia.min()].Ea.values[0]
                if two_step_drift_offset:
                    drift = OCP_a - OCP_c
                else:
                    drift = 0.0
                if drift > 0.02:
                    print("Warning: drift=", drift, "V")
                df.Ea = df.Ea - drift
                df.Ec = df.Ec

                df.Ic[df.Ec < OCP_c] *= -1
                df.Ia[df.Ea < OCP_a] *= -1

                df_rev = (
                    df[["Ec", "Ic"]].sort_index(ascending=False).reset_index(drop=True)
                )
                df_IE = pd.DataFrame(
                    {"I": df_rev.Ic.append(df.Ia), "E": df_rev.Ec.append(df.Ea)}
                ).reset_index(drop=True)
                df_IE["i_density"] = df_IE["I"] / self.area
                df_IE["i_density_abs"] = df_IE.i_density.abs()
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
        return self.data.E[
            self.data.i_density_abs == self.data.i_density_abs.min()
        ].values[0]


class Tafit:
    """ data fitting object: Main object to store, process, fit, plot, data and results"""

    def __init__(self, info):
        # input attributes
        self.info = info
        self.area = info.get_area()
        self.data = info.get_data()

        # output attributes can be added dynamically
        self.Ecorr = None
        self.Icorr = None
        self.Ba = None
        self.Bc = None

        self.B = None
        self.Rp = None
        self.Icorr_LPR = None
        # initilize figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        # plot empty data, create line objects
        (self.line_data_sel,) = self.ax.plot(
            [], [], "C0.", markersize=3, label="selceted data"
        )
        (self.line__data_dis1,) = self.ax.plot(
            [], [], "C1.", markersize=3, label="disgarded data"
        )
        (self.line__data_dis2,) = self.ax.plot([], [], "C1.", markersize=3)

        (self.line_guess,) = self.ax.plot(
            [], [], "--g", alpha=0.5, label="Initial Guess"
        )
        # fitted BV equation
        (self.line_bv,) = self.ax.plot([], [], "-r", label="B_V fit")
        (self.line_bvf,) = self.ax.plot([], [], "-r", label="B_V_F fit")
        # tangent line
        (self.line_tan1,) = self.ax.plot([], [], "--r", alpha=0.5)
        (self.line_tan2,) = self.ax.plot([], [], "--r", alpha=0.5)

    def print_out(self):
        return self.result

    def BV_LPR_manual(
        self,
        data_range,
        df_IE,
        fname="",
        taf_init=200,
        R=0,
        auto_zoom=True,
        grid_on=True,
        logx=True,
    ):
        """
        Helper function: mask data, initial guess, fit Butler_Volmer equation, output results
        """
        # mask data
        start = data_range[0]
        stop = data_range[1]

        self.R = R

        I = df_IE.i_density.values
        E = df_IE.E.values - I * self.area * self.R  # Post IR compensation

        I_select = I[start:stop]
        E_select = E[start:stop]

        I_select_abs = np.abs(I_select)
        ind_min = np.where(np.abs(I_select) == np.abs(I_select).min())[0][0]
        OCP = E_select[ind_min]  # open circuit potential

        ############## start of guess parameters from scan#############
        # anodic
        I_select_a = I_select[I_select > 0]
        E_select_a = E_select[I_select > 0]

        Ba_scan, intercept_a = np.polyfit(
            np.log10(I_select_a)[-taf_init:-1], E_select_a[-taf_init:-1], 1
        )  # quick fit slope

        # cathodic
        I_select_c = np.abs(I_select[I_select < 0])
        E_select_c = E_select[I_select < 0]
        Bc_scan, intercept_c = np.polyfit(
            np.log10(I_select_c)[0:taf_init], E_select_c[0:taf_init], 1
        )  # quick fit slope

        Icorr = 10 ** (-(intercept_a - intercept_c) * 1.0 / (Ba_scan - Bc_scan))

        ############## end of guess parameters from scan###############

        # bound based on guess
        bound = (
            [OCP - 0.001, Icorr * 0.01, Ba_scan - 0.20, Bc_scan - 0.20],
            [OCP + 0.001, Icorr * 100.0, Ba_scan + 0.20, Bc_scan + 0.20],
        )

        p_guess = [OCP, Icorr, Ba_scan, Bc_scan]

        popt, pcov = curve_fit(
            BVeq, E_select, I_select, p_guess, bounds=bound
        )  # popt is optimal parameter array
        ############## end of Fitting parameters from scan#############

        # out put
        taf_series = pd.Series(popt, index=["Ecorr", "Icorr", "Ba", "Bc"])

        self.Ecorr = taf_series.Ecorr
        self.Icorr = taf_series.Icorr
        self.Ba = taf_series.Ba
        self.Bc = taf_series.Bc
        self.B = self.Ba * abs(self.Bc) / (2.303 * (self.Ba + abs(self.Bc)))

        # create figure frame
        # plot fit
        _ = np.linspace(E.min(), E.max(), 1000)  # spaced temporary E for plotting

        if logx:
            self.line_data_sel.set_data(np.abs(I_select), E_select)  # selceted data
            self.line__data_dis1.set_data(
                np.abs(I[0:start]), E[0:start]
            )  # disgarded data
            self.line__data_dis2.set_data(
                np.abs(I[stop:-1]), E[stop:-1]
            )  # disgarded data
            self.line_guess.set_data(
                np.abs(BVeq(_, OCP, Icorr, Ba_scan, Bc_scan)), _
            )  # Initial Guess
            self.line_bv.set_data(np.abs(BVeq(_, *popt)), _)  # Fitted from observation
            self.ax.semilogx()

            self.line_tan1.set_data(
                self.Icorr * 10 ** ((_ - self.Ecorr) / self.Ba), _
            )  # tangent line
            self.line_tan2.set_data(self.Icorr * 10 ** ((_ - self.Ecorr) / self.Bc), _)

        else:
            self.line_data_sel.set_data((I_select), E_select)  # selceted data
            self.line__data_dis1.set_data((I[0:start]), E[0:start])  # disgarded data
            self.line__data_dis2.set_data((I[stop:-1]), E[stop:-1])  # disgarded data
            self.line_guess.set_data(
                (BVeq(_, OCP, Icorr, Ba_scan, Bc_scan)), _
            )  # Initial Guess
            self.line_bv.set_data((BVeq(_, *popt)), _)  # Fitted from observation
            self.ax.set_xscale("linear")

            self.line_tan1.set_data([], [])
            self.line_tan2.set_data([], [])

        if self.area == 1:
            plt.xlabel("Current [A]")
        else:
            plt.xlabel("Current density, i $[A/m^2]$")
        plt.ylabel("Potential, E [V]")
        plt.title(str(fname))
        plt.legend(loc="best")

        self.ax.relim()
        self.ax.autoscale_view()

        if auto_zoom:
            if logx:
                plt.xlim(0.1 * I_select_abs.min(), 10 * I_select_abs.max())
            else:
                plt.xlim(1.1 * I_select.min(), 1.1 * I_select.max())
            # plt.ylim(E_select.min()*1.05,E_select.max()*1.05)
        self.ax.grid(grid_on, which="both")
        # # self.fig.show()
        # # self.fig.canvas.draw()
        # plt.show()

        print("range vs Ecorr: {:.3f}~{:.3f} V".format(
            E_select[0] - self.Ecorr, E_select[-1] - self.Ecorr
        ))

        print((
            "Goodness of fit, R2_score:",
            r2_score(I_select, BVeq(E_select, *popt)),
            "Chi squared:",
            np.sum((I_select - BVeq(E_select, *popt)) ** 2 / BVeq(E_select, *popt)),
        ))

        print("\r")
        print("Unit:V, A\n", taf_series)
        print("guess", p_guess)
        print("bound", bound)

        # # LPR
        # df_LPR = self.data[((self.data.E > OCP - 0.02) & (self.data.E < OCP + 0.02))]
        # df_LPR1 = self.data[((self.data.E > OCP + 0.005) & (self.data.E < OCP + 0.02))]
        # df_LPR2 = self.data[((self.data.E > OCP - 0.02) & (self.data.E < OCP - 0.005))]

        # Rp1, _ = np.polyfit(df_LPR1.I, df_LPR1.E, 1)  # quick fit slope
        # Rp2, _ = np.polyfit(df_LPR2.I, df_LPR2.E, 1)  # quick fit slope
        # self.Rp = np.mean([Rp1, Rp2])
        # self.Icorr_LPR = self.B / self.Rp / self.area
        # LPR_series = pd.Series(
        #     [self.B, self.Rp, self.Icorr_LPR], index=["B", "Rp", "Icorr_LPR"]
        # )

        # if False:
        #     fig1, ax1 = plt.subplots(figsize=(8, 6))
        #     ax1.plot(df_LPR.I, df_LPR.E, ".")
        #     ax1.plot(df_LPR1.I, df_LPR1.E, ".")
        #     ax1.plot(df_LPR2.I, df_LPR2.E, ".")
        #     ax1.set_xlabel("I(A)")
        #     ax1.set_ylabel("E(V)")
        #     plt.show()

        # print("Unit:V, Ohm, A")
        # print(LPR_series)
        # self.result = taf_series.append(LPR_series)
        # return plt.gcf()

    def BV_LPR_interact_streamlit(self, anodic_range, cathodic_range):
        """
        interact method of BV_LPR_manual
        anodic/cathodic range vs open circuit, default initial value +-0.15 V
        """
        df_IE = self.data
        # define good range
        start_val = self.info.get_quick_Ecorr() - cathodic_range
        end_val = self.info.get_quick_Ecorr() + anodic_range

        start = (np.abs(self.info.data["E"].values - (start_val))).argmin()
        end = (np.abs(self.info.data["E"].values - (end_val))).argmin()

        fig = self.BV_LPR_manual(
        data_range = st.sidebar.slider("data range",0,len(self.data),(int(start),int(end)),1),
        df_IE = self.data,
        fname="",
        taf_init=st.sidebar.slider("initial guess data points",0,200,20,1),
        R=st.sidebar.slider("R",0,1000,0,1),
        auto_zoom=st.sidebar.checkbox("auto_zoom", True),
        grid_on=st.sidebar.checkbox("grid on",True),
        logx=st.sidebar.checkbox("logx",True),
        )

        # st.pyplot(self.fig)
        # st.write(self.result)
        ####



def main():
    st.title('Curve Fitting for Potentiodynamic Scan Data')

    # uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

    # if uploaded_file is not None:
    #     # Process the file based on its type
    #     if uploaded_file.name.endswith('.xlsx'):
    #         df = pd.read_excel(uploaded_file)
    #     elif uploaded_file.name.endswith('.csv'):
    #         df = pd.read_csv(uploaded_file)

        # Use the data in your analysis functions
        # For example:
        # info = Info(filename=uploaded_file.name, pd_dfIE=df, use_pd_df=True)
        # tafit = Tafit(info)

    # validation
    # Known parameters
    Ecorr = -0.5
    Icorr = 1e-5
    Ba = 0.2
    Bc = -0.3
    pars = (Ecorr,Icorr,Ba,Bc)
    df=pd.DataFrame(columns=['I','E'])
    E = np.linspace(Ecorr-0.2,Ecorr+0.2,2400) # scan range OCP+-0.2V, sample rate 1s at 0.167 mV/s
    df['I'] = BVeq(E=E, Eeq=Ecorr, i0=Icorr, Ba=Ba, Bc=Bc)
    df['E'] = E+2e-3*np.random.randn(2400)
    valid1 = Tafit(Info('',pd_dfIE=df,use_pd_df=True))

    
#       validation
    valid1.BV_LPR_interact_streamlit(0.15,0.15)

    # Call a function that uses these parameters
    # For example:
    # fig = tafit.BV_LPR_interact_streamlit(anodic_range, cathodic_range)
    st.pyplot(valid1.fig)

    # Display other results
    # st.write(tafit.result)

if __name__ == "__main__":
    main()
