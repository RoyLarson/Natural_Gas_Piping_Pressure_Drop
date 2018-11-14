import numpy as np
import pandas as pd
from collections import namedtuple
from findiff import FinDiff

MW_AIR = 28.9647
R_FT_LB_PER_LBMOL_DEGR = 1545.348963
FT_LB_PER_BTU = 778.169
FT_LB_PER_PSI_FT3 = 144
R_BTU_PER_LBMOL_DEGR = R_FT_LB_PER_LBMOL_DEGR  / FT_LB_PER_BTU
R_PSI_FT3_PER_LBMOL_DEGR = R_BTU_PER_LBMOL_DEGR / FT_LB_PER_PSI_FT3
F_TO_R = 459.67
LB_PER_KG = 0.45359237
G_c_lbm_ft_per_lbf_s2 = 32.174
G_ft_sec2 = 32.174
IN_PER_FT = 12

class SimpleNG:
    """
    Calculates the equation of state for a natural gas that is solely specified by the Molecular Weight
    References
    Isobaric Specific Heat Capacity of Natural Gas as a Function of Specific Gravity, Pressure, Temperature
    Lateef A Kareem, Tajudeen M Iwalewa, James E Omeke
    Journal of Natural Gas Science and Engineering April 17, 2014
    """
    MW_air = MW_AIR #lb/lbmol
    R_ft_lb = R_FT_LB_PER_LBMOL_DEGR#ft-lb/(lbmol * deg R)
    R_btu =  R_BTU_PER_LBMOL_DEGR # Btu/(lbmol * deg R)
    R_psi_ft3 = R_PSI_FT3_PER_LBMOL_DEGR # psi*ft^3/(lbmol (deg R))
    ideal_specific_heat_coefficients = np.array([[-10.9602, 25.9033],
                                                 [2.1517e-1, -6.8687e-2],
                                                 [-1.337e-4, 8.6387e-5],
                                                 [3.1474e-8, 2.8396e-8]])
    # using np.nan as a filler for the coefficient array so I can copy the equation
    # in Isobaric...  journal page number 80  exactly            
    residual_specific_heat_coefficients = np.array([np.nan, 4.80828, -4.01563, -.0700681, .0567, 2.36642, -3.82421, 7.71784])
    F_to_R = F_TO_R

    def __init__(self, specific_gravity=None, molecular_weight=None):
        """
        Default Values for Tc and Pc come from promax estimation of the injection gas properties
        injection gas critical temp from promax=-17.48194377594 F
        injection gas critical pressure from promax =1447.328746236 psia
        """
        if specific_gravity:
            self.SG = specific_gravity
        elif molecular_weight:
            self.MW = molecular_weight
        else:
            raise Exception('Need Molecular Weight or Specific Gravity')

        self.Pc = self._calc_Pc()
        self.Tc = self._calc_Tc()

    @property
    def MW(self):
        return self._MW

    @MW.setter
    def MW(self, molecular_weight):
        self._MW = molecular_weight
        self.Pc = self._calc_Pc()
        self.Tc = self._calc_Tc()

    @property
    def SG(self):
        return self._MW / self.MW_air
        
    @SG.setter
    def SG(self, specific_gravity):
        self.MW = specific_gravity * self.MW_air
        self.Pc = self._calc_Pc()
        self.Tc = self._calc_Tc()

    def _calc_Pc(self):
        """
        Psuedo Critical Pressure estimate from specific gravity
        From Sutton 1985
        Returns in psia
        """
        return 756.8-131.07 * self.SG - 3.6 * self.SG ** 2
        
    def _calc_Tc(self):
        """
        Psuedo Critical Temperature estimate from specific gravity
        From Sutton 1985
        Returns in deg R
        """
        return 169.2+349.5 * self.SG - 3.6 * self.SG ** 2

   
    def _ideal_specific_heat_const_p(self, temperature_R):
        """
        From:
        Isobaric Specific Heat Capacity of Natural Gas as a Function of Specific Gravity, Pressure, and Temperature
        returns C_p in ft_lbf/(lbm R)
        """
        sg = self.SG
        sg_mat = np.array([sg, 1]).T
        T = temperature_R-self.F_to_R
        T_mat = np.array([1,T, T**2, T**3])

        Cp_ideal = self.ideal_specific_heat_coefficients @ sg_mat @ T_mat
        return Cp_ideal*self.btu_to_ft_lb/self.MW

    def _specific_heat_residual_const_p(self, temperture_R, pressure_psia):
        """
        From:
        Isobaric Specific Heat Capacity of Natural Gas as a Function of Specific Gravity, Pressure, and Temperature
        returns delta_C_p in ft_lbf/(lbm R)
        Equation Form
        R*(term_1/term_2 - term_3/term_4)
        """ 
        a = self.residual_specific_heat_coefficients
        t = 1 / self.reduced_temp(temperature_R)
        Pr = self.reduced_pressure(pressure_psia)
        Prt = Pr*t
        R_specific = self.R_ft_lb/self.MW
        term_1 = (1 + (a[1] * np.exp(a[2] * (1 - t)** 2) * Prt)** 2)
        term_2 = (a[7] + a[6] * Prt + a[5] * Prt ** 2 + a[4] * Prt ** 3)
        term_3 = (a[3] * Prt ** 6) * (a[1] * np.exp(1 - t ** 2) * Prt)** 2
        term_4 = (a[7] + a[6] * Prt + a[5] * Prt ** 2 + a[4] * Prt ** 3)** 3
        return R_specific * (term_1 / term_2 - term_3 / term_4)
        
    def molar_specific_volume(self, temperature_R, pressure_psia):
        """
        acf per mole
        """
        Z = self.compressibility(temperature_R, pressure_psia)
        return Z*temperature_R*self.R_psi_ft3/pressure_psia

    def mass_specific_volume(self, temperature_R, pressure_psia):
        return self.molar_specific_volume(temperature_R, pressure_psia)/self.MW

    def specific_heat_const_p(self, temperature_R, pressure_psia):
        return self._specific_heat_residual_const_p(temperature_R, pressure_psia) + \
                self._ideal_specific_heat_const_p(temperature_R, pressure_psia)

    def Cp(self, temperature_R, pressure_psia):
        """
        Constant Pressure specific Heat
        """
        return self.specific_heat_const_p(temperature_R, pressure_psia)

    def specific_heat_const_v(self, temperature_R, pressure_psia):
        """
        returning the ideal gas version of specific heat at constant volume
        Cv = Cp+T*((dP/dT)_V)**2/(dP/dV)_T
        ft_lbf/(lb R)
        """
        P = pressure_psia
        T = temperature_R
        dT = T * .001
        dP = P * .001
        Cp = self.Cp(T, P)
        R = self.R_ft_lb/self.MW
        Z = self.compressibility(T, P)
        d_dT = FinDiff(0, dT, acc=2)
        dZ_dT = d_dT(self.compressibility)
        d_dP = FinDiff(1, dP, acc=2)
        dZ_dP = d_dP(self.compressibility)
        dV_dP_T = (R*T/P)*(dZ_dP + Z/P)
        dV_dT_P = (R / P) * (T * dZ_dT + Z)
        
        return Cp - T*(dV_dT_P**2)/dV_dP_T

    def ratio_of_specific_heats(self, temperature_R, pressure_psia):
        """
        ratio of specific heats
        """
        Cp = self.specific_heat_const_p(temperature_R, pressure_psia)
        Cv = self.specific_heat_const_v(temperature_R, pressure_psia)
        return Cp / Cv
        
    def reduced_pressure(self, pressure_psia):
        """
        Default value of p_crit is in psia
        from promax estimate of injection gas
        """
        return pressure_psia/self.Pc

    def reduced_temp(self, temperature_R):
        """
        Default value of t_crit is in deg. Rankine
        from promax estimate of injection gas
        """
        return temperature_R / self.Tc
        
    def compressibility(self, temperature_R, pressure_psia):
        """
        From:
        Efficient Estimation of Natural Gas Compressibility Factor Using a Rigorous Method
        Journal of Natural Gas Engineering V 16 - August 2014
        This is the Shell Oil Company Correlation which had the least deviation of the data for the points
        that uses only available data in this EOS
        """    
        Tr = self.reduced_temp(temperature_R)
        Pr = self.reduced_pressure(pressure_psia)
        A = -.101- .36 * Tr + 1.3868 * np.sqrt(Tr - .919)
        B = .021+ .04275 / (Tr - .65)
        D = .122 * np.exp(-11.3 * (Tr - 1))
        E = .622- .224 * Tr
        F =  (.0657 / (Tr - .85)) - .037
        G = .32 * np.exp(-19.53 * (Tr - 1))
        C = Pr*(E+F*Pr+G*Pr**4)
        return A+B*Pr+(1-A)*np.exp(-C)-D*(Pr/10)**4

    def density(self, temperature_R, pressure_psia):
        """
        R = Universal Gas Constant in psi*ft^3/(lbmol * degR) = 10.73159(2)
        returns density in lb/ft**3
        """
        T = temperature_R
        P = pressure_psia
        MW = self.MW
        R = self.R_psi_ft3 # Universal Gas Constant in psi*ft^3/(lbmol * degR)
        z = self.compressibility(temperature_R, pressure_psia)
        return P / (z * (R / MW) * T)

    def viscosity(self, temperature_R, pressure_psia):
        """
        From New Correlations to Predict Natural Gas Viscosit and Compressibility Factor
        Journal of Petroleum Science and Engineering May-15-2010
        Added a leading 0 to the coefficients to make the indicies the same
        Returns viscosity in lb/ft*s
        """    
        B = np.array([np.nan, 1.022872e0, -1.651432e0, 5.757386e0, -7.389282e-2, 8.389065e-2, 2.977476e-1, 1.451318e0, 4.682506e0, 1.918239e0, -9.844968e-2])
        rho = self.density(temperature_R, pressure_psia)
        MW_T = self.MW / (temperature_R-self.F_to_R)
        
        mu = np.log((B[1] + B[2] * MW_T + B[3] * MW_T**2+ B[4] * rho + B[5] * rho ** 2 + B[6] * rho ** 3) / \
                    (1 + B[7] * MW_T + B[8] * MW_T ** 2 + B[9] * MW_T ** 3 + B[10] * rho))
        return mu * 0.000671968994813

    def isothermal_comp(self, temperature_F, pressure_psia):
        """
        From New Correlations to Predict Natural Gas Viscosit and Compressibility Factor
        Journal of Petroleum Science and Engineering May-15-2010
        Isothermal compressibility of a gas
        c_g = 1/p - 1/z*(dz/dp)@Constant T
        """
        P = pressure_psia
        T = temperature_F
        dP = .001*P
        z = self.compressibility(T, P)
        d_dp = FinDiff(1, dP, acc=2)
        dz_dp = d_dp(self.compressibility)
        return 1/P + (1/z)*dz_dp



class Path:
    def __init__(self, points):
        self.points = points

    @classmethod
    def from_list_of_points(cls, Xs, Ys, Zs):
        if (len(Xs) != len(Ys)) or (len(Xs) != len(Zs)):
            raise Exception('number of points for North East and Elevation must be the same')

        points = pd.DataFrame(zip(Xs, Ys, Zs), columns=['Xs', 'Ys', 'Zs'])
        points['distance'] = 0
        points.distance = np.sqrt((self.points.Xs - self.points.Xs.shift(-1))** 2 + (self.points.Ys - self.points.Ys.shift(-1))** 2 + (self.points.Zs - self.points.Zs.shift(-1))** 2)
        points['cumm_distance'] = self.points.distance.cumsum().shift(1)
        points = points.fillna(0)
        return cls(points)

    @classmethod
    def from_list_of_lengths_and_angles(cls, lengths, angle_to_horizontal):
        if len(lengths) != len(angles):
            raise Exception('Number of lengths provided != number of angles')

        zeros = np.zeros(len(lengths) + 1)
        lengths = lengths[:]
        lengths.append(0)

        points = pd.DataFrame([0, 0, 0], columns=['Xs', 'Ys', 'Zs'])
        

        len_ang = pd.DataFrame(list(zip(lengths[:-1], angles)), columns=['len', 'ang'])

        len_ang['delta_X'] = np.abs(len_ang.len * np.cos(len_ang.ang * np.pi / 180))
        len_ang['delta_Z'] = len_ang.len * np.sin(len_ang.ang * np.pi / 180)

        new_points = pd.DataFrame(columns=['Xs', 'Ys', 'Zs'])
        new_points.Xs = len_ang.delta_X.cumsum()
        new_points.Zs = len_ang.delta_Z.cumsum()
        new_points.Ys = 0
        points = points.append(new_points, ignore_index=True, sort=False)
        points['distance'] = lengths
        points['cumm_distance'] = points.distance.cumsum().shift(1)
        points = points.fillna(0)
        return cls(points)

    def dh_dl(self, at_distance):
        prev_point = self.points.loc[self.points.cumm_distance <= at_distance].iloc[-1]
        next_point = self.points.loc[self.points.cumm_distance > at_distance].iloc[0]
        return (next_point.Zs - prev_point.Zs) / prev_point.distance
        
    def next_distance(self, at_distance):
        return self.points.loc[self.points.cumm_distance > at_distance].iloc[0]

class Pipe:
    def __init__(self, pipe_ID_in, roughness_in, path=None, lengths_ft=None, angles_from_horizontal_deg=None):
        self.ID = pipe_ID_in*12 #store in feet
        self.roughness = roughness_in * 12  #store in feet
        self.area = np.pi*self.ID**2/4
        if path:
            self.path = path
        elif lengths_ft and angles_from_horizontal_deg:
            self.path = Path.from_lengths_and_angles(lengths_ft, angles_from_horizontal_deg)
        else:
            raise Exception('Need a path or a length and angle')

        self.next_point_dist = self.path.cumm_distance.iloc[1]
        self._dh_dl = self.path.dh_dl(0)

    def dh_dl(self, at_distance):
        if at_distance > self.next_point_dist:
            self.next_point_dist = self.path.next_distance(at_distance)
            self._dh_dl = self.path.dh_dl(at_distance)
        return self._dh_dl               
        

class GasFlow:
    _std_P_psia = 14.696
    _std_T_F = 59.0
    _R_psi_ft3 = R_PSI_FT3_PER_LBMOL_DEGR  #psi*ft**3 /(deg_R * lbmol)
    _R_lbf = self._R_psi_ft3 *FT_LB_PER_PSI_FT3
    _scf_per_lbmol = self._R_psi_ft3 * (self.std_T_F + F_TO_R) / self.std_P_psia
    _g = G_ft_sec2  # standard gravity ft/sec^2

    def __init__(self, *, rate_mscfd, pipe, fluid, dp_with_flow=True):
        """
        dp_with_flow = True if solving dp in the direction of flow
        dp_with_flow = False if solving dp in the against the direction of flow
        """

        self.Q = rate_mscfd
        self.pipe = pipe
        self.fluid = fluid
        if dp_with_flow:
            self.direction = 1
        else:
            self.direction = -1
        self._mdot = self.mass_flowrate()
        self._full_turbulent_f = (1 / (-2 * np.log10((self.pipe.roughness / self.pipe.ID) / 3.7)))** 2
        self._partial_full_turb_switch = 35.235 * ((self.pipe.roughness / self.pipe.ID)** -1.1039)
        self._f = self._full_turbulent_f

    def reynolds_number(self):
        """
        Re = mdot*D_h/(A*mu) = mdot/(mu*D_h*pi/4) for circular pipes
        """
        ID = self.pipe.ID
        mu = self.fluid.viscosity(self.T, self.P)

        return self._mdot/(mu*ID*np.pi/4)
        
    @property
    def mass_flowrate(self):
        """Returns mass flow in lb/sec"""
        scfs = self.Q*1000/(24*60*60)
        return (scfs / self._scf_per_lbmol) * self.fluid.MW
        
    @mass_flowrate.setter
    def mass_flowrate(self, mdot_lbm_s):
        self._mdot = mdot_lbm_s
        scfs = self._mdot * self._scf_per_lbmol / self.fluid.MW
        self.Q = scfs*24*60*60/1000

    @property
    def P(self):
        return self.pressure
    
    @P.setter
    def P(self, P):
        self.pressure = P

    @property
    def T(self):
        return self.temperature

    @T.setter
    def T(self, temperature_F):
        self.temperature = temperature_F

    def velocity(self):
        return self._mdot / (self.fluid.density(self.T, self.P) * self.pipe.area)
        
    def friction_factor(self):
        """
        From Considerations About Equations for Steady State Flow In Natural Gas Pipelines
        Paulo M Coelho and Carlos Pinho 2007 
        J. of the Braz. Soc. of Mech. Sci. & Eng. July-September 2007, Vol. XXIX, No. 3 / 263
        """

        e = self.pipe.roughness
        ID = self.pipe.ID
        RE = self.reynolds_number()

        
        if RE > self._partial_full_turb_switch:
            return self._full_turbulent_f
        elif RE < 2500:
            return 64/RE
        else:
            f = self._f + .00001 # add small delta so that the while loop enters (2 orders of magnitude greater than condition)
            while abs(self._f-f)>.0000001:
                "Function is know to be recursive and quick to converge.  "
                self._f = f
                f = (1 / (-2 * np.log10(2.825 / (Re * f ** .5))))** 2
            
            self._f = f
            return f


    def dstate_dl(self, at_distance, pressure_psia, temperature_R = None):
        """
        
        """
        self.P = pressure_psia # make it easier to write the equations
        self.T = temperature_R # make it easier to write the equations
        rho = self.fluid.density(self.T, self.P)
        v = 
        f = self.friction_factor()
        D = self.pipe.ID
        
        dhdl = self.pipe.dhdl(at_distance)
        dudl = u / rho  # TODO
        Gravity_Force = rho * G_ft_sec2 / G_c_lbm_ft_per_lbf_s2 * dhdL
        Friction_Force = f*self.pipe.area*rho*((v_g**2)/(2*G_ft_sec2))/IN_PER_FT**2
        return np.array(dpdl, dvdl, dTdl)





def get_R_matrix_1P(U_matrix,theta,regimeNumber = 0):
    
    # Check if U_matrix is acceptable
    if U_matrix.min() < 0:
        return ValueError #np.asmatrix(np.zeros([7,1])),0
    P = np.float(U_matrix[0])/144 # psig
    v_g = np.float(U_matrix[1]) # ft/s
    T = np.float(U_matrix[2]) # degF
    
    dens_g = dens_gas(P,T,grav_g)  
    
    # Wall Shear Force Coefficients
    A_wg = 4/(2*r) # 1/ft
    d_hg = 2*r # ft
    Re_wg = dens_g * v_g * d_hg / (0.000672*visc_g) # dimensionless
    # Calculate Fanning Friction Factor at Wall Surface
    if Re_wg < 2300:      
        f_wg = 16/Re_wg
    else:
        f_wg = Colebrook(Re_wg,roughness,d_hg)
    F_wg = f_wg*A_wg*dens_g*pow(v_g,2)/2/32.174/144 # psi/ft
    
    F_gg = dens_g*a_g/a_g*math.sin(theta)/144 # psi/ft
    
    ##### Governing Matrix Elements for Two-Phase Flow #####
    # a11
    ddens_g__dp = grav_g*144/53.2146/(T+459.67)/144 # (lbm/ft^3)/(lbf/ft^2)
    a11 = v_g*ddens_g__dp
    
    # a12
    a12 = dens_g
    
    # a13
    ddens_g__dt = (-144*grav_g*P-144*grav_g*14.7) / (53.2146*pow(T+459.67,2))
    a13 = v_g*ddens_g__dt
    
    # a21
    a21 = 1
    
    # a22
    a22 = dens_g*v_g/32.174
    
    # a31
    Cp_g = 1.15*pow(1.008,T)*pow(T,-0.944)+0.533*pow(1.110,(P+14.7)/1000)*pow((P+14.7)/1000,0.0216)*pow(grav_g/0.6,0.025) # Specific Heat Capacity of Natural Gas - Moshfeghian (2011), Btu/lbm-degF
    eta_g = ((0.044-0.0474)/(130-110)*(T-110)+0.0474)/144 # Joule-Thomson Coefficient of Methane from NIST Website @ 115 psia, degF/(lbf/ft^2)    
    a31 = -dens_g*v_g*eta_g*Cp_g
    
    # a32
    J = 778.169 # Energy Conversion Factor, lbf-ft/Btu
    a32 = dens_g*pow(v_g,2)/32.174/J
    
    # a33
    a33 = -dens_g*v_g*Cp_g                                                                                     # I added negative !!!!       #######################################################################################################
    
    # b2
    b2 = (-F_gg - F_wg)*144
    
    # b3
    Q_st = 0 # Total heat input per unit volume, Btu/s/ft^3 - Assume insulated pipe (no heat transfer)
    b3 = Q_st - 1/J*(v_g*b2)  
    
    A_matrix = np.matrix([[a11,a12,a13],
                          [a21,a22,  0],
			         [a31,a32,a33]])

    B_matrix = np.matrix([[0],
				   [b2],
				   [b3]])
       
    #print("\nB:\n",B_matrix)
    #print("A:\n",A_matrix,"\nB:\n",B_matrix)
    R_matrix = A_matrix.I*B_matrix
    #R_matrix2 = B_matrix.T*A_matrix.I
    
    #return A_matrix,B_matrix
    return R_matrix, regimeNumber







