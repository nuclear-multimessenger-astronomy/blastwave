import numpy as np
try:
    from . import blastwave_extension as _extension
except ImportError:
    import blastwave_extension as _extension
from ._grid import Uniform

_C = 29979245800.0
_Mass_P = 1.672622e-24
_mJy = 1e-26
_MPC = 3.09e24
_MAS = 1.0 / 206264806.24709466

class Jet:
    def __init__(             # It is the user's responsibility to make sure input values are valid.
        self,
        profiles,             # [tuple of tabulated data]: (angles, Eiso, Lorentz factor)
        nwind,                # [wind density scale]: n = nwind * (r / 1e17)^-2 + nism (cm^-3)
        nism,                 # [ism density scale]: n = nwind * (r / 1e17)^-2 + nism (cm^-3)
        tmin=10.0,            # [simulation start time]: (s)
        tmax=1e10,           # [simulation end time]: (s)
        grid=Uniform(257),    # [cell edge angles]: start with 0 and end with pi.
        tail=True,            # [isotropic tail]: add an extremely low energy low velocity isotropic tail for safty
        spread=True,          # [spreading]: spread or not
        spread_mode=None,     # [spread mode]: "none", "ode" (VegasAfterglow-style), or "pde" (finite-volume). Overrides spread if set.
        k=2.0,                # [CSM density power-law index]: n_csm ∝ r^{-k}. k=0 ISM, k=2 wind.
        cal_level=1,          # [calibration level]: 0: no calibration. 1: BM all time. 2: smoothly go from BM to ST (ST is dangerous)
        rtol=1e-6,            # [primitive variable solver tolerance]: Don't change it unless you know what is going on.
        cfl=0.9,              # [cfl number]: Don't change it unless you know what is going on.
        include_reverse_shock=False,  # [reverse shock]: include reverse shock dynamics and emission
        sigma=0.0,            # [magnetization]: ejecta magnetization parameter
        eps_e_rs=0.1,         # [RS electron energy fraction]
        eps_b_rs=0.01,        # [RS magnetic energy fraction]
        p_rs=2.3,             # [RS electron spectral index]
        t0_injection=0.0,     # [energy injection timescale]: (s)
        l_injection=0.0,      # [energy injection luminosity]: (erg/s)
        m_dot_injection=0.0,  # [mass injection rate]: (g/s)
        magnetar_l0=None,      # [erg/s] isotropic-equiv injection luminosity (scalar or list; 0/None = disabled)
        magnetar_t0=None,      # [s] spin-down timescale (scalar or list)
        magnetar_q=None,       # power-law decay index (scalar or list)
        magnetar_ts=None,      # [s] injection start time (scalar or list; 0 = from beginning)
    ):
        # save
        theta, energy, lf = profiles
        self.theta_data = theta
        self.energy_data = energy
        self.lf_data = lf
        self.nwind = nwind
        self.nism = nism
        self.tmin = tmin
        self.tmax = tmax
        self.rtol = rtol
        self.cfl = cfl
        self.grid = grid
        self.tail = tail
        self.spread = spread
        self.spread_mode = spread_mode
        self.k = k
        self.cal_level = cal_level
        self.include_reverse_shock = include_reverse_shock
        self.sigma = sigma
        self.eps_e_rs = eps_e_rs
        self.eps_b_rs = eps_b_rs
        self.p_rs = p_rs
        self.t0_injection = t0_injection
        self.l_injection = l_injection
        self.m_dot_injection = m_dot_injection
        # Normalize magnetar params: scalar → single-element list, None → empty list
        def _to_list(v, default=None):
            if v is None:
                return [] if default is None else default
            if isinstance(v, (int, float)):
                return [float(v)]
            return list(v)
        ml0 = _to_list(magnetar_l0)
        n_ep = len(ml0)
        self.magnetar_l0 = ml0
        self.magnetar_t0 = _to_list(magnetar_t0, [1.0] * n_ep)
        self.magnetar_q = _to_list(magnetar_q, [2.0] * n_ep)
        self.magnetar_ts = _to_list(magnetar_ts, [0.0] * n_ep)

        # solve jet
        jet_config = self._configJet()
        self._jet = _extension.Jet(jet_config)
        self._jet.solveJet()
    
    # ---------- PDE original data ---------- #
    @property
    def t_pde(self):
        result = np.array(self._jet.getT())
        return result
    
    # original y: Msw, Mej, beta_gamma_sq, beta_th, R (shape = [5, ntheta, nt])
    @property
    def y_pde(self):
        result = np.array(self._jet.getY())
        return result
    
    @property
    def theta_pde(self):
        result = np.array(self._jet.getTheta())
        return result

    # ---------- PDE data interpolation ---------- #
    # rest mass excluded energy (erg / sr)
    def dE0_dOmega(self, t, theta):
        return self._jet.interpolateE0(t, theta)
    
    # swetp-up mass (g / sr)
    def dMsw_dOmega(self, t, theta):
        return self._jet.interpolateMsw(t, theta)
    
    # ejecta mass (g / sr)
    def dMej_dOmega(self, t, theta):
        return self._jet.interpolateMej(t, theta)

    # four velocity
    def beta_gamma(self, t, theta):
        return self._jet.interpolateBetaGamma(t, theta)
    
    # polar velocity
    def beta_theta(self, t, theta):
        return self._jet.interpolateBetaTh(t, theta)
    
    # radius (cm)
    def R(self, t, theta):
        return self._jet.interpolateR(t, theta)

    # ---------- Radiation Related ---------- #

    # Simply calculate time t by equal arrival time surface. Just for fun!
    def EATS(self, t, theta, phi, theta_v, z):
        return self._jet.calculateEATS(t, theta, phi, theta_v, z)

    # specific intensity at jet sphreical coordinate [cgs] Could be useful for debug
    def Intensity(self, t, nu, theta, phi, P, model="sync"):
        # config parameters
        self._jet.configParameters(P)

        # config radiation model
        self._jet.configIntensity(model)

        try:
            I = self._jet.calculateIntensity(t, nu, theta, phi)
        except Exception as e:
            raise e
        
        return I
    
    # flux density [mJy]
    def FluxDensity(self, t, nu, P, model="sync", rtol=1e-3, max_iter=100, force_return=True, flux_method=None, ebl=False):
        # config parameters
        self._jet.configParameters(P)

        # config radiation model
        if isinstance(model, str):
            self._jet.configIntensity(model)
        else:
            self._jet.configIntensityPy(model)

        # config flux method (reset each call to avoid stale state)
        self._jet.configFluxMethod(flux_method if flux_method is not None else "")

        try:
            L = self._jet.calculateLuminosity(t, nu, rtol, max_iter, force_return)
        except Exception as e:
            raise e

        flux = L * (1 + P["z"]) / 4 / np.pi / (P["d"] * _MPC) ** 2 / _mJy

        if ebl:
            tau = _extension.ebl_tau_array(np.atleast_1d(nu * (1 + P["z"])), P["z"])
            flux = flux * np.exp(-tau)

        return flux
    
    # flux density from forward shock only [mJy]
    def FluxDensity_forward(self, t, nu, P, model="sync", rtol=1e-3, max_iter=100, force_return=True, ebl=False):
        self._jet.configParameters(P)
        if isinstance(model, str):
            self._jet.configIntensity(model)
        else:
            self._jet.configIntensityPy(model)

        L = self._jet.calculateLuminosityForward(t, nu, rtol, max_iter, force_return)
        flux = L * (1 + P["z"]) / 4 / np.pi / (P["d"] * _MPC) ** 2 / _mJy

        if ebl:
            tau = _extension.ebl_tau_array(np.atleast_1d(nu * (1 + P["z"])), P["z"])
            flux = flux * np.exp(-tau)

        return flux

    # flux density from reverse shock only [mJy]
    def FluxDensity_reverse(self, t, nu, P, model="sync", rtol=1e-3, max_iter=100, force_return=True, ebl=False):
        self._jet.configParameters(P)
        if isinstance(model, str):
            self._jet.configIntensity(model)
        else:
            self._jet.configIntensityPy(model)

        L = self._jet.calculateLuminosityReverse(t, nu, rtol, max_iter, force_return)
        flux = L * (1 + P["z"]) / 4 / np.pi / (P["d"] * _MPC) ** 2 / _mJy

        if ebl:
            tau = _extension.ebl_tau_array(np.atleast_1d(nu * (1 + P["z"])), P["z"])
            flux = flux * np.exp(-tau)

        return flux

    # flux [erg/s/cm^2]
    def Flux(self, t, nu1, nu2, P, model="sync", rtol=1e-3, max_iter=100, force_return=True):
        # config parameters
        self._jet.configParameters(P)

        # config radiation model
        if isinstance(model, str):
            self._jet.configIntensity(model)
        else:
            self._jet.configIntensityPy(model)
        
        try:
            L = self._jet.calculateFreqIntL(t, nu1, nu2, rtol, max_iter, force_return)
        except Exception as e:
            raise e
        
        return L * (1 + P["z"]) / 4 / np.pi / (P["d"] * _MPC) ** 2
    
    def WeightedAverage(self, t, nu, P, radiation_model="sync", average_model="offset", rtol=1e-3, max_iter=100, force_return=True):
        # config parameters
        self._jet.configParameters(P)

        # config radiation model
        if isinstance(radiation_model, str):
            self._jet.configIntensity(radiation_model)
        else:
            self._jet.configIntensityPy(radiation_model)

        # config average model
        if isinstance(average_model, str):
            self._jet.configAvgModel(average_model)
        else:
            self._jet.configAvgModelPy(average_model)

        # calculate weighted average
        try:
            weighted_average = self._jet.WeightedAverage(t, nu, rtol, max_iter, force_return)
        except Exception as e:
            raise e
        
        return weighted_average

    # apparent superluminal motion [mas]
    def Offset(self, t, nu, P, model="sync", rtol=1e-3, max_iter=100, force_return=True):
        offset_cgs = self.WeightedAverage(t, nu, P, radiation_model=model, average_model="offset", rtol=rtol, max_iter=max_iter, force_return=force_return)

        return offset_cgs / P["d"] / _MPC / (1.0 + P["z"]) / (1.0 + P["z"]) / _MAS
    
    # size along the jet axis [mas]
    def SizeX(self, t, nu, P, model="sync", rtol=1e-3, max_iter=100, force_return=True):
        # calculate xscale. The following notes are for myself in case I forget what is going on.
        # First, ∫x dL = xc * ∫dL based on xc defination.
        # Then, 
        # sigma_x^2 = ∫(x - xc)^2 dL / ∫dL 
        #           = ∫(x^2 - 2*x*xc + xc^2)dL / ∫dL 
        #           = (∫x^2 dL - 2*xc*∫x dL + xc^2*∫dL) / ∫dL
        #           = (∫x^2 dL - 2*xc^2*∫dL + xc^2*∫dL) / ∫dL
        #           = (∫x^2 dL / ∫dL) - xc^2
        # So, I only need to calculate the weighted avergae of x^2, not the original expression.

        # calculate offset
        xc = self.WeightedAverage(t, nu, P, radiation_model=model, average_model="offset", rtol=rtol, max_iter=max_iter, force_return=force_return)

        # calculate x_sq
        x_sq = self.WeightedAverage(t, nu, P, radiation_model=model, average_model="sigma_x", rtol=rtol, max_iter=max_iter, force_return=force_return)

        # xscale
        sigma_x = np.sqrt(x_sq - xc * xc)

        return sigma_x / P["d"] / _MPC / (1.0 + P["z"]) / (1.0 + P["z"]) / _MAS
    
    # size perpendicular to the jet axis [mas]
    def SizeY(self, t, nu, P, model="sync", rtol=1e-3, max_iter=100, force_return=True):
        sigma_y_sq = self.WeightedAverage(t, nu, P, radiation_model=model, average_model="sigma_y", rtol=rtol, max_iter=max_iter, force_return=force_return)
        sigma_y = np.sqrt(sigma_y_sq)

        return sigma_y / P["d"] / _MPC / (1.0 + P["z"]) / (1.0 + P["z"]) / _MAS

    # [cgs] specific intensity at LOS frame coordinate (x_tilde, y_tilde). This method is intended for sky map.
    def IntensityOfPixel(self, t, nu, x_offset, y_offset, P, model="sync"):
        # config parameters
        self._jet.configParameters(P)

        # config Intensity model
        self._jet.configIntensity(model)

        # intensity
        try:
            intensity = self._jet.IntensityOfPixel(t, nu, x_offset, y_offset)
        except Exception as e:
            raise e
        
        return intensity
    
    def _configJet(self):
        # initialize parameter object
        jet_config = _extension.JetConfig()
        jet_config.nwind = self.nwind
        jet_config.nism = self.nism
        jet_config.k = self.k
        jet_config.tmin = self.tmin
        jet_config.tmax = self.tmax
        jet_config.rtol = self.rtol
        jet_config.cfl = self.cfl
        jet_config.spread = self.spread
        # Determine spread_mode: explicit setting takes precedence over bool
        if self.spread_mode is not None:
            jet_config.spread_mode = self.spread_mode
            jet_config.spread = (self.spread_mode != "none")
        else:
            jet_config.spread_mode = "pde" if self.spread else "none"
            jet_config.spread = self.spread
        jet_config.cal_level = self.cal_level
        jet_config.include_reverse_shock = self.include_reverse_shock
        jet_config.sigma = self.sigma
        jet_config.eps_e_rs = self.eps_e_rs
        jet_config.eps_b_rs = self.eps_b_rs
        jet_config.p_rs = self.p_rs
        jet_config.t0_injection = self.t0_injection
        jet_config.l_injection = self.l_injection
        jet_config.m_dot_injection = self.m_dot_injection
        jet_config.magnetar_l0 = self.magnetar_l0
        jet_config.magnetar_t0 = self.magnetar_t0
        jet_config.magnetar_q = self.magnetar_q
        jet_config.magnetar_ts = self.magnetar_ts

        # generate grid
        theta_edge = self.grid
        theta = np.array([(theta_edge[i] + theta_edge[i + 1]) / 2 for i in range(len(theta_edge) - 1)])
        jet_config.theta_edge = theta_edge

        # add isotropic tail
        if self.tail:
            self.energy_data[self.energy_data <= np.max(self.energy_data) * 1e-12] = np.max(self.energy_data) * 1e-12
            self.lf_data[self.lf_data <= 1.005] = 1.005

        # Estimate theta_c from the energy profile: angle where E drops to 1/e of peak
        e_peak = np.max(self.energy_data)
        above_threshold = self.theta_data[self.energy_data > e_peak / np.e]
        jet_config.theta_c = float(above_threshold[-1]) if len(above_threshold) > 0 else 0.1

        # interpolate initial condition to grid points
        E0 = np.interp(theta, self.theta_data, self.energy_data / 4.0 / np.pi / _C ** 2.0)
        lf0 = np.interp(theta, self.theta_data, self.lf_data)

        # get mej and initial velocity
        Mej0 = E0 / (lf0 - 1)
        beta0 = np.sqrt(1.0 - 1.0 / lf0 ** 2)
        
        # analytically expand (coast) the blastwave from t=0 to t=tmin
        R0 = beta0 * _C * self.tmin
        _R0_REF = 1e17
        Msw_ism = self.nism * _Mass_P * R0 ** 3 / 3.0
        Msw_csm = self.nwind * _Mass_P * _R0_REF ** self.k * R0 ** (3.0 - self.k) / (3.0 - self.k) if self.nwind > 0 else 0.0
        Msw0 = Msw_csm + Msw_ism
        Eb0 = E0 + Mej0 + Msw0
        
        # config jet initial condition
        jet_config.Eb = Eb0
        jet_config.Ht = np.zeros_like(theta)
        jet_config.Msw = Msw0
        jet_config.Mej = Mej0
        jet_config.R = R0
        
        return jet_config