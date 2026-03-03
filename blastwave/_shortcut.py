from ._blastwave import Jet
from ._grid import *
from ._jet_type import *

def FluxDensity_tophat(t, nu, P, tmin=10.0, tmax=1e10, spread=True, cal_level=1, cfl=0.9, model="sync", rtol=1e-3, max_iter=100, force_return=True, include_rs=False, rs_params=None, flux_method=None, magnetar_l0=None, magnetar_t0=None, magnetar_q=None, magnetar_ts=None, ebl=False):
    # Extract RS parameters if provided
    rs_kwargs = {}
    if include_rs:
        rs_kwargs["include_reverse_shock"] = True
        if rs_params:
            for key in ("sigma", "eps_e_rs", "eps_b_rs", "p_rs", "duration", "t0_injection", "l_injection", "m_dot_injection"):
                if key in rs_params:
                    rs_kwargs[key] = rs_params[key]

    # simulation
    jet = Jet(
        TopHat(P["theta_c"], P["Eiso"], lf0=P["lf"]),    # jet profile
        P["A"],                        # wind number density scale
        P["n0"],                       # ism number density scale
        tmin=tmin,                     # [simulation start time]: (s)
        tmax=tmax,                    # [simulation end time]: (s)
        grid=ForwardJetRes(P["theta_c"], 129),    # [cell edge angles]: must start with 0 and end with pi.
        tail=True,                     # [isotropic tail]: add an extremely low energy low velocity isotropic tail for safty
        spread=spread,                   # w/wo spreading effect
        cal_level=cal_level,                   # [calibration level]: 0: no calibration. 1: BM all time. 2: smoothly go from BM to ST (dangerous)
        rtol=1e-6,                     # [primitive variable solver tolerance]: Don't change it unless you know what is going on.
        cfl=cfl,                       # [cfl number]: Don't change it unless you know what is going on.
        eps_e=P.get("eps_e", 0.0),
        eps_b=P.get("eps_b", 0.0),
        p_fwd=P.get("p", 2.3),
        magnetar_l0=magnetar_l0,
        magnetar_t0=magnetar_t0,
        magnetar_q=magnetar_q,
        magnetar_ts=magnetar_ts,
        **rs_kwargs,
    )

    # flux density
    flux = jet.FluxDensity(
        t,           # [second] observing time span
        nu,                # [Hz]     observing frequency
        P,                 # parameter dictionary
        model=model,      # emissivity model
        rtol=rtol,         # integration tolerance
        max_iter=max_iter,
        force_return=force_return,
        flux_method=flux_method,
        ebl=ebl,
    )

    return flux

def FluxDensity_gaussian(t, nu, P, tmin=10.0, tmax=1e10, spread=True, cal_level=1, cfl=0.9, model="sync", rtol=1e-3, max_iter=100, force_return=True, include_rs=False, rs_params=None, flux_method=None, magnetar_l0=None, magnetar_t0=None, magnetar_q=None, magnetar_ts=None, ebl=False):
    # Extract RS parameters if provided
    rs_kwargs = {}
    if include_rs:
        rs_kwargs["include_reverse_shock"] = True
        if rs_params:
            for key in ("sigma", "eps_e_rs", "eps_b_rs", "p_rs", "duration", "t0_injection", "l_injection", "m_dot_injection"):
                if key in rs_params:
                    rs_kwargs[key] = rs_params[key]

    # simulation
    jet = Jet(
        Gaussian(P["theta_c"], P["Eiso"], lf0=P["lf"]),    # jet profile
        P["A"],                        # wind number density scale
        P["n0"],                       # ism number density scale
        tmin=tmin,                     # [simulation start time]: (s)
        tmax=tmax,                    # [simulation end time]: (s)
        grid=ForwardJetRes(P["theta_c"], 129),    # [cell edge angles]: must start with 0 and end with pi.
        tail=True,                     # [isotropic tail]: add an extremely low energy low velocity isotropic tail for safty
        spread=spread,                   # w/wo spreading effect
        cal_level=cal_level,                   # [calibration level]: 0: no calibration. 1: BM all time. 2: smoothly go from BM to ST (dangerous)
        rtol=1e-6,                     # [primitive variable solver tolerance]: Don't change it unless you know what is going on.
        cfl=cfl,                       # [cfl number]: Don't change it unless you know what is going on.
        eps_e=P.get("eps_e", 0.0),
        eps_b=P.get("eps_b", 0.0),
        p_fwd=P.get("p", 2.3),
        magnetar_l0=magnetar_l0,
        magnetar_t0=magnetar_t0,
        magnetar_q=magnetar_q,
        magnetar_ts=magnetar_ts,
        **rs_kwargs,
    )

    # flux density
    flux = jet.FluxDensity(
        t,           # [second] observing time span
        nu,                # [Hz]     observing frequency
        P,                 # parameter dictionary
        model=model,      # emissivity model
        rtol=rtol,         # integration tolerance
        max_iter=max_iter,
        force_return=force_return,
        flux_method=flux_method,
        ebl=ebl,
    )

    return flux

def FluxDensity_spherical(t, nu, P, k=2.0, tmin=10.0, tmax=1e10, ntheta=17, cal_level=1, model="sync", rtol=1e-3, max_iter=100, force_return=True, include_rs=False, rs_params=None, flux_method=None, magnetar_l0=None, magnetar_t0=None, magnetar_q=None, magnetar_ts=None, ebl=False):
    # Extract RS parameters if provided
    rs_kwargs = {}
    if include_rs:
        rs_kwargs["include_reverse_shock"] = True
        if rs_params:
            for key in ("sigma", "eps_e_rs", "eps_b_rs", "p_rs", "duration", "t0_injection", "l_injection", "m_dot_injection"):
                if key in rs_params:
                    rs_kwargs[key] = rs_params[key]

    # simulation — spherical = isotropic, no spreading, no tail needed
    jet = Jet(
        Spherical(P["Eiso"], lf0=P["lf"]),
        P["A"],
        P["n0"],
        tmin=tmin,
        tmax=tmax,
        grid=Uniform(ntheta),
        tail=False,
        spread=False,
        spread_mode="none",
        k=k,
        cal_level=cal_level,
        rtol=1e-6,
        eps_e=P.get("eps_e", 0.0),
        eps_b=P.get("eps_b", 0.0),
        p_fwd=P.get("p", 2.3),
        magnetar_l0=magnetar_l0,
        magnetar_t0=magnetar_t0,
        magnetar_q=magnetar_q,
        magnetar_ts=magnetar_ts,
        **rs_kwargs,
    )

    # flux density
    flux = jet.FluxDensity(
        t,
        nu,
        P,
        model=model,
        rtol=rtol,
        max_iter=max_iter,
        force_return=force_return,
        flux_method=flux_method,
        ebl=ebl,
    )

    return flux

def FluxDensity_powerlaw(t, nu, P, tmin=10.0, tmax=1e10, spread=True, cal_level=1, cfl=0.9, model="sync", rtol=1e-3, max_iter=100, force_return=True, include_rs=False, rs_params=None, flux_method=None, magnetar_l0=None, magnetar_t0=None, magnetar_q=None, magnetar_ts=None, ebl=False):
    # Extract RS parameters if provided
    rs_kwargs = {}
    if include_rs:
        rs_kwargs["include_reverse_shock"] = True
        if rs_params:
            for key in ("sigma", "eps_e_rs", "eps_b_rs", "p_rs", "duration", "t0_injection", "l_injection", "m_dot_injection"):
                if key in rs_params:
                    rs_kwargs[key] = rs_params[key]

    # simulation
    jet = Jet(
        PowerLaw(P["theta_c"], P["Eiso"], lf0=P["lf"], s=P["s"]),    # jet profile
        P["A"],                        # wind number density scale
        P["n0"],                       # ism number density scale
        tmin=tmin,                     # [simulation start time]: (s)
        tmax=tmax,                    # [simulation end time]: (s)
        grid=ForwardJetRes(P["theta_c"], 129),    # [cell edge angles]: must start with 0 and end with pi.
        tail=True,                     # [isotropic tail]: add an extremely low energy low velocity isotropic tail for safty
        spread=spread,                   # w/wo spreading effect
        cal_level=cal_level,                   # [calibration level]: 0: no calibration. 1: BM all time. 2: smoothly go from BM to ST (dangerous)
        rtol=1e-6,                     # [primitive variable solver tolerance]: Don't change it unless you know what is going on.
        cfl=cfl,                       # [cfl number]: Don't change it unless you know what is going on.
        eps_e=P.get("eps_e", 0.0),
        eps_b=P.get("eps_b", 0.0),
        p_fwd=P.get("p", 2.3),
        magnetar_l0=magnetar_l0,
        magnetar_t0=magnetar_t0,
        magnetar_q=magnetar_q,
        magnetar_ts=magnetar_ts,
        **rs_kwargs,
    )

    # flux density
    flux = jet.FluxDensity(
        t,           # [second] observing time span
        nu,                # [Hz]     observing frequency
        P,                 # parameter dictionary
        model=model,      # emissivity model
        rtol=rtol,         # integration tolerance
        max_iter=max_iter,
        force_return=force_return,
        flux_method=flux_method,
        ebl=ebl,
    )

    return flux
