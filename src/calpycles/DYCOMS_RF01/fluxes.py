"""Calculate constant surface fluxes for the DYCOMS RF01 case."""

# Third-party
import numpy as np

# mypy: ignore-errors
# pylint: disable=invalid-name

# constants (form Stevens2005)
p0 = 1000e2  # reference pressure, Pa
p = 1017.8e2  # surface pressure, Pa
eps = 0.622  # ratio of gas constants
cp = 1.015 * 1e3  # specific heat capacity of air (between dry and moist) [J/kg/K]
L = 2.47 * 1e6  # latent heat of vaporization, J/kg
rho = 1.22  # air density at surface
u0 = np.sqrt(6**2 + 4.25**2)  # weakened geostrophic wind in boundary layer, m/s


def get_T_from_theta(theta):  # K
    """Convert potential temperature to temperature."""
    return theta / (p0 / p) ** 0.286  # K


def get_theta_from_T(T):  # K
    """Convert temperature to potential temperature."""
    return T * (p0 / p) ** 0.286  # K


def get_saturation_pressure(T):  # K
    """Calculate saturation pressure from temperature (from metpy, Bolton1980)."""
    T = T - 273.15  # C
    es = 6.112 * np.exp(17.67 * T / (T + 243.5)) * 1e2
    return es  # Pa


def get_saturation_specific_humidity(T):  # K
    """Calculate saturation specific humidity from temperature."""
    es = get_saturation_pressure(T)  # Pa
    qts = es * eps / (p - (1 - eps) * es)
    return qts  # kg/kg


def lh_flux(sst=292.5, qtg=9e-3, cm=0.0011, u=u0):
    """Calculate the latent heat flux.

    Args
        sst: Sea surface temperature in Kelvin. Default is 292.5 K.
        qtg: Specific humidity in kg/kg. Default is 9e-3 kg/kg.
        cm=cd=cq: Transfer coefficient (dimensionless). Default is 0.0011.
        u: Wind speed in m/s.

    Returns
        float: Latent heat flux in W/m².

    """
    qts = get_saturation_specific_humidity(sst)  # kg/kg
    return rho * L * cm * (qts - qtg) * u  # W/m2


def sh_flux(sst=292.5, T=290.4, cm=0.0011, u=u0):
    """Calculate the sensible heat flux.

    Args
        sst: Sea surface temperature in Kelvin. Default is 292.5 K.
        T: Temperature in Kelvin. Default is 290.4 K.
        cm=cd=cq: Transfer coefficient (dimensionless). Default is 0.0011.
        u: Wind speed in m/s.

    Returns
        float: Sensible heat flux in W/m².

    """
    return rho * cp * cm * (sst - T) * u  # W/m2


def test_fluxes():
    """Test the fluxes computation."""
    print("### Default values (from the paper):")
    print(f"{sh_flux(sst=292.5,T=290.4)=:2.2f} (reported: 15)")
    print(f"{lh_flux(sst=292.5,)=:2.2f} (reported: 115)")
    print()
    print("### Adjusted values to fit the fluxes:")
    print(f"{sh_flux(sst=292.37,T=290.87)=:2.2f} (reported: 15)")
    print(f"{lh_flux(sst=292.37)=:2.2f} (reported: 115)")


if __name__ == "__main__":

    test_fluxes()
