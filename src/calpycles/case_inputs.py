# mypy: ignore-errors
# flake8: noqa
# noqa
# pylint: skip-file

names_by_case = {
    "DYCOMS_RF01": {
        # "fields_lower":["u","v","w","s","temperature"],
        "fields_lower": ["w", "s", "temperature"],
        "fields_upper": ["ql", "qt"],
        # "fields_upper":["ql","qt","temperature","buoyancy_frequency"],
        "profiles": [
            "w_mean2",
            "w_mean3",
            "qt_mean",
            "ql_mean",
            "thetali_mean",
            "cloud_fraction",
        ],
        "timeseries": [
            "boundary_layer_height",
            "lwp",
            "cloud_fraction",
            "cloud_top",
            "cloud_base",
            "cloud_top_mean",
            "cloud_base_mean",
        ],
        "spectra": ["energy_spectrum"],
        "times": [14400],
        "t0_timeseries": 3600,  # 1 h spiup
        "average_times": slice(7200, 14400),  # final 2h
        # "times":[120],
        # "times":[0,3600,7200,10800,14400],
        "z_lims": [0, 1200],
        "z_levels": [800],
        "parameters": [
            "vg",
            "ug",
            "divergence",
            "zi",
            "thetal_g",
            "thetal_1",
            "qtg",
            "qt_1",
            "ft",
            "fq",
            "cm",
            "sst",
            "p_surface",
        ],
    },
}

lims_by_var = {
    "boundary_layer_height": [0, 1200],
    "energy_spectrum": [1e-4, 1e0],  # [1e2, 1e8],
    "energy_spectrum_w": [1e-4, 1e0],  # [1e2, 1e8],
    "qt_spectrum": [1e-12, 1e-8],
    "s_spectrum": [1e-5, 1e0],
    "thetali_spectrum": [1e-6, 1e-3],
    # # DYCOMS-II: as in (Pressel 2017)
    # "cloud_fraction":[0.2,1.1],
    # "lwp":[0,0.12],
    # "cloud_top":[200,900],
    # "cloud_base":[200,900],
    # # DYCOMS-II: enlarged parameter ranges
    "cloud_fraction": [0.7, 1.1],
    "lwp": [0, 0.2],
    "cloud_top": [0, 1200],
    "cloud_top_ts_fit": [0, 1200],
    "cloud_base": [0, 1200],
    "cloud_base_ts_fit": [0, 1200],
}

nan_by_var = {
    "cloud_base": 99999.9,
    "cloud_top": -99999.9,
    "cloud_base_mean": 99999.9,
    "cloud_top_mean": -99999.9,
}
