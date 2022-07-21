# -*- coding: utf-8 -*-
"""

RADIS Synthetic Spectrum Generator for Fitting Benchmarking Process

This module is to synthesize various spectra for the benchmarking process.

Generated spectra are stored in ./data/<spectrum-type>/spectrum in RADIS .spec file,
while their corresponding ground-truths are stored in ./data/<spectrum-type>/ground-truth
in JSON format.

For each spectrum type, after acquiring most stable fitting pipeline, the model,
the algorithm and other specific settings will be stored as a model file in
./fitting-module/<spectrum-type>/model.py.

"""

import json
import numpy as np
import math

from radis import plot_diff, Spectrum, SpectrumFactory



# Generate a synthetic spectrum and log its information
def synthSpectrumGenerate():

    # -------------------------------------- EDIT HERE! -------------------------------------- #

    # Parameters for ground-truth.
    from_wl = 2000
    to_wl = 2300
    wunit = "cm-1"
    molecule = "CO"
    isotope = "1"
    pressure = 0.1
    Tgas = ""
    Tvib = 2000
    Trot = 600
    mole_fraction = 0.5
    wstep = 0.001
    path_length = 1
    slit = 1
    slit_unit = "nm"
    name = f"synth-{molecule}-{isotope}-{from_wl}-{to_wl}-{wunit}-P{pressure}-t{Tgas}-v{Tvib}-r{Trot}-mf{mole_fraction}-p{path_length}-sl{slit}{slit_unit}"

    # Fitting pipeline required
    method = "leastsq"
    fit_var = "radiance"
    normalize = True
    max_loop = 200

    # ---------------------------------------------------------------------------------------- #

    spec_type = "nonLTE"

    # Generate synthetic spectrum based on parameters for ground-truth
    sf = SpectrumFactory(
        wmin = from_wl,
        wmax = to_wl,
        wunit = wunit,                  # "cm-1" or "nm"
        molecule = molecule,
        isotope = isotope,
        pressure = pressure,            # in bar
        mole_fraction = mole_fraction,
        path_length = path_length,      # in cm
        wstep = wstep,
    )

    if spec_type == "LTE":
        sf.fetch_databank(
            "hitran",
            load_columns = "equilibrium"
        )
        s = sf.eq_spectrum(Tgas = Tgas)
    else:
        sf.fetch_databank(
            "hitemp",
            load_columns = "noneq"
        )
        s = sf.non_eq_spectrum(
            Tvib = Tvib,
            Trot = Trot,
        )
    
    s.apply_slit(slit, slit_unit)        # Apply slit function to reduce resolution
    s.offset(-0.2, "nm")                 # Make the spectrum to feature offset, typically 0.2 nm
    
    # Show it to make sure nothing goes wrong
    s.plot(show = True)


    s_wav, s_val = s.get('radiance')

    # Reproduce noises for the spectrum. The noise should be around 0.5% of max radiance
    noise_scale = max([val for val in s_val if not(math.isnan(val))]) * 0.005 # Prevent NaN
    s_val += np.random.normal(size = s_val.size, scale = noise_scale)

    # Reproduce dilatation by applying non-linear transformation, with scale 0.66% of deviation
    # wav_mean = np.mean([wav for wav in s_wav if not(math.isnan(wav))])
    # print(wav_mean)
    # s_wav = wav_mean + (s_wav - wav_mean) * 1.0066


    # Re-generate noise-added spectrum
    s_synth = Spectrum(
        {
            fit_var : (s_wav, s_val)
        },
        wunit = wunit,
        units = {
            fit_var : s.units[fit_var]
        }
    )
    # Show it to make sure nothing goes wrong, again
    s_synth.plot(show = True)


    # Save the above spectrum
    save_dir = f"./{spec_type}/ground-truth/"

    if wunit == "cm-1":
        wunit = "cm" # To avoid mistakes with the pressure value next to it when reading fileName
    fileName_spec = f"{name}.spec"

    s_synth.store(save_dir + fileName_spec)

    if wunit == "cm":
        wunit = "cm-1" # Revert cm back to cm-1


    # Log the information to ground-truth file
    fileName_json = f"{name}.json"
    gt_dir = save_dir + fileName_json

    written_info = {
        "model": {
            "fileName": fileName_spec,
            "molecule": molecule,
            "isotope": isotope,
            "wmin": from_wl,
            "wmax": to_wl,
            "wunit": wunit,
            "pressure": pressure,
            "mole_fraction": mole_fraction,
            "wstep": wstep,
            "path_length": path_length,
            "slit": f"{slit} {slit_unit}"
        },
        "fit": {
            "Tgas": Tgas,
            "Tvib": Tvib,
            "Trot": Trot,
        },
        "pipeline": {
            "method": method,
            "fit_var": fit_var,
            "normalize": normalize,
            "max_loop": max_loop,
        }
    }
    # Remove temperatures that does not include in the fitting process
    if written_info["fit"]["Tgas"] == "":
        written_info["fit"].pop("Tgas")
    if written_info["fit"]["Tvib"] == "":
        written_info["fit"].pop("Tvib")
    if written_info["fit"]["Trot"] == "":
        written_info["fit"].pop("Trot")

    # Write the information of spectrum into a new JSON file
    with open(gt_dir, 'w') as f:
        json.dump(written_info, f, indent = 2)
        print("JSON file successfully created.")

    # Plot difference between 2 spectra to make sure lol
    plot_diff(s, s_synth, method = ["diff", "ratio"], show = True)



if __name__ == "__main__":

    synthSpectrumGenerate()