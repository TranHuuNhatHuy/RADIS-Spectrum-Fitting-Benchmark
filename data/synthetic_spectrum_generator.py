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
import radis

from radis import calc_spectrum, experimental_spectrum, plot_diff, Spectrum



# Generate a synthetic spectrum and log its information
def synthSpectrumGenerate():

    # -------------------------------------- EDIT HERE! -------------------------------------- #

    # Parameters for ground-truth.
    from_wl = 7500
    to_wl = 8000
    wunit = "cm-1"
    molecule = "O2"
    isotope = "1"
    pressure = 1.01325
    Tgas = 298
    Tvib = ""
    Trot = ""
    mole_fraction = 0.21
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

    spec_type = "LTE"

    # Generate synthetic spectrum based on parameters for ground-truth
    s = (
        calc_spectrum(
            from_wl,
            to_wl,
            wunit = wunit,                  # "cm-1" or "nm"
            Tgas = Tgas,                    # in K
            molecule = molecule,
            isotope = isotope,
            pressure = pressure,            # in bar
            mole_fraction = mole_fraction,
            path_length = path_length,      # in cm
            wstep = 0.001,
            databank = "hitran",
            name = name)
        .apply_slit(slit, slit_unit)        # Apply slit function to reduce resolution
        .offset(-0.2, "nm")                 # Make the spectrum to feature offset, typically 0.2 nm
    )
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
    spectrum_dir = f"./{spec_type}/spectrum/"

    if wunit == "cm-1":
        wunit = "cm" # To avoid mistakes with the pressure value next to it when reading fileName
    fileName_spec = f"{name}.spec"

    s_synth.store(spectrum_dir + fileName_spec)

    if wunit == "cm":
        wunit = "cm-1" # Revert cm back to cm-1


    # Log the information to ground-truth file
    fileName_json = f"{name}.json"
    gt_dir = f"./{spec_type}/ground-truth/{fileName_json}" # Grouth-truth file directory

    written_info = {
        "fileName": fileName_spec,
        "molecule": molecule,
        "isotope": isotope,
        "wmin": from_wl,
        "wmax": to_wl,
        "wunit": wunit,
        "pressure": pressure,
        "mole_fraction": mole_fraction,
        "path_length": path_length,
        "slit": f"{slit} {slit_unit}",
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
    # Remove temperatures that does not include in the fitting process (such as when LTE)
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