import numpy as np

import bilby

import lalsimulation as lalsim
import lal
import argparse
import os
import glob
import sys
from astropy.cosmology import z_at_value, Planck18
from astropy import units

roll_of = 0.4


def parse_cmd():
    parser=argparse.ArgumentParser()
    parser.add_argument("--psd_dir", help = "directory containing power spectral densities of detectors")
    args=parser.parse_args()
    return args
args=parse_cmd()


chirp_mass =  1.3
mass_ratio = 0.875
a_1 = 0.
a_2 = 0.
tilt_1 = 0.
tilt_2 = 0.
phi_12 = 0.
luminosity_distance = 75
theta_jn =1.5
phi_jl = 0.
psi=0
phase=1.0
ra = 3.0
dec = -1.2

trigger_time = 1264079376 
minimum_frequency = 10
sampling_frequency = 4096
duration =400
start_time = trigger_time - duration

psd_files=glob.glob(args.psd_dir+'*.txt')

#z=z_at_value(Planck18.luminosity_distance,float(luminosity_distance)*units.Mpc).value
approximant = 'TaylorF2ThreePointFivePN'
#sys.exit()

outdir = '/home/jason/Bilby/my_own_runs/bilby_run10/bilby_run3.0_-1.2/outdir'+approximant+'/'+str(luminosity_distance)+'_'+str(int(chirp_mass*100)/100.)+'_'+str(int(chirp_mass*100)/100.)+'/'
if(not os.path.exists(outdir)):
    os.makedirs(outdir)
label = 'bns_example'
#outdir = 'pe_dir'
bilby.core.utils.setup_logger(outdir=outdir, label=label)
logger = bilby.core.utils.logger

#file_to_det={'H1':"aligo",'L1':'aligo','V1':'avirgo','K1':'kagra'}
chirp_mass_min=0.92
chirp_mass_max=1.7


#minimum_frequency = 10
reference_frequency = 20


#DICTIONARY FOR INJECTION VALUES
#np.random.seed(88170235)
injection_parameters = dict(
    chirp_mass=chirp_mass,
    mass_ratio=mass_ratio,
    a_1=a_1, 
    a_2=a_2,
    tilt_1=tilt_1, 
    tilt_2=tilt_2, 
    theta_jn=theta_jn,
    luminosity_distance=luminosity_distance, 
    phi_jl=phi_jl,
    psi=psi, 
    phase=phase, 
    geocent_time=trigger_time, 
    phi_12=phi_12,
    ra=ra, 
    dec=dec
)


#GENERATE A WAVEFORM
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
#    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=dict(
        waveform_approximant=approximant,
        reference_frequency=reference_frequency,
    )
)


'''
#PSD TO INTERFEROMETER 
psd_filenames = {
    'H1': 'aligo_O4high_extrapolated.txt',
    'L1': 'aligo_O4high_extrapolated.txt',
    'V1': 'avirgo_O4high_NEW.txt'
}

ifo_list = bilby.gw.detector.InterferometerList([])
for det in ["H1", "L1", "V1"]:
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    freq, asd = np.loadtxt(psd_filenames[det], unpack=True)
    psd = asd**2
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freq, psd_array=psd
    )
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time
    )
    ifo_list.append(ifo)

ifo_list.inject_signal(
    parameters=injection_parameters,
    waveform_generator=waveform_generator
)

logger.info("Finished Injecting signal")
logger.info("Saving IFO data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)
'''





#PSD TO INTERFEROMETER
file_to_det={'H1':"aligo",'L1':'aligo','V1':'avirgo','K1':'kagra'}
interferometers =bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1','K1'])

for ifo in interferometers:
    for fn in psd_files:
        if file_to_det[ifo.name] in fn:
            print(ifo.name,fn)
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=fn)

interferometers.set_strain_data_from_zero_noise(sampling_frequency, duration, start_time=injection_parameters['geocent_time'] - duration + 2.)
interferometers.inject_signal(
    parameters=injection_parameters,
    waveform_generator=waveform_generator
)




'''#PRIORS
priors = bilby.gw.prior.BNSPriorDict()
priors.pop("mass_1")
priors.pop("mass_2")

mc=injection_parameters['chirp_mass']
priors['chirp_mass'].minimum = mc * 0.95
priors['chirp_mass'].maximum = mc * 1.05
priors["mass_ratio"].minimum = 0.125
priors["mass_ratio"].maximum = 1
#priors['chi_1'] = bilby.core.prior.Uniform(minimum=-0.05, maximum=0.05, name="chi_1")
#priors['chi_2'] = bilby.core.prior.Uniform(minimum=-0.05, maximum=0.05, name="chi_2")
#priors['lambda_1'] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000., name="lambda_1")
#priors['lambda_2'] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000., name="lambda_1")
#priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$'
)


    luminosity_distance=bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=min(10,float(luminosity_distance)-10), maximum=max(450,float(luminosity_distance)+450), unit='Mpc', latex_label='$d_L$'),
priors["luminosity_distance"] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=min(10,float(D)-10), maximum=max(100,float(D)+100), unit='Mpc', latex_label='$d_L$')
'''



#PRIORS
prior_dictionary = dict(
    chirp_mass=bilby.gw.prior.Uniform(name='chirp_mass', minimum=1.25, maximum=1.35),
    mass_ratio=bilby.gw.prior.Uniform(name='mass_ratio', minimum=0.33, maximum=1),
    mass_1=bilby.gw.prior.Constraint(name='mass_1', minimum=1, maximum=2.4),
    mass_2=bilby.gw.prior.Constraint(name='mass_2', minimum=1, maximum=2.4),
    a_1=bilby.gw.prior.Uniform(name='a_1', minimum=0, maximum=0.05,
                               latex_label='$a_1$', unit=None, boundary=None),
    a_2=bilby.gw.prior.Uniform(name='a_2', minimum=0, maximum=0.05,
                               latex_label='$a_2$', unit=None, boundary=None),
    tilt_1=bilby.core.prior.DeltaFunction(peak=0.0),
    tilt_2=bilby.core.prior.DeltaFunction(peak=0.0),
    phi_12=bilby.core.prior.DeltaFunction(peak=0.0),
    phi_jl=bilby.gw.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi,
                                  boundary='periodic', latex_label='$\\phi_{JL}$', unit=None),
    luminosity_distance=bilby.gw.prior.UniformComovingVolume(name='luminosity_distance',
                                                             minimum=10, maximum=500, latex_label='$d_L$',
                                                             unit ='Mpc', boundary=None),
    dec=bilby.core.prior.DeltaFunction(peak=1.16535),
    ra=bilby.core.prior.DeltaFunction(peak=2.9254),
    theta_jn=bilby.prior.Sine(name='theta_jn', latex_label='$\\theta_{JN}$',
                              unit=None, minimum=0, maximum=np.pi, boundary=None),
    psi=bilby.gw.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic',
                               latex_label='$\\psi$', unit=None)
)

priors = bilby.gw.prior.BBHPriorDict(dictionary=prior_dictionary)

# set a small margin on time of arrival
priors['geocent_time'] = bilby.core.prior.DeltaFunction(
    peak=trigger_time
)


#ROQ LIKELIHOOD
search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.binary_neutron_star_roq,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=dict(
        waveform_approximant=approximant,
        reference_frequency=reference_frequency
    )
)
roq_params = np.array(
    [(minimum_frequency, sampling_frequency / 2, duration, chirp_mass_min, chirp_mass_max, 0)],
    dtype=[("flow", float), ("fhigh", float), ("seglen", float), ("chirpmassmin", float), ("chirpmassmax", float), ("compmin", float)]
)
likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
    interferometers = interferometers, 
    waveform_generator = search_waveform_generator,
    priors = priors,
    linear_matrix="/home/jason/Bilby/my_own_runs/bilby_run02.1/roq/basis_256s.hdf5", 
    quadratic_matrix="/home/jason/Bilby/my_own_runs/bilby_run02.1/roq/basis_256s.hdf5",
    roq_params=roq_params,
    distance_marginalization=False, 
    phase_marginalization=True
)

# SAMPLING
npool = 100
nact = 10
nlive = 2000
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler='dynesty',
    use_ratio=True,
    nlive=nlive,
    walks=100,
    maxmcmc=5000,
    nact=nact,
    npool=npool,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
#    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    result_class=bilby.gw.result.CBCResult,
)   


result.plot_corner()
