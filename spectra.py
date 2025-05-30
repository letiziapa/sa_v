import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py
from scipy import signal, differentiate
from scipy.signal import butter, filtfilt
from noControl import matrix, TransferFunc, AR_model
from scipy.interpolate import interp1d
import scipy.integrate

np.random.seed(42)

#structure of the hdf5 object 
def print_hdf5_structure(obj, indent=0):
    """Stampa ricorsivamente la struttura di un file HDF5"""
    for key in obj:
        print(" " * indent + f"📂 {key}")
        if isinstance(obj[key], h5py.Group):
            print_hdf5_structure(obj[key], indent + 2)
        elif isinstance(obj[key], h5py.Dataset):
            print(" " * (indent + 2) + f"🔢 Dataset: {obj[key].shape}, {obj[key].dtype}")

def print_items(dset):
# with h5py.File('SaSR_test.hdf5', 'r') as f:
#     dset = f['SR/V1:ENV_CEB_SEIS_V_dec']
    for key, value in dset.attrs.items():
        print(f"{key}: {value}")

#--------------------------Channels---------------------------#
tower = 'SR'
channels = ['V1:Sa_' + tower + '_F0_LVDT_V_500Hz',
            'V1:Sa_' + tower + '_F1_LVDT_V_500Hz', 
            'V1:Sa_' + tower + '_F2_LVDT_V_500Hz',
            'V1:Sa_' + tower + '_F3_LVDT_V_500Hz', 
            'V1:Sa_' + tower + '_F4_LVDT_V_500Hz']                
            #not considering F7

#required data
#f = h5py.File("SaSR_test.hdf5", "r")
f = h5py.File("SaSR_test.hdf5", "r")
dset = f['SR/V1:ENV_CEB_SEIS_V_dec']
seism = f['SR/V1:ENV_CEB_SEIS_V_dec'][:] #seismic data
seism = (seism[2000:])*1e-6  #remove the first 2000 samples 
#seism = seism *1e-6
# modify the seismic data to add a noisy complex component zbar
sigma = 1/np.sqrt(2) #standard deviation of the noise
zbar = np.random.normal(0, sigma, len(seism)) + 1j * np.random.normal(0, sigma, len(seism))
#phases = np.exp(1j * 2 * np.pi * np.random.rand(len(seism))) #random phases
#seism = seism * zbar #add noise to the seismic data (in time domain or frequency domain?)

#constants
nperseg = 2 ** 16 #samples per segment (useful for the PSD)
#T = 1800 #signal duration in seconds
T = 1768 #since we have removed the first 2000 samples, the signal duration is reduced
t = np.linspace(0, T, len(seism)) #time vector

#parameter to be used in the time evolution
dt = 1e-3 #time step

#window = np.hanning(len(seism)) #Hanning window to remove spectral leakage
 #apply the window to the seismic data
#take the fourier transform of the data
ftransform = np.fft.fft(seism)
#ftransform = ftransform * zbar

frequencies = np.fft.fftfreq(len(seism), d = 1/62.5)
half = len(frequencies) // 2 #half of the frequencies array (positive frequencies only)

X_f = np.zeros_like(ftransform, dtype=complex) #create an array of zeros with the same shape as V
nonzero = frequencies != 0 #boolean mask: true if freq is not zero
# #nonzero = freq != 0 #boolean mask: true if freq is not zero
# #for all non-zero frequencies, divide the FT by 2 pi f the take the IFT to get the displacement
X_f[nonzero] = ftransform[nonzero] / (1j * 2 * np.pi * frequencies[nonzero])
zt = np.fft.ifft(X_f)

#multiply the FT by 2 pi f then take the IFT to get the acceleration
acc = ftransform * (frequencies * 2 * np.pi * 1j)
At = np.fft.ifft(acc)


#calculate the PSDs to plot the velocity and acceleration spectra
fAcc, psdAcc = signal.welch(At.real, fs = 62.5, window='hann', nperseg=nperseg)
fVel, psdVel = signal.welch(seism.real, fs = 62.5, window='hann', nperseg=nperseg)
fZ, psdZ = signal.welch(zt.real, fs = 62.5, window='hann', nperseg=nperseg)


def force_function(t, mass, acceleration):
    return mass * np.real(acceleration) 

def evolution(evol_method, Nt_step, dt, physical_params, signal_params,
              F, file_name = None):
    """
    Simulates the temporal evolution of the system under the influence of an
    external force.

    Parameters
    ----------
    evol_method : function
        The function used to evolve the system (e.g. Euler or ARMA methods).
    Nt_step : int
        The number of temporal steps to simulate.
    dt : float
        The time step size.
    physical_params : list
        The list of physical parameters for the system.
    signal_params : list
        The list of parameters for the external force signal.
    F : function
        The function modeling the external force.
    file_name : str, optional
        The name of the file to save simulation data. Default is None.

    Returns
    -------
    tuple
        A tuple containing the time grid and the arrays of velocities
        and positions for each mass.
    """
    # Initialize the problem
    tmax = Nt_step * dt  # maximum time
    tt = np.arange(0, tmax, dt)  # time grid
    #print(len(tt), 'time steps')  # number of time steps
    #tt = np.arange(0, T, 1/62.5) #time grid based on the sampling frequency
    y0 = np.array(
        (0, 0, 0, 0, 0, 0, 0., 0., 0., 0., 0., 0.))  # initial condition
    y_t = np.copy(y0)  # create a copy to evolve it in time
    #run the simulation on the finer time grid
    F_signal = F(tt, *signal_params)  # external force applied over time (cambia)


    # Initialize lists for velocities and positions
    v1, v2, v3, v4, v5, v6 = [[], [], [], [], [], []]
    x1, x2, x3, x4, x5, x6 = [[], [], [], [], [], []]

    # compute the system matrices
    A, B = matrix(*physical_params)

    # time evolution when the ext force is applied
    i = 0
    for t in tt:
        Fi = F_signal[i]  # evaluate the force at time t
        i = i + 1
        y_t = evol_method(y_t, A, B, Fi)  # evolve to step n+1
        v1.append(y_t[0])
        v2.append(y_t[1])
        v3.append(y_t[2])
        v4.append(y_t[3])
        v5.append(y_t[4])
        v6.append(y_t[5])
        x1.append(y_t[6])
        x2.append(y_t[7])
        x3.append(y_t[8])
        x4.append(y_t[9])
        x5.append(y_t[10])
        x6.append(y_t[11])

    # save simulation's data (if a file name is provided)
    if file_name is not None:
        data = np.column_stack((tt, v1, v2, v3, v4, v5, v6,
                                x1, x2, x3, x4, x5, x6))
       # np.savetxt(os.path.join(data_dir, file_name), data,
        #           header='time, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6')

    return (tt, np.array(v1), np.array(v2), np.array(v3), np.array(v4),
            np.array(v5), np.array(v6), np.array(x1), np.array(x2),
            np.array(x3), np.array(x4), np.array(x5), np.array(x6))

# temporal steps (for simulation)
Nt_step = 1768000
#physical parameters of the system
gamma = [5, 5, 5, 5, 5]  # viscous friction coeff [kg/m*s]
M = [160, 125, 120, 110, 325, 82]  # filter mass [Kg]
K = [700, 1500, 3300, 1500, 3400, 564]  # spring constant [N/m]

F = force_function

wn = 2*np.pi*frequencies[:half]

physical_params = [*M, *K, *gamma, dt]

# Interpolate the acceleration onto the simulation time grid
# Time vector for seismic data (real data)
tt_data = np.arange(0, T, 1 / 62.5)  

# Time vector for simulation
tmax = Nt_step * dt  
tt_sim = np.arange(0, tmax, dt)
print(len(tt_sim), 'time steps for simulation')
print(tmax, 'max time for simulation')

# Interpolation
interp_acc = interp1d(tt_data, At, kind='linear', bounds_error=False, fill_value=0.0)
At_interp = interp_acc(tt_sim)


simulation_params = [AR_model, Nt_step, dt] 
#signal_params = [M[0], At] 
signal_params = [M[0], At_interp] 

tt, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6 = (
                        evolution(*simulation_params, physical_params, signal_params,
                        F, file_name = None))

Tf, poles = TransferFunc(wn, *M, *K, *gamma)


# Compute the magnitude of the transfer function (from simulation)
H = (np.real(Tf) ** 2 + np.imag(Tf) ** 2) ** (1 / 2)
print(len(x6), 'x6 Number of samples')
#apply a Hanning window to the data to remove spectral leakage
window = np.hanning(len(seism))

x6_interp = interp1d(tt_sim, x6, kind='linear', bounds_error=False, fill_value=0.0)(tt_data)
v6_interp = interp1d(tt_sim, v6, kind='linear', bounds_error=False, fill_value=0.0)(tt_data)
# vf_out = np.fft.fft(v6_interp*window)  # output signal (v6)
# vf_in = np.fft.fft(seism*window)  # input signal (seismic data)
# #output in frequency domain (resampled to match the data time vector)
#xf_in = np.fft.fft(X_f)  # input signal (zt)
#xf_out = np.fft.fft(x6_interp)
#xf_out = np.fft.fft(x6_resampled*window)
# Experimental transfer function
force = F(tt_sim, M[0], At_interp)  # external force applied over time
force_interp = interp1d(tt_sim, force, kind='linear', bounds_error=False, fill_value=0.0)(tt_data)

#only keep positive frequencies
#the frequencies array is symmetric, so we only need the first half
xf_in = np.fft.fft(zt * window)
xf_in = xf_in[:half]
xf_out = np.fft.fft(x6_interp*window)
#xf_out = xf_out[:half]
vf_in = np.fft.fft(seism)  # input signal (seismic data)
vf_out = np.fft.fft(v6_interp)  # output signal (v6)
#trfn = vf_out / vf_in
# # Compute transfer function and its magnitude
trfn = xf_out / np.fft.fft(force_interp*window) # experimental transfer function
# trfn = (vf_out)/vf_in
Hfn = (np.real(trfn) ** 2 + np.imag(trfn) ** 2) ** (1 / 2)
Hfn_mag = np.abs(Hfn)

# xout_frequency_domain = trfn * X_f
# xout_time_domain = np.fft.ifft(xout_frequency_domain)
#vout_frequency_domain = vf_in * trfn # output in frequency domain (vout)
#vout_time_domain = np.fft.ifft(vout_frequency_domain)
if __name__ == '__main__':
    #print the structure of the dataset
    print_hdf5_structure(f)
    print_items(dset)
    
    #plot velocity and displacement (time domain)
    plt.figure(figsize=(8, 6))
    plt.suptitle('Seisimic data (V1:ENV_CEB_SEIS_V_dec)', fontsize=16, y = 0.95)

    plt.subplot(2, 1, 1)
    plt.plot(tt_data, seism, label='Velocity')
    plt.ylabel('Amplitude [m/s]')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(tt_data, zt.real, label='Displacement', color='darkorange')
    plt.ylabel('Amplitude [m]')
    plt.legend()
    plt.grid()
    plt.xlabel('Time [s]')
    #plt.savefig('figures/velocity_displacement.png')

    # plt.figure(figsize=(14, 4))

    # plt.suptitle('Amplitude spectra', fontsize=16)

    # plt.subplot(1, 3, 1)
    # plt.loglog(fVel, np.sqrt(psdVel), label='Velocity')
    # plt.ylabel('Amplitude [m/s/$\sqrt{Hz}$]')
    # plt.xlabel('Frequency [Hz]')
    # plt.grid(which = 'both', axis = 'both')
    # plt.legend()

    # plt.subplot(1, 3, 2)
    # plt.loglog(fZ, np.sqrt(psdZ), label='Displacement', color='darkorange')
    # plt.ylabel('Amplitude [m/$\sqrt{Hz}$]')
    # plt.xlabel('Frequency [Hz]')
    # plt.grid(which = 'both', axis = 'both')
    # plt.legend()

    # plt.subplot(1, 3, 3)
    # plt.loglog(fAcc, np.sqrt(psdAcc), label ='Acceleration', color='green')
    # plt.ylabel('Amplitude [m/s$^2$/$\sqrt{Hz}$]')
    # plt.xlabel('Frequency [Hz]')
    # plt.grid(which = 'both', axis = 'both')
    # plt.legend()
    # plt.tight_layout()
    #plt.savefig('figures/amplitude_spectra.png')

    fig = plt.figure(figsize=(9, 5))
    plt.title('Transfer function without control', size=13)
    plt.xlabel('Frequency [Hz]', size=12)
    plt.ylabel('|x$_{out}$/x$_0$|', size=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which='both', ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()

    #plt.plot(freq, H[0], linestyle='-', linewidth=1, marker='', color='steelblue', label='output $x_1$')
    plt.plot(abs(frequencies)[:half], H[5], linestyle='-', linewidth=1, marker='', color='steelblue', label='output $x_{pl}$')
    plt.legend()
    #plt.savefig('figures/FREQARRAY_transfer_function_no_control.png')
    fig = plt.figure(figsize=(8, 5))
    plt.title('Time evolution', size=13)
    plt.xlabel('Time [s]', size=12)
    plt.ylabel('x [m]', size=12)
    plt.grid(True, ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()

    #plt.plot(tt, x1, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, M$_1$')
    #plt.plot(tt, x2, linestyle='-', linewidth=1, marker='', color='black', label='x2, M$_2$')
    #plt.plot(tt, x3, linestyle='-', linewidth=1, marker='', color='red', label='x3, M$_3$')
    #plt.plot(tt, x4, linestyle='-', linewidth=1, marker='', color='green', label='x4, M$_4$')
    #plt.plot(tt, x5, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x7, M$_7$')
    plt.plot(tt_sim, x6, linestyle='-', linewidth=1, marker='',color='blue', label='x$_{out}$, M$_{out}$') #ultima massa
    plt.legend()
    #plt.savefig('figures/time_evolution.png')

    plt.figure(figsize=(8, 5))
    plt.loglog(abs(frequencies)[:half], H[5][:half], label="Theoretical TF", color="blue")
    plt.loglog(abs(frequencies),abs(trfn), label="Experimental TF", color="red", alpha=0.7)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True, which='both')
    plt.legend()
    plt.title("Transfer Function Comparison")
    plt.savefig('figures/NUOVAtransfer_function_comparison.png')

    # plt.figure(figsize=(10, 5))
    # #plt.plot(tt_sim, vout_time_domain, label='Output (vout)', color='steelblue')
    # #plt.plot(tt_sim, vout_time_domain, label='Output (vout)', color='steelblue')

    # plt.plot(tt_sim, v6_interp, label='Output (v6)', color='darkorange', alpha=0.7)
    # plt.legend()
    # plt.figure(figsize=(10, 4))
    # plt.plot(tt_data, zt.real, label='Top displacement (zt)', color='darkorange')
    # plt.plot(tt_data, x6_resampled, label='Bottom displacement (x6)', color='steelblue', alpha=0.7)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Displacement [m]")
    # plt.title("Top vs Bottom Displacement (Time Domain)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    # plt.figure(figsize=(10, 5))

# # Frequency axis (must match FFT length)
    
#     freq_pos = frequencies[:half]

# # Magnitudes
#     mag_in = np.abs(xf_in)
#     mag_out = np.abs(xf_out)

# # Plot
#     plt.loglog(freq_pos, mag_in, label='Input (zt)', color='darkorange')
#     plt.loglog(freq_pos, mag_out, label='Output (x6)', color='steelblue', alpha=0.7)
#     plt.xlabel("Frequency [Hz]")
#     plt.ylabel("Magnitude")
#     plt.title("FFT Magnitudes: zt vs x6_resampled")
#     plt.grid(True, which='both', ls='--', alpha=0.5)
#     plt.legend()
#     plt.tight_layout()


    plt.show()

