import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py
from scipy import linalg
from scipy import signal
from scipy.signal import butter, filtfilt
from noControl import matrix, TransferFunc, AR_model
from scipy.interpolate import interp1d

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
f = h5py.File("SaSR_test.hdf5", "r")
dset = f['SR/V1:ENV_CEB_SEIS_V_dec']
seism = f['SR/V1:ENV_CEB_SEIS_V_dec'][:] #seismic data
seism = seism[2000:]

#constants
nperseg = 2 ** 16 #samples per segment (useful for the PSD)
T = 1768 #signal duration in seconds
# dt = 1/62.5 #sampling frequency
t = np.linspace(0, T, len(seism)) #time vector

#parameter from noControl.py to be used in the time evolution
dt = 1e-3 #time step

ftransform = np.fft.fft(seism)
frequencies = np.fft.fftfreq(len(seism), d = 1/62.5)
    
X_f = np.zeros_like(ftransform, dtype=complex) #create an array of zeros with the same shape as V
nonzero = frequencies != 0 #boolean mask: true if freq is not zero
#for all non-zero frequencies, divide the FT by 2 pi f the take the IFT to get the displacement
X_f[nonzero] = ftransform[nonzero] / (1j * 2 * np.pi * frequencies[nonzero])
zt = np.fft.ifft(X_f)

#multiply the FT by 2 pi f then take the IFT to get the acceleration
acc = ftransform * (frequencies * 2 * np.pi * 1j)
At = np.fft.ifft(acc)


#calculate the PSDs
fAcc, psdAcc = signal.welch(At.real, fs = 62.5, window='hann', nperseg=nperseg)
fVel, psdVel = signal.welch(seism.real, fs = 62.5, window='hann', nperseg=nperseg)
fZ, psdZ = signal.welch(zt.real, fs = 62.5, window='hann', nperseg=nperseg)

half = len(seism) // 2  
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
    y0 = np.array(
        (0, 0, 0, 0, 0, 0, 0., 0., 0., 0., 0., 0.))  # initial condition
    y_t = np.copy(y0)  # create a copy to evolve it in time
    print(f"tt: {len(tt)}")
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

Nt_step = seism.size  # temporal steps = number of samples

# Parameters of the system
gamma = [5, 5, 5, 5, 5]  # viscous friction coeff [kg/m*s]
M = [160, 125, 120, 110, 325, 82]  # filter mass [Kg]
K = [700, 1500, 3300, 1500, 3400, 564]  # spring constant [N/m]

F = force_function

wn = 2*np.pi*frequencies
# Simulation 
physical_params = [*M, *K, *gamma, dt]
simulation_params = [AR_model, Nt_step, dt] 
signal_params = [M[0], At]  
#tt = np.linspace(0, Nt_step * dt, Nt_step, endpoint=False)

tt, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6 = (
                        evolution(*simulation_params, physical_params, signal_params,
                        F, file_name = None))

Tf, poles = TransferFunc(wn, *M, *K, *gamma)
H = (np.real(Tf) ** 2 + np.imag(Tf) ** 2) ** (1 / 2)

#x6_interp = interp1d(tt, x6, kind='linear', bounds_error=False, fill_value=0.0)(t)
#v6_interp = interp1d(tt, v6, kind='linear', bounds_error=False, fill_value=0.0)(t)

psdVel_interpolated = interp1d(fVel, psdVel, kind='linear', bounds_error=False, fill_value=0.0)(frequencies)
psdAcc_interpolated = interp1d(fAcc, psdAcc, kind='linear', bounds_error=False, fill_value=0.0)(frequencies)
psdZ_interpolated = interp1d(fZ, psdZ, kind='linear', bounds_error=False, fill_value=0.0)(frequencies)

xout_frequency_domain = psdZ_interpolated * H[5] 
xout_time_domain = np.fft.ifft(xout_frequency_domain)
vout_frequency_domain = psdVel_interpolated * H[5]
vout_time_domain = np.fft.ifft(vout_frequency_domain)
acc_out_frequency_domain = psdAcc_interpolated * H[5]
acc_out_time_domain = np.fft.ifft(acc_out_frequency_domain)

transfer_function_velocity = vout_frequency_domain / psdVel_interpolated
transfer_function_displacement = xout_frequency_domain / psdZ_interpolated
transfer_function_acceleration = acc_out_frequency_domain / psdAcc_interpolated

if __name__ == '__main__':
    SA = f[tower]

    #print the structure of the dataset
    print_hdf5_structure(f)
    
    #plot velocity and displacement (time domain)
    plt.figure(figsize=(8, 6))
    plt.suptitle('Seisimic data (V1:ENV_CEB_SEIS_V_dec)', fontsize=16, y = 0.95)

    plt.subplot(2, 1, 1)
    plt.plot(t, seism, label='Velocity')
    plt.ylabel('Amplitude [$\mu$m/s]')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(t, zt.real, label='Displacement', color='darkorange')
    plt.ylabel('Amplitude [$\mu$m]')
    plt.legend()
    plt.grid()
    plt.xlabel('Time [s]')
    plt.savefig('figures/velocity_displacement_top.png')
    #plt.show()

    plt.figure(figsize=(14, 4))

    plt.suptitle('Amplitude spectra', fontsize=16)

    plt.subplot(1, 3, 1)
    plt.loglog(fVel, np.sqrt(psdVel), label='Velocity')
    plt.ylabel('Amplitude [m/s/$\sqrt{Hz}$]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(which = 'both', axis = 'both')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.loglog(fZ, np.sqrt(psdZ), label='Displacement', color='darkorange')
    plt.ylabel('Amplitude [m/$\sqrt{Hz}$]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(which = 'both', axis = 'both')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.loglog(fAcc, np.sqrt(psdAcc), label ='Acceleration', color='green')
    plt.ylabel('Amplitude [m/s$^2$/$\sqrt{Hz}$]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(which = 'both', axis = 'both')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('figures/amplitude_spectra.png')

    fig = plt.figure(figsize=(5, 5))
    plt.title('Time evolution', size=13)
    plt.xlabel('Time [s]', size=12)
    plt.ylabel('x [$\mu$m]', size=12)
    plt.grid(True, ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()

    #plt.plot(tt, x1, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, M$_1$')
    #plt.plot(tt, x2, linestyle='-', linewidth=1, marker='', color='black', label='x2, M$_2$')
    #plt.plot(tt, x3, linestyle='-', linewidth=1, marker='', color='red', label='x3, M$_3$')
    #plt.plot(tt, x4, linestyle='-', linewidth=1, marker='', color='green', label='x4, M$_4$')
    #plt.plot(tt, x5, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x7, M$_7$')
    plt.plot(tt, x6, linestyle='-', linewidth=1, marker='',color='blue', label='x$_{out}$, M$_{out}$') #ultima massa
    plt.legend()
    #plt.savefig('figures/time_evolution.png')

    plt.figure(figsize=(10, 6))
    # plt.loglog(abs(frequencies),abs(Tf[5]), label="Theoretical TF", color="blue")
    plt.loglog(abs(frequencies), (transfer_function_displacement), label="Experimental TF (from v)", color="red", alpha=0.4, marker='o')
    plt.loglog(abs(frequencies), (transfer_function_velocity), label="Experimental TF (from x)", color="orange", alpha=0.4, marker='x', markersize =3)
    plt.loglog(abs(frequencies), (transfer_function_acceleration), label="Experimental TF (from acceleration)", color="green", alpha=0.4, marker='.', markersize=2)
    plt.loglog(frequencies[:half],H[5][:half], label="Theoretical TF", color="blue", alpha =0.7, linewidth=0.5)


    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True, which='both')
    plt.legend()
    plt.title("Transfer Function Comparison (calculated from PSDs)")
    #plt.ylim(1e-21, 1e5)
    plt.figure(figsize=(10, 6))
    plt.title('Displacement of final mass')
    plt.plot(t, xout_time_domain.real, linewidth=1, marker='', color='darkorange', label='x$_{out}$ (from IFT)') #ultima massa
    plt.ylabel('Displacement [$\mu$m]')
    plt.xlabel('Time [s]')
    plt.show()

