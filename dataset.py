import torch.multiprocessing as mp
import torch
import numpy as np
from tqdm import tqdm

class HybridFieldChannelDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        # variables
        self.set_size = kwargs.get('set_size')
        self.miniset_size = kwargs.get('miniset_size')
        assert self.set_size % self.miniset_size == 0, 'set_size must be divisible by miniset_size'

        self.n_antennas = kwargs.get('n_antennas')
        self.n_rf = kwargs.get('n_rf')
        self.n_paths_min = kwargs.get('n_paths_min')
        self.n_paths_max = kwargs.get('n_paths_max')
        self.carrier_freq = kwargs.get('carrier_freq')
        self.los_path_len = kwargs.get('los_path_len')
        self.scat_dist_min = kwargs.get('scat_dist_min')
        self.scat_dist_max = kwargs.get('scat_dist_max')
        self.snr_db_min = kwargs.get('snr_db_min')
        self.snr_db_max = kwargs.get('snr_db_max')
        self.n_workers = kwargs.get('n_workers')
        self.n_sc = kwargs.get('n_subcarriers')
        self.bandwidth = kwargs.get('bandwidth')

        # system
        self.device = kwargs.get('device')
        self.dtype = np.complex64
        self.real_dtype = np.float32

        # constants
        self.c = 3e8                                                                                                # speed of light
        self.n_t = 2.24 - 0.025j                                                                                    # refractive index
        self.sigma_rough = 0.088e-3                                                                                 # roughness factor
        self.lambda_c = self.c / self.carrier_freq                                                                  # carrier wavelength
        self.d_a = self.lambda_c / 5                                                                                # antenna spacing
        self.d_sub = 56 * self.lambda_c                                                                             # subarray spacing
        self.frequencies = self.carrier_freq + (np.arange(self.n_sc)-(self.n_sc-1)/2) * (self.bandwidth/self.n_sc)  # subcarries
        self.k_abs = np.loadtxt(f'data/k_abs/k_abs_{self.n_sc}_sc.txt')                                             # molecular absorption coefficients
        self.array_apperture = np.sqrt(2) * ((np.sqrt(self.n_antennas / self.n_rf) - 1) * self.d_a * np.sqrt(self.n_rf) + \
                                             (np.sqrt(self.n_rf) - 1) * self.d_sub)
        self.rayleigh_distance = 2 * (self.array_apperture**2) / self.lambda_c
        self.delay_spread = 0.1 * self.los_path_len / self.c
        los_spread = self.c / (4 * np.pi * self.frequencies * self.los_path_len)        # LoS spread loss
        los_abs = np.exp(-0.5 * self.k_abs * self.los_path_len)                         # LoS absorption loss
        self.los_alpha = los_spread * los_abs                                           # LoS attenuation

        # others
        self.antenna_array_shape = [int(np.sqrt(self.n_antennas)), int(np.sqrt(self.n_antennas))]
        self.subarray_shape = [int(np.sqrt(self.n_rf)), int(np.sqrt(self.n_rf))]
        self.array_of_subarrays_shape = [int(np.sqrt(self.n_antennas/self.n_rf)), int(np.sqrt(self.n_antennas/self.n_rf))]
        self.subarray_length = int(np.sqrt(self.n_antennas // self.n_rf))
        self.dft_upa_dict = self.get_upa_dict()                             # Generate the dictionary matrix for each component UPA

        meas_dict = np.load(kwargs.get('meas_mat_path'), allow_pickle=True).item()
        self.meas_mat = np.array(meas_dict['meas_mat'], dtype=self.dtype)
        self.w_rf = np.array(meas_dict['w_rf'], dtype=self.dtype)

        assert (self.meas_mat.shape[1] == self.n_antennas) and (self.w_rf.shape[1] == self.n_antennas)

    def get_planar_response(self, theta, phi, subc_ind):
        response = np.zeros(self.antenna_array_shape, dtype=self.dtype)    # Initialize response matrix
        for n1 in range(self.subarray_shape[0]):
            for n2 in range(self.subarray_shape[1]):
                for m1 in range(self.array_of_subarrays_shape[0]):
                    for m2 in range(self.array_of_subarrays_shape[1]):
                        length_subarray_x = (self.array_of_subarrays_shape[0] - 1) * self.d_a
                        length_subarray_y = (self.array_of_subarrays_shape[1] - 1) * self.d_a
                        
                        position_x = (n1 * length_subarray_x + n1 * self.d_sub + m1 * self.d_a)
                        position_y = (n2 * length_subarray_y + n2 * self.d_sub + m2 * self.d_a)
                        
                        index_y = n1 * self.array_of_subarrays_shape[0] + m1
                        index_x = n2 * self.array_of_subarrays_shape[1] + m2
                        
                        response[index_x, index_y] = (1 / np.sqrt(self.n_antennas)) * np.exp(
                            1j * 2 * np.pi * self.frequencies[subc_ind] * 
                            (position_x * np.sin(theta) * np.cos(phi) +
                            position_y * np.sin(theta) * np.sin(phi)) / self.c
                        )
        return response

    def get_spherical_response(self, theta, phi, r_l, subc_ind):
        response = np.zeros(self.antenna_array_shape, dtype=self.dtype)     # Initialize response matrix
        
        x = r_l * np.cos(phi) * np.sin(theta)
        y = r_l * np.sin(phi) * np.sin(theta)
        z = r_l * np.cos(theta)

        for n1 in range(self.subarray_shape[0]):
            for n2 in range(self.subarray_shape[1]):
                for m1 in range(self.array_of_subarrays_shape[0]):
                    for m2 in range(self.array_of_subarrays_shape[1]):
                        length_subarray_x = (self.array_of_subarrays_shape[0] - 1) * self.d_a
                        length_subarray_y = (self.array_of_subarrays_shape[1] - 1) * self.d_a
                        
                        position_x = (n1 * length_subarray_x + n1 * self.d_sub + m1 * self.d_a)
                        position_y = (n2 * length_subarray_y + n2 * self.d_sub + m2 * self.d_a)
                        
                        under_sqrt = (x - position_x)**2 + (y - position_y)**2 + z**2
                        d = np.sqrt(under_sqrt)
                        
                        index_y = n1 * self.array_of_subarrays_shape[0] + m1
                        index_x = n2 * self.array_of_subarrays_shape[1] + m2
                        
                        response[index_x, index_y] = (1 / np.sqrt(self.n_antennas)) * \
                            np.exp(-1j * 2 * np.pi * self.frequencies[subc_ind] * d / self.c)
        return response

    def get_hybrid_field_channel(self, rng):
        n_paths = rng.integers(low=self.n_paths_min, high=self.n_paths_max+1, dtype=np.uint8)
        tau = rng.uniform(low=self.los_path_len/self.c, high=self.los_path_len/self.c+self.delay_spread, size=n_paths).astype(self.real_dtype)  # delays of NLoS paths
        tau[0] = self.los_path_len / self.c                                                                                                     # delay of LoS path
        r_l = rng.uniform(low=self.scat_dist_min, high=self.scat_dist_max, size=n_paths-1).astype(self.real_dtype)                              # distances from scatterers to center of array
        parphi_in = (np.pi / 2) * rng.random(n_paths-1, dtype=self.real_dtype)                                                                  # incident angles for NLoS paths
        parphi_ref = np.arcsin((1 / self.n_t) * np.sin(parphi_in))                                                                              # refraction angles for NLoS paths
        theta = np.pi * rng.random(n_paths, dtype=self.real_dtype) - np.pi / 2                                                                  # Elevation AoA
        phi = 2 * np.pi * rng.random(n_paths, dtype=self.real_dtype) - np.pi                                                                    # Azimuth AoA
        channel = np.zeros((self.n_sc, self.antenna_array_shape[0], self.antenna_array_shape[1]), dtype=self.dtype)                             # channel buffer

        for subc_ind in range(self.n_sc):
            for path_idx in range(n_paths):
                if path_idx == 0:   # LoS path
                    if self.los_path_len > self.rayleigh_distance:
                        path_response = self.los_alpha[subc_ind] * self.get_planar_response(theta[path_idx], phi[path_idx], subc_ind) * \
                            np.exp(-1j * 2 * np.pi * self.frequencies[subc_ind] * tau[path_idx])
                    else:
                        path_response = self.los_alpha[subc_ind] * self.get_spherical_response(theta[path_idx], phi[path_idx], self.los_path_len, subc_ind) * \
                            np.exp(-1j * 2 * np.pi * self.frequencies[subc_ind] * tau[path_idx])
                else:               # NLoS path
                    gamma = (np.cos(parphi_in[path_idx-1]) - self.n_t * np.cos(parphi_ref[path_idx-1])) / \
                            (np.cos(parphi_in[path_idx-1]) + self.n_t * np.cos(parphi_ref[path_idx-1]))
                    exp_factor = -8 * (np.pi ** 2) * (self.frequencies[subc_ind]**2) * (self.sigma_rough**2) * \
                                (np.cos(parphi_in[path_idx-1])**2) / (self.c**2)
                    rho = np.exp(exp_factor)
                    alpha = abs(gamma * rho) * self.los_alpha[subc_ind]

                    if r_l[path_idx-1] > self.rayleigh_distance:
                        path_response = alpha * self.get_planar_response(theta[path_idx], phi[path_idx], subc_ind) * \
                            np.exp(-1j * 2 * np.pi * self.frequencies[subc_ind] * tau[path_idx])
                    else:
                        path_response = alpha * self.get_spherical_response(theta[path_idx], phi[path_idx], r_l[path_idx-1], subc_ind) * \
                            np.exp(-1j * 2 * np.pi * self.frequencies[subc_ind] * tau[path_idx])
                
                channel[subc_ind] += path_response

        channel_power = np.sum(np.abs(channel)**2)
        channel *= np.sqrt(self.n_sc * self.n_antennas / channel_power)    # Normalize the channel
        return channel

    def get_upa_dict(self):
        # Create DFT matrices
        ver_basis = (1 / np.sqrt(np.array(self.subarray_length, dtype=self.real_dtype))) * np.fft.fft(np.eye(self.subarray_length, dtype=self.real_dtype)).astype(self.dtype)
        hor_basis = (1 / np.sqrt(np.array(self.subarray_length, dtype=self.real_dtype))) * np.fft.fft(np.eye(self.subarray_length, dtype=self.real_dtype)).astype(self.dtype)
        # Create a 2D-DFT basis by the means of Kronecker product
        return np.kron(ver_basis, hor_basis)

    def transform_by_subarray(self, channel):
        channel_transf = []
        for i in range(int(np.sqrt(self.n_rf))):
            for j in range(int(np.sqrt(self.n_rf))):
                # Extract subarray from H
                channel_subarray = channel[:, i * self.subarray_length:(i + 1) * self.subarray_length, 
                                        j * self.subarray_length:(j + 1) * self.subarray_length]
                result = np.einsum('gik,mkj->mij', self.dft_upa_dict.T.conj()[None, :, :], channel_subarray.reshape([self.n_sc, self.dft_upa_dict.shape[0],1], order='F'))
                channel_transf.append(result)
        channel_transf = np.concatenate(channel_transf, axis=1)
        return channel_transf

    def get_channel_response(self, rng):
        channel = self.get_hybrid_field_channel(rng)
        return self.transform_by_subarray(channel)

    def get_measurement(self, channel, rng):
        snr_db = rng.uniform(low=self.snr_db_min, high=self.snr_db_max, size=1)
        sigma = np.sqrt(1 / 10**(snr_db / 10), dtype=self.real_dtype)
        noise = sigma * (rng.standard_normal((self.n_sc, self.n_antennas), dtype=self.real_dtype) + 1j * rng.standard_normal((self.n_sc, self.n_antennas), dtype=self.real_dtype))/np.sqrt(2, dtype=self.real_dtype)
        meas = self.meas_mat[None, :, :] @ channel + self.w_rf @ noise[:, :, None]
        return meas.squeeze(-1)

    def _getitem(self, rng):
        channel = self.get_channel_response(rng)
        channel_re_im = np.concatenate((np.real(channel), np.imag(channel)), axis=1).squeeze()
        measurement = self.get_measurement(channel, rng)
        measurement_re_im = np.concatenate((measurement.real, measurement.imag), axis=1)
        return channel_re_im, measurement_re_im

    def _get_miniset(self, seed):
        rng = np.random.default_rng(seed)
        channels = np.zeros((self.miniset_size, self.n_sc, 2*self.n_antennas), dtype=self.real_dtype)
        measurements = np.zeros((self.miniset_size, self.n_sc, 2*self.w_rf.shape[0]), dtype=self.real_dtype)
        for i in range(self.miniset_size):
            channels[i], measurements[i] = self._getitem(rng)
        return channels, measurements

    def update_dataset(self, init_seed):
        seeds = (init_seed*self.set_size//self.miniset_size+np.arange(self.set_size//self.miniset_size)).astype(int)
        with mp.Pool(processes=self.n_workers) as p:
            out_list = list(tqdm(p.imap_unordered(self._get_miniset, seeds), total=self.set_size//self.miniset_size))
        channels, measurements = [list(x) for x in zip(*out_list)]
        channels = np.concatenate(channels, axis=0)
        measurements = np.concatenate(measurements, axis=0)
        self.channels = torch.tensor(channels, dtype=torch.float, device=self.device)
        self.measurements = torch.tensor(measurements, dtype=torch.float, device=self.device)

    def delete_dataset(self):
        del self.channels
        del self.measurements

    def __len__(self):
        return self.set_size
    
    def __getitem__(self, idx):
        return self.channels[idx], self.measurements[idx]

if __name__ == '__main__':

    args_dict = {
        'set_size': 10,
        'miniset_size': 10,
        'n_antennas': 1024,
        'n_rf': 4, 
        'n_paths_min': 5,
        'n_paths_max': 5,
        'carrier_freq': 300e9,
        'n_subcarriers': 32,
        'bandwidth': 15e9,
        'los_path_len': 30,
        'scat_dist_min': 10,
        'scat_dist_max': 25,
        'snr_db_min': 0,
        'snr_db_max': 20,
        'meas_mat_path': 'data/CSmatrix1024_512_AoSA.npy',
        'n_workers': 128
    }

    dataset = HybridFieldChannelDataset(**args_dict)
    dataset.update_dataset(0)
    print(dataset.measurements)
    print(dataset.channels)