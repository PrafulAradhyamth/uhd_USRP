import multiprocessing
import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
import reedsolo
import os
import pickle
import glob
import pandas as pd
from pathlib import Path
import re
import hashlib
import os
from typing import Dict, List, Any
from tqdm import tqdm
# import mmwave.dsp as dsp
from idcodes.idcodes_polarfec import POLARFEC_U8, POLARFEC_U16, POLARFEC_U32, POLARFEC_U64
from sim.idsimulation import IDSIMULATION
from sim.dict_util import write_dict, load_dict

class RxProcessor:
    """
    A class to encapsulate the functionalities for processing received radio frequency (RF) signals.
    This includes reading complex wave files, packet detection, BPSK demodulation,
    FEC decoding, SNR calculation, and saving results.
    """

    def __init__(self, info_length=8, code_length=128, rs_ecc_length=16):
        """
        Initializes the RxProcessor with parameters for FEC and ID simulation.

        Args:
            info_length (int): The information length for polar codes.
            code_length (int): The codeword length for polar codes.
            rs_ecc_length (int): The error correction capability for Reed-Solomon codec.
        """
        self.info_length = info_length
        self.code_length = code_length
            # Load the pilot signal
        pilot_path = os.path.join('.', 'refdata', 'pilot.npy')
        if not os.path.exists(pilot_path):
            raise FileNotFoundError(f"Pilot file not found at {pilot_path}. Please ensure the file exists.")
        pilot = np.load(pilot_path)
        self.pilot = pilot.astype(np.complex64)
        self.polar = POLARFEC_U8(self.info_length, self.code_length)
        self.rs = reedsolo.RSCodec(rs_ecc_length - 1)
        # These will be initialized dynamically based on the config.pkl file per folder
        self.index_config = None
        self.idsim = None

    def load_num_from_pkl(self, pkl_path="config.pkl"):
        """
        Load an integer from a pickle file.

        Args:
            pkl_path (str): Path to the pickle file.

        Returns:
            int: The integer value loaded from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        # take the path 
       
        if not Path(pkl_path).exists():
            raise FileNotFoundError(f"File {pkl_path} does not exist.")

        with open(pkl_path, 'rb') as f:
            num = pickle.load(f)
        return num

    def rs_decode_single_py_1(self, bits_1):
        """
        Decodes Reed-Solomon encoded bits.

        Args:
            bits_1 (numpy.ndarray): 128 bits (8 bytes) to decode.

        Returns:
            numpy.ndarray: Decoded bits as a numpy array, or zeros on error.
        """
        try:
            code_bytes = np.packbits(bits_1).tobytes()
            decoded = self.rs.decode(code_bytes)
            if isinstance(decoded, tuple):
                decoded = decoded[0]
            return np.unpackbits(np.frombuffer(decoded, dtype=np.uint8)).astype(np.int32)
        except reedsolo.ReedSolomonError:
            return np.zeros(8, dtype=np.int32)

    def read_complex_file(self, file_path):
        """
        Reads a complex64 file and returns the data as a numpy array.

        Args:
            file_path (str): Path to the complex64 file.

        Returns:
            numpy.ndarray: A numpy array of complex64 data, or None if an error occurs.
        """
        try:
            data = np.fromfile(file_path, dtype=np.complex64)
            return data
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def decode_fec(self, bits, fec_type='polar'):
        """
        Decode FEC bits using the specified FEC type.

        Args:
            bits (numpy.ndarray): A numpy array of bits to decode.
            fec_type (str): Type of FEC to use ('polar' or 'rs').

        Returns:
            numpy.ndarray: Decoded bits as a numpy array.

        Raises:
            ValueError: If an unsupported FEC type is provided.
        """
        if fec_type == 'polar':
            llr = [1.0 if bit == 0 else -1.0 for bit in np.unpackbits(bits)]
            if len(llr) < 128:
                # 'i' and 'count_failures' were global variables in the original script
                # and are not directly accessible here. Adjusting for class context.
                llr = np.zeros(128).tolist()
            decoded = self.polar.decode_polarfec(llr)
            decoded_bits = [int(bit) for bit in decoded]
            return np.packbits(decoded_bits)
        elif fec_type == 'rs':
            try:
                code_bytes = np.packbits(bits).tobytes() # Corrected from bits_1 to bits
                decoded = self.rs.decode(code_bytes)
                if isinstance(decoded, tuple):
                    decoded = decoded[0]
                return np.unpackbits(np.frombuffer(decoded, dtype=np.uint8)).astype(np.int32)
            except reedsolo.ReedSolomonError:
                return np.zeros(8, dtype=np.int32)
        elif fec_type == 'eh':
            print("EH FEC decoding is not implemented yet.")
            return np.zeros(self.info_length * 8, dtype=np.uint8) # Return a placeholder array
        else:
            raise ValueError("Unsupported FEC type. Use 'polar', 'rs', or 'eh'.")

    def read_bits_file(self, file_path):
        """
        Reads a bits file and returns the data as a numpy array of uint8.

        Args:
            file_path (str): Path to the bits file.

        Returns:
            numpy.ndarray: A numpy array of uint8 data, or None if an error occurs.
        """
        try:
            data = np.fromfile(file_path, dtype=np.uint8)
            return data
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def find_packet_starts_ind(self,rx_signal, preamble, threshold_ratio=0.8, packet_length=1024):
        # Perform correlation (use 'valid' mode so preamble fits entirely within rx_signal)
        matched_filter = np.conj(preamble[::-1])
        output = np.convolve(rx_signal, matched_filter, mode='valid')
        correlation_magnitude = np.abs(output)
        
        # Set a threshold to detect peaks in correlation
        threshold = threshold_ratio * np.max(correlation_magnitude)
        
        # Find all indices where the correlation exceeds the threshold
        packet_starts = np.where(correlation_magnitude > threshold)[0]
        
        # Optional: Filter out nearby duplicates
        # a Minimum gap of packet_length to avoid multiple detections of the same packet
        filtered_starts = []
        last_index = -packet_length
        for idx in packet_starts:
            if idx - last_index >= packet_length:
                filtered_starts.append(idx)
                last_index = idx

        print("filtered starts: num pakets: ",  len(filtered_starts))
        return filtered_starts

    def calculate_snr(self, pilot, received_signal):
        """
        Calculate the Signal-to-Noise Ratio (SNR) between a known signal and a received signal.

        Args:
            pilot (numpy.ndarray): The ideal known signal (complex64 datatype).
            received_signal (numpy.ndarray): The received signal (complex64 datatype).

        Returns:
            tuple: A tuple containing:
                - float: SNR in decibels (dB).
                - numpy.ndarray: The segment of the received signal aligned with the pilot.

        Raises:
            ValueError: If aligned segment is shorter than pilot or lengths mismatch.
        """
        pilot = np.array(pilot, dtype=complex)
        received_signal = np.array(received_signal, dtype=complex)

        correlation = correlate(received_signal, pilot, mode='valid')
        start_index = np.argmax(np.abs(correlation))
        aligned_segment = received_signal[start_index : start_index + len(pilot)]
        # norlmalize the aligned segment
        aligned_segment = aligned_segment / np.linalg.norm(aligned_segment)
        # Normalize the pilot signal
        pilot = pilot / np.linalg.norm(pilot)
        # Phase correction
        mean_phase_offset = np.angle(np.vdot(aligned_segment, pilot))
        aligned_segment = aligned_segment * np.exp(-1j * mean_phase_offset)

        if len(aligned_segment) < len(pilot):
            raise ValueError("Aligned segment is shorter than known signal. Check signal lengths or correlation results.")
        if len(aligned_segment) != len(pilot):
            raise ValueError("Aligned segment and pilot must have the same length.")

        rx_signal_power = np.log10(np.mean((np.abs(aligned_segment)**2)))
        
        # Ensure idsim and index_config are set before using them
        if self.idsim is None or self.index_config is None:
            raise RuntimeError("IDSIMULATION object or index_config not initialized. Call init_idsimulation first.")

        num_msg_bits = self.idsim.tag_bits_len
        codeword_len = self.idsim.make_idsimconfig(self.index_config)['codword_len_for_fec']
        code_rate_correction_factor = np.log10(num_msg_bits / codeword_len)
        code_rate_correction_factor = 0 # gives EsNo
        pilot_scaling_factor = 1 # since we are using normalized pilot signal does not need to scale the pilot signal
        rx_noise_power = np.log10(np.mean((np.abs((np.abs(aligned_segment) - np.abs(pilot_scaling_factor * pilot)))**2)))

        if rx_noise_power == 0:
            return np.inf, aligned_segment

        snr_db = 10 * (rx_signal_power - rx_noise_power - code_rate_correction_factor)
        return snr_db, aligned_segment

    def plot(self, y, title="Signal", xlabel="Sample Index", ylabel="Amplitude", show=True):
        """
        Plots the real part of a signal.

        Args:
            y (numpy.ndarray): The signal to plot.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            show (bool): Whether to display the plot immediately.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(np.real(y))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if show:
            plt.show()
        else:
            plt.close()

    def bpsk_demodulate_pipeline(self, complex_waveform):
        """
        Simulates a BPSK demodulation pipeline similar to GNU Radio blocks:
        - Mapping symbols to bits
        - Differential decoding
        - Packing bits into bytes

        Args:
            complex_waveform (numpy.ndarray): Numpy array of complex BPSK symbols.

        Returns:
            numpy.ndarray: A numpy array of bits representing the decoded bitstream.
        """
        # Step 1: Map complex symbols to bits (1+0j -> 1, -1+0j -> 0)
        raw_bits = np.where(np.real(complex_waveform) >= 0, 1, 0).astype(np.uint8)

        # Step 2: Differential decoding (XOR with previous bit)
        diff_decoded = np.zeros_like(raw_bits)
        prev = 0
        for i, bit in enumerate(raw_bits):
            diff_decoded[i] = bit ^ prev
            prev = bit

        # Step 3: Pack bits into bytes (simulate pack_k_bits_bb(8))
        # Pad with zeros if not multiple of 8
        pad_len = (8 - len(diff_decoded) % 8) % 8
        padded_bits = np.concatenate([diff_decoded, np.zeros(pad_len, dtype=np.uint8)])
        packed_bytes = np.packbits(padded_bits)
        packed_bits_unpacked = np.unpackbits(packed_bytes) # Unpack back to bits for consistent output
        return packed_bits_unpacked

    def BPSK_decoder_from_complex(self, complex_signal):
        """
        Decode a BPSK signal from complex values to bits.

        Args:
            complex_signal (numpy.ndarray): A numpy array of complex values representing the BPSK signal.

        Returns:
            numpy.ndarray: A numpy array of bits (0s and 1s).
        """
        complex_signal = np.asarray(complex_signal, dtype=np.complex64)
        bits = np.where(np.real(complex_signal) < 0, 0, 1).astype(np.uint8)
        return bits

    def rough_match_index(self,frame, preamble, threshold):
        frame = np.asarray(frame, dtype=np.uint8)
        preamble = np.asarray(preamble, dtype=np.uint8)
        L = len(preamble)
        min_dist = L + 1  # max possible + 1
        best_index = 0

        for i in range(len(frame) - L + 1):
            window = frame[i:i+L]
            distance = np.sum(window != preamble)
            if distance <= threshold and distance < min_dist:
                min_dist = distance
                best_index = i

        return best_index, min_dist

    def bitwisepreamble(self,preamble= [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1] * 2):
        """
        Convert a preamble list to a bitwise representation.
        
        Parameters:
        - preamble: List of integers (0s and 1s).
        
        Returns:
        - A numpy array of bits.
        """
        modified_preamble = np.zeros(len(preamble) * 2, dtype=np.uint8)
        modified_preamble[1::2] = preamble
        modified_preamble = np.unpackbits(modified_preamble)

        # Convert the preamble to a numpy array of uint8
        preamble_array = np.array(preamble, dtype=np.uint8)
        preamble_array = np.unpackbits(np.array(preamble, dtype=np.uint8))
        # Unpack bits
        return preamble_array

    def read_tx_csv_file(self, tx_csv_path):
        """
        Reads a CSV file and returns the first 16 columns of the first row as a NumPy array,
        preserving the original data types.

        Args:
            tx_csv_path (str): Path to the CSV file.

        Returns:
            numpy.ndarray: A NumPy array containing the data, or None if an error occurs.
        """
        try:
            df = pd.read_csv(tx_csv_path, header=None)
            df = df.iloc[0, :16]
            return df.to_numpy()
        except Exception as e:
            print(f"Error reading CSV file {tx_csv_path}: {e}")
            return None

    def process_complex_wave_file(self, rx_complex_wave_path, pilot, tx_encoded_16byte_data, tx_encoded_tag_data,tx_encoded_tag_pos, radius=7, threshold_ratio=0.8, packet_length=1024):
        """
        Process a complex wave file to extract packets, calculate SNR, BER, and Type I acceptance error.

        Args:
            rx_complex_wave_path (str): Path to the complex wave file.
            pilot (numpy.ndarray): The ideal known signal (complex64 datatype).
            tx_encoded_16byte_data (numpy.ndarray): 16-byte (128-bit) transmitted data (uint8).
            tx_encoded_tag_data (numpy.ndarray): Transmitted tag data.
            radius (int): Hamming distance threshold for acceptance.
            threshold_ratio (float): Ratio to determine the threshold for packet detection.
            packet_length (int): Length of each packet in bits.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Array of extracted raw 16-byte data packets.
                - list: List of decoded bit arrays per frame (FEC decoded).
                - dict: Dictionary of aggregated results (avg_snr, avg_ber, etc.).
        """
        rx_complex_wave = self.read_complex_file(rx_complex_wave_path)
        if rx_complex_wave is None:
            print(f"Failed to read complex wave file: {rx_complex_wave_path}")
            return np.array([]), [], {}
        # need to handel empty rx wave files 

        pkt_start_idx = self.find_packet_starts_ind(rx_complex_wave, pilot, threshold_ratio=threshold_ratio, packet_length=packet_length)
        
        # Initialize arrays with appropriate sizes
        num_packets = len(pkt_start_idx) - 1
        if num_packets <= 0:
            print(f"No valid packets found in {rx_complex_wave_path}")
            return np.array([]), [], {}

        snr_results = np.zeros(num_packets, dtype=np.float32)
        ber_results = np.zeros(num_packets, dtype=np.float32)
        false_positive_errors_results = np.zeros(num_packets, dtype=np.uint8)
        false_negative_errors_results = np.zeros(num_packets, dtype=np.uint8)
        data_16 = np.zeros((num_packets, 16), dtype=np.uint8)
        fec_decoded_frame_bits_out = []

        accept_count = 0
        total_count = 0
        count_correct_tags_received = 0
        
        tx_encoded_16byte_data_bits = np.unpackbits(np.array(tx_encoded_16byte_data, dtype=np.uint8))
        tx_encoded_tag_data = tx_encoded_tag_data.astype(self.idsim.tag_np_dtype)
        tx_encoded_tag_pos = tx_encoded_tag_pos[0]
        tx_encoded_tag_pos_in = tx_encoded_tag_pos[1]
        

        for i in range(num_packets):
            # i = num_packets -20 # DEBUG
            # i = 15939 # DEBUG
            frame = rx_complex_wave[pkt_start_idx[i]:pkt_start_idx[i+1]]
            
            # Ensure frame has enough data for processing
            if len(frame) == 0:
                print(f"Empty frame at index {i}. Skipping.")
                continue

            snr_db, aligned_segment = self.calculate_snr(pilot, frame)
            snr_results[i] = snr_db
            
            bits = self.bpsk_demodulate_pipeline(frame)
            preamble_array = self.bitwisepreamble()
            start_idx, distance = self.rough_match_index(bits, preamble_array, threshold=10)
            
            # Ensure start_idx is valid before proceeding
            if start_idx == -1:
                print(f"Preamble not found in frame {i}. Skipping.")
                continue

            # 32 bytes of padding after preamble
            start_data_idx = start_idx + len(preamble_array) + 32 * 8
            
            # Ensure start_data_idx is within bounds
            if start_data_idx >= len(frame):
                print(f"Data start index out of bounds for frame {i}. Skipping.")
                continue

            data_bit_len = self.idsim.make_idsimconfig(self.index_config)['codword_len_for_fec']
            
            # Ensure frame has enough data for frame_data extraction
            if start_data_idx + data_bit_len > len(frame):
                print(f"Not enough data for frame_data extraction in frame {i}. Skipping.")
                continue
            
            frame_data = frame[start_data_idx:start_data_idx + data_bit_len]

            if len(bits[start_idx:]) >= packet_length:
                frame_bits = bits[start_idx:start_idx + packet_length]
            else:
                padding_length = packet_length - len(bits[start_idx:])
                padding = np.zeros(padding_length, dtype=np.uint8)
                frame_bits = np.concatenate((bits[start_idx:], padding))

            data = np.packbits(frame_bits)
            # Ensure data has enough elements before slicing
            if len(data) < 74: # 58:74 implies at least 74 elements needed
                print(f"Not enough packed data for slicing in frame {i}. Skipping.")
                continue

            data_extracted = data[58:74]  # Extract 16 bytes (128 bits)
            rx_bits = np.unpackbits(data_extracted)
            
            # Ensure rx_bits has the expected length for run_my_decoder
            expected_rx_bits_len = self.code_length # Assuming code_length is 128
            if len(rx_bits) != expected_rx_bits_len:
                print(f"Rx bits length mismatch for frame {i}. Expected {expected_rx_bits_len}, got {len(rx_bits)}. Skipping.")
                continue

            # LLR conversion (though 'y' is passed directly to run_my_decoder)
            # llr = [1.0 if bit == 0 else -1.0 for bit in rx_bits] # Not directly used in the call below

            bit_errors, decoded_tag, estimated_tag_x, bits_hat_demod, false_positive_errors, false_negative_errors = \
                self.idsim.run_my_decoder(
                    encoded_arr_x=tx_encoded_tag_data,
                    tag_pos=tx_encoded_tag_pos,
                    tag_pos_in=tx_encoded_tag_pos_in,
                    y=frame_data * -1, # Assuming frame_data is complex and needs to be scaled
                    bits_hat_demod=rx_bits,
                    no=self.idsim.ebnodb2no(20, self.idsim.num_bits_per_symbol, self.idsim.coderate_fec)
                )
            
            # Ensure decoded_tag is a numpy array for consistent comparison
            if not isinstance(decoded_tag, np.ndarray):
                decoded_tag = np.array(decoded_tag)

            # Count how many decoded tags are correct
            if self.idsim.encoder_str == "EHRSDID":
                count_correct_tags_received += np.all((data_extracted == tx_encoded_tag_data))
            else:
                # Reshape tx_encoded_tag_data[0] to match decoded_tag's shape if necessary
                if tx_encoded_tag_data.ndim > 1:
                    target_tag = tx_encoded_tag_data[0].reshape(-1,)
                else:
                    target_tag = tx_encoded_tag_data
                count_correct_tags_received += np.all((decoded_tag == target_tag))

            ber_results[i] = bit_errors
            false_positive_errors_results[i] = false_positive_errors
            false_negative_errors_results[i] = false_negative_errors

            hamming_distance = np.count_nonzero(tx_encoded_16byte_data_bits != rx_bits)
            total_count += 1
            if hamming_distance <= radius:
                accept_count += 1

            # FEC decoding
            fec_decoded_frame_bits_out.append(decoded_tag) # Appending the decoded_tag from idsim
            data_16[i] = data_extracted

        total_processed_received_tags = num_packets # Adjusted to reflect actual processed packets
        ratio_correct_received_tags = count_correct_tags_received / total_processed_received_tags if total_processed_received_tags > 0 else 0.0
        print(f"Total processed received tags: {total_processed_received_tags}, Correctly received tags: {count_correct_tags_received}, Ratio %age: {ratio_correct_received_tags*100:.2f}%")
        
        total_fp_errors_scalar = np.sum(false_positive_errors_results)
        ratio_false_positive_tags = total_fp_errors_scalar / total_processed_received_tags if total_processed_received_tags > 0 else 0.0
        print(f"Total processed received tags: {total_processed_received_tags}, False Positive received: {total_fp_errors_scalar}, Ratio %age: {ratio_false_positive_tags*100:.2f}%")
        
        total_fn_errors_scalar = np.sum(false_negative_errors_results)
        ratio_false_negative_tags = total_fn_errors_scalar / total_processed_received_tags if total_processed_received_tags > 0 else 0.0
        print(f"Total processed received tags: {total_processed_received_tags}, False Negative received: {total_fn_errors_scalar}, Ratio %age: {ratio_false_negative_tags*100:.2f}%")

        avg_snr = np.sum(snr_results)/ total_processed_received_tags
        print(f"Average SNR: {avg_snr:.2f} dB")
        avg_ber = np.mean(ber_results)
        print(f"Average BER: {avg_ber:.4f}")
        avg_false_positive_errors = np.mean(false_positive_errors_results)
        print(f"Average False Positive Errors: {avg_false_positive_errors:.2f}")
        avg_false_negative_errors = np.mean(false_negative_errors_results)
        print(f"Average False Negative Errors: {avg_false_negative_errors:.2f}")

        dict_results = {
            'avg_snr': avg_snr,
            'avg_ber': avg_ber,
            'total_processed_received_tags': total_processed_received_tags,
            'num_false_positive_errors_results': total_fp_errors_scalar,
            'num_false_negative_errors_results': total_fn_errors_scalar,
            'count_correct_tags_received': count_correct_tags_received,
            'ratio_correct_received_tags': ratio_correct_received_tags,
            'avg_false_positive_errors': avg_false_positive_errors,
            'avg_false_negative_errors': avg_false_negative_errors,
        }

        return data_16, fec_decoded_frame_bits_out, dict_results

    def process_all_complex_wave_files(self, rx_complex_wave_dir, pilot, radius=7, threshold_ratio=0.8, packet_length=1024):
        """
        Process all complex wave files in the specified directory and save results to CSV files.

        Args:
            rx_complex_wave_dir (str): Path to the directory containing complex wave files.
            pilot (numpy.ndarray): The ideal known signal (complex64 datatype).
            radius (int): Hamming distance threshold for acceptance.
            threshold_ratio (float): Ratio to determine the threshold for packet detection.
            packet_length (int): Length of each packet in samples.
        """
        # Load index_config and initialize idsim for the current folder
        config_file_path = os.path.join((rx_complex_wave_dir), "config.pkl")
        try:
            self.index_config = self.load_num_from_pkl(pkl_path=config_file_path)
            self.idsim = IDSIMULATION(index_config=self.index_config)
        except FileNotFoundError:
            print(f"Config file not found at {config_file_path}. Skipping folder: {rx_complex_wave_dir}")
            return
        except Exception as e:
            print(f"Error loading config or initializing IDSIMULATION for {rx_complex_wave_dir}: {e}. Skipping folder.")
            return

        complex_files = [f for f in os.listdir(rx_complex_wave_dir) if f.startswith('rxwave_tg')]
        
        # Extract gain value from the folder name (e.g., 'rsidrxgain30' -> '30')
        gain_match = re.search(r'rsidrxgain(\d+)', os.path.basename(os.path.dirname(rx_complex_wave_dir)))
        if not gain_match:
            print(f"Could not extract gain value from folder path: {rx_complex_wave_dir}. Skipping all files in this folder.")
            return
        gain_val = gain_match.group(1) # RX gain value
        for file in complex_files:
            file_path = os.path.join(rx_complex_wave_dir, file)
            # Extract gain value from the file name (e.g., 'rxwave_tg30' -> '30')
            match = re.search(r'tg(\d+)', file)
            # Extract the number if found
            if match:
                tx_gain_value = int(match.group(1))
                # print(gain_value)
            else:
                print("No match found.")
            # Construct CSV paths using the extracted gain value
            tx_encoded_16byte_data_path = os.path.join(os.path.dirname(rx_complex_wave_dir), 'csv', f'tx_encoded_data_16_8_tg{tx_gain_value}.csv')
            tx_encoded_tag_data_path = os.path.join(os.path.dirname(rx_complex_wave_dir), 'csv', f'tx_tag_data_tg{tx_gain_value}.csv')
            tx_encoded_tag_pos_path = os.path.join(os.path.dirname(rx_complex_wave_dir), 'csv', f'tx_tag_pos_tg{tx_gain_value}.csv')

            if not os.path.exists(tx_encoded_16byte_data_path):
                print(f"Warning: Transmit encoded 16-byte data CSV not found at {tx_encoded_16byte_data_path}. Skipping file {file}.")
                continue
            if not os.path.exists(tx_encoded_tag_data_path):
                print(f"Warning: Transmit encoded tag data CSV not found at {tx_encoded_tag_data_path}. Skipping file {file}.")
                continue
            if not os.path.exists(tx_encoded_tag_pos_path):
                print(f"Warning: Transmit encoded tag pos CSV not found at {tx_encoded_tag_pos_path}. Skipping file {file}.")
                continue

            tx_encoded_16byte_data = self.read_tx_csv_file(tx_encoded_16byte_data_path)
            tx_encoded_tag_data = self.read_tx_csv_file(tx_encoded_tag_data_path)
            tx_encoded_tag_pos = self.read_tx_csv_file(tx_encoded_tag_pos_path)
            
            if tx_encoded_16byte_data is None or tx_encoded_tag_data is None or tx_encoded_tag_pos is None:
                print(f"Skipping {file} due to issues reading TX CSV data.")
                continue

            data_16, fec_decoded_data, dict_results = self.process_complex_wave_file(
                file_path, pilot, tx_encoded_16byte_data, tx_encoded_tag_data,tx_encoded_tag_pos, radius, threshold_ratio, packet_length
            )
            
            # Save dict_results to a pickle file and txt file
            output_folder = os.path.join(os.path.dirname(rx_complex_wave_dir), 'output_folder')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            dict_path_to_save = os.path.join(output_folder, f"{file}_results_dict")
            write_dict(dict_path_to_save, dict_results)
            
            # Save data_16 to CSV
            if data_16.size > 0: # Check if data_16 is not empty
                data_16_df = pd.DataFrame(data_16, dtype=np.uint8)
                output_file = os.path.join(output_folder, f"{file}_results.csv")
                data_16_df.to_csv(output_file, index=False, header=False)
            else:
                print(f"No data_16 to save for {file}.")

            # Save FEC decoded data to CSV
            if fec_decoded_data and len(fec_decoded_data) > 0:
                # Ensure all sub-arrays in fec_decoded_data have the same length for DataFrame creation
                # Pad if necessary, or determine a common length
                max_len = max(len(arr) for arr in fec_decoded_data) if fec_decoded_data else 0
                padded_fec_data = [np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=0) for arr in fec_decoded_data]

                fec_decoded_df = pd.DataFrame(padded_fec_data, columns=[f'Bit {i+1}' for i in range(max_len)])
                fec_decoded_output_file = os.path.join(output_folder, f"{file}_fec_decoded_results.csv")
                fec_decoded_df.to_csv(fec_decoded_output_file, index=False, header=False)
            else:
                print(f"No FEC decoded data to save for {file}.")
                           
            print(f"Processed {file} and saved results to {output_folder}")


def worker(rx_wave_complex_wave_folder):  
    try:
        # print(f"\nProcessing folder: {rx_wave_complex_wave_folder}")
        # Instantiate the RxProcessor class
        processor = RxProcessor()
        processor.process_all_complex_wave_files(rx_wave_complex_wave_folder, processor.pilot, radius=7, threshold_ratio=0.8, packet_length=1024)
        return rx_wave_complex_wave_folder
    except Exception as e:
        return {"error": str(e), "config": rx_wave_complex_wave_folder}
    


# Main execution block
if __name__ == "__main__":
    
    # path_experiment_folder = "/home/praful/eco/idcodes/grc/paper/experiemnt_wo_attenuator"
    #     # path_experiment_folder = "/home/praful/eco/idcodes/grc/paper/experiment_with_30db_attenuator"
    #     # path_experiment_folder = "/home/praful/eco/idcodes/grc/paper/experiment_with_30dB_attenuator_lower_gain_range"

    rx_wave_complex_wave_folders = glob.glob("./experiemnt_wo_attenuator/paper_rx_files_*/rsidrxgain*/complex")
    # Get all folders in paper_rx_files that match the pattern rsidrxgain*
    # folder_idx = 'experiemnt_wo_attenuator/paper_rx_files_0'
    # rx_wave_complex_wave_folders = glob.glob(os.path.join('.', folder_idx, 'rsidrxgain*', 'complex'))
    #DEBUG
    # rx_wave_complex_wave_folders = [
    #                                 './paper_rx_files_0/rsidrxgain15/complex',
    #                                  './paper_rx_files_0/rsidrxgain20/complex',
    #                                 './paper_rx_files_0/rsidrxgain25/complex',
    #                                 './paper_rx_files_0/rsidrxgain10/complex',
    #                                 './paper_rx_files_0/rsidrxgain5/complex'
    #                                 ]
    # rx_wave_complex_wave_folders = [
    # '././experiemnt_wo_attenuator/paper_rx_files_1/rsidrxgain31/complex']#,
    # '././experiemnt_wo_attenuator/paper_rx_files_1/rsidrxgain21/complex',
    # '././experiemnt_wo_attenuator/paper_rx_files_1/rsidrxgain11/complex',
    # '././experiemnt_wo_attenuator/paper_rx_files_1/rsidrxgain16/complex',
    # '././experiemnt_wo_attenuator/paper_rx_files_1/rsidrxgain6/complex',
    # '././experiemnt_wo_attenuator/paper_rx_files_1/rsidrxgain1/complex',
    # '././experiemnt_wo_attenuator/paper_rx_files_1/rsidrxgain26/complex'
    # ]
    # #DEBUG
    # rx_wave_complex_wave_folders = [
    #                                 # './paper_rx_files_0/rsidrxgain15/complex',
    #                                 # './paper_rx_files_0/rsidrxgain20/complex',
    #                                 './paper_rx_files_0/rsidrxgain25/complex',
    #                                 # './paper_rx_files_0/rsidrxgain10/complex',
    #                                 # './paper_rx_files_0/rsidrxgain5/complex'
    #                                 ]
    rx_wave_complex_wave_folders = [folder for folder in rx_wave_complex_wave_folders if 'rxgain30' not in folder]
    rx_wave_complex_wave_folders = [folder for folder in rx_wave_complex_wave_folders if 'rxgain90' not in folder]
    # folder_idx = 'paper_rx_files_3'
    # rx_wave_complex_wave_folders = [
    #                                 # './paper_rx_files_3/rsidrxgain15/complex',
    #                                 # './paper_rx_files_3/rsidrxgain20/complex',
    #                                 './paper_rx_files_3/rsidrxgain25/complex',
    #                                 # './paper_rx_files_3/rsidrxgain10/complex',
    #                                 # './paper_rx_files_3/rsidrxgain5/complex'
    #                                 ]
    # Loop through each folder and process the complex wave files


    num_processors = 12
    with multiprocessing.Pool(processes=num_processors) as pool:
        with tqdm(total=len(rx_wave_complex_wave_folders)) as pbar:
            results = []

            def update_progress(result):
                results.append(result)
                pbar.update(1)

            for rx_wave_complex_wave_folders in rx_wave_complex_wave_folders:
                pool.apply_async(worker, args=(rx_wave_complex_wave_folders,), callback=update_progress)

            pool.close()
            pool.join()  # Wait for all processes to complete


