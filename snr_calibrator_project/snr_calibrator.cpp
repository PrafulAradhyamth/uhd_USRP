#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/stream.hpp>
#include <uhd/types/tune_request.hpp>
#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <algorithm>
#include <fftw3.h>

using namespace std::chrono_literals;

// Parameters
const double sample_rate = 32e3;
const double center_freq = 915e6;
const size_t num_samples = 2048;
const std::vector<double> tx_gains = {0, 5, 10, 15, 20};

// ---------- DSP UTILITIES ----------

// Generate simple BPSK waveform
std::vector<std::complex<float>> generate_bpsk_waveform(size_t len) {
    std::vector<std::complex<float>> waveform(len);
    for (size_t i = 0; i < len; ++i) {
        float val = (i % 2 == 0) ? 1.0f : -1.0f;
        waveform[i] = std::complex<float>(val, 0.0f);
    }
    return waveform;
}

// Compute average power
double compute_power(const std::vector<std::complex<float>>& vec) {
    double power = 0.0;
    for (const auto& s : vec) power += std::norm(s);
    return power / vec.size();
}

// 1. Power Ratio Method
double compute_power_ratio_snr(const std::vector<std::complex<float>>& signal,
                               const std::vector<std::complex<float>>& noise) {
    double signal_power = compute_power(signal);
    double noise_power = compute_power(noise);
    return 10.0 * std::log10(signal_power / noise_power);
}

// 2. FFT-based SNR Method
double compute_fft_snr(const std::vector<std::complex<float>>& rx_signal) {
    int N = rx_signal.size();
    std::vector<double> power_spectrum(N);

    fftw_complex* in = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
    fftw_complex* out = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i = 0; i < N; i++) {
        in[i][0] = rx_signal[i].real();
        in[i][1] = rx_signal[i].imag();
    }

    fftw_execute(plan);

    for (int i = 0; i < N; i++) {
        double re = out[i][0];
        double im = out[i][1];
        power_spectrum[i] = (re * re + im * im);
    }

    int band = N / 8;
    double signal_power = 0.0, noise_power = 0.0;
    for (int i = 0; i < N; i++) {
        if (i >= N/2 - band && i <= N/2 + band) signal_power += power_spectrum[i];
        else noise_power += power_spectrum[i];
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return 10.0 * std::log10(signal_power / noise_power);
}

// 3. Correlation-based SNR Method
double compute_correlation_snr(const std::vector<std::complex<float>>& rx_signal,
                               const std::vector<std::complex<float>>& known_signal) {
    int N = rx_signal.size();
    int M = known_signal.size();
    std::vector<double> corr(N - M + 1);

    for (int i = 0; i <= N - M; ++i) {
        std::complex<double> acc(0, 0);
        for (int j = 0; j < M; ++j) {
            acc += std::conj(known_signal[j]) * rx_signal[i + j];
        }
        corr[i] = std::norm(acc);
    }

    double peak = *std::max_element(corr.begin(), corr.end());
    double sidelobe_sum = 0.0;
    for (double v : corr) sidelobe_sum += v;
    sidelobe_sum -= peak;
    double noise_power = sidelobe_sum / (corr.size() - 1);

    return 10.0 * std::log10(peak / noise_power);
}

// ---------- MAIN PROGRAM ----------

int main() {
    auto usrp = uhd::usrp::multi_usrp::make("addr=192.168.0.112");

    usrp->set_rx_rate(sample_rate);
    usrp->set_tx_rate(sample_rate);
    usrp->set_rx_freq(center_freq);
    usrp->set_tx_freq(center_freq);

    usrp->set_rx_gain(10);
    usrp->set_tx_antenna("TX/RX");
    usrp->set_rx_antenna("RX2");

    std::vector<std::complex<float>> waveform = generate_bpsk_waveform(num_samples);

    // Setup streams
    uhd::stream_args_t stream_args("fc32");
    auto rx_stream = usrp->get_rx_stream(stream_args);
    auto tx_stream = usrp->get_tx_stream(stream_args);

    std::ofstream outfile("snr_calibration.csv");
    outfile << "TX_Gain_dB,SNR_Power_dB,SNR_FFT_dB,SNR_Corr_dB\n";

    for (double tx_gain : tx_gains) {
        std::cout << "\nTesting TX Gain: " << tx_gain << " dB" << std::endl;
        usrp->set_tx_gain(tx_gain);

        // Transmit waveform
        uhd::tx_metadata_t md;
        md.start_of_burst = true;
        md.end_of_burst = true;
        tx_stream->send(&waveform.front(), waveform.size(), md);

        std::this_thread::sleep_for(300ms);

        // Receive signal
        std::vector<std::complex<float>> signal_buf(num_samples);
        uhd::rx_metadata_t rx_md;
        rx_stream->recv(&signal_buf.front(), signal_buf.size(), rx_md);

        // Transmit zeros (for noise)
        std::vector<std::complex<float>> zeros(num_samples, {0.0f, 0.0f});
        tx_stream->send(&zeros.front(), zeros.size(), md);
        std::this_thread::sleep_for(300ms);

        // Receive noise
        std::vector<std::complex<float>> noise_buf(num_samples);
        rx_stream->recv(&noise_buf.front(), noise_buf.size(), rx_md);

        // Compute SNRs
        double snr_power = compute_power_ratio_snr(signal_buf, noise_buf);
        double snr_fft   = compute_fft_snr(signal_buf);
        double snr_corr  = compute_correlation_snr(signal_buf, waveform);

        // Log results
        outfile << tx_gain << "," << snr_power << "," << snr_fft << "," << snr_corr << "\n";
        std::cout << "SNR Power: " << snr_power << " dB, "
                  << "FFT: " << snr_fft << " dB, "
                  << "Corr: " << snr_corr << " dB" << std::endl;
    }

    outfile.close();
    std::cout << "\n Calibration complete. Results saved to snr_calibration.csv\n";
    return 0;
}
