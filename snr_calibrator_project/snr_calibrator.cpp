#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/thread.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/stream.hpp>
#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <fftw3.h>
#include <numeric>


// ========================== SIGNAL GENERATION ===========================

std::vector<std::complex<float>> generate_barker_bpsk() {
    int barker[] = {1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1};
    std::vector<std::complex<float>> waveform;
    for (int i = 0; i < 1000; ++i) {
        for (int b : barker)
            waveform.emplace_back((float)b, 0.0f);
    }
    return waveform;
}

// ========================== DSP UTILITIES ===========================

double compute_power(const std::vector<std::complex<float>>& vec) {
    double power = 0.0;
    for (const auto& v : vec) power += std::norm(v);
    return power / vec.size();
}

double compute_power_snr(const std::vector<std::complex<float>>& signal,
                         const std::vector<std::complex<float>>& noise) {
    return 10.0 * std::log10(compute_power(signal) / compute_power(noise));
}

double compute_fft_snr(const std::vector<std::complex<float>>& signal) {
    size_t N = signal.size();
    fftw_complex* in = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
    fftw_complex* out = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (size_t i = 0; i < N; ++i) {
        in[i][0] = signal[i].real();
        in[i][1] = signal[i].imag();
    }

    fftw_execute(plan);

    std::vector<double> spectrum(N);
    for (size_t i = 0; i < N; ++i)
        spectrum[i] = std::pow(out[i][0], 2) + std::pow(out[i][1], 2);

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    int band = N / 8;
    double signal_power = 0, noise_power = 0;
    for (size_t i = 0; i < N; ++i) {
        if (i > N/2 - band && i < N/2 + band)
            signal_power += spectrum[i];
        else
            noise_power += spectrum[i];
    }
    return 10.0 * std::log10(signal_power / noise_power);
}

double compute_correlation_snr(const std::vector<std::complex<float>>& rx,
                               const std::vector<std::complex<float>>& tx) {
    int N = rx.size(), M = tx.size();
    std::vector<double> corr(N - M + 1);
    for (int i = 0; i <= N - M; ++i) {
        std::complex<float> sum(0, 0);
        for (int j = 0; j < M; ++j)
            sum += std::conj(tx[j]) * rx[i + j];
        corr[i] = std::norm(sum);
    }

    double peak = *std::max_element(corr.begin(), corr.end());
    double avg_noise = (std::accumulate(corr.begin(), corr.end(), 0.0) - peak) / (corr.size() - 1);
    return 10.0 * std::log10(peak / avg_noise);
}

// ========================== SAVE FUNCTION ===========================

void save_waveform(const std::vector<std::complex<float>>& buffer, const std::string& fname) {
    std::ofstream f(fname, std::ios::binary);
    f.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(std::complex<float>));
}

// ========================== MAIN ===========================

int main() {
    uhd::set_thread_priority_safe();
    auto usrp = uhd::usrp::multi_usrp::make("addr=192.168.0.112");

    size_t chan = 0;
    double rate = 10e6;
    double freq = 2.45e9;
    double tx_gain = 20;
    double rx_gain = 20;

    usrp->set_tx_subdev_spec(uhd::usrp::subdev_spec_t("A:0"));
    usrp->set_rx_subdev_spec(uhd::usrp::subdev_spec_t("A:0"));

    usrp->set_tx_rate(rate);
    usrp->set_rx_rate(rate);
    usrp->set_tx_freq(freq);
    usrp->set_rx_freq(freq);
    usrp->set_tx_gain(tx_gain);
    usrp->set_rx_gain(rx_gain);

    std::cout << "USRP Configured: " << freq / 1e6 << " MHz @ " << rate / 1e6 << " Msps\n";

    // Generate waveform
    auto tx_waveform = generate_barker_bpsk();
    std::vector<std::complex<float>> rx_buffer(4096 * 10);

    // Prepare TX
    uhd::stream_args_t tx_args("fc32"); tx_args.channels = {chan};
    auto tx_stream = usrp->get_tx_stream(tx_args);
    uhd::tx_metadata_t tx_md;
    tx_md.start_of_burst = true;
    tx_md.end_of_burst = false;
    tx_md.has_time_spec = false;

    // Prepare RX
    uhd::stream_args_t rx_args("fc32"); rx_args.channels = {chan};
    auto rx_stream = usrp->get_rx_stream(rx_args);
    uhd::rx_metadata_t rx_md;
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.num_samps = rx_buffer.size();
    stream_cmd.stream_now = true;
    rx_stream->issue_stream_cmd(stream_cmd);

    // Wait for RX pipeline to warm up
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Transmit waveform
    size_t tx_sent = tx_stream->send(&tx_waveform.front(), tx_waveform.size(), tx_md);
    tx_md.start_of_burst = false;
    tx_md.end_of_burst = true;
    tx_stream->send("", 0, tx_md);

    // Receive
    size_t rx_received = rx_stream->recv(&rx_buffer.front(), rx_buffer.size(), rx_md, 3.0);
    rx_buffer.resize(rx_received);
    save_waveform(rx_buffer, "rx_waveform.bin");

    std::cout << "TX sent: " << tx_sent << " | RX received: " << rx_received << " samples\n";

    // Dummy noise for method 1 — in real app, measure noise separately
    std::vector<std::complex<float>> dummy_noise(rx_received, {0.01f, 0.01f});

    // SNR Estimations
    double snr_power = compute_power_snr(rx_buffer, dummy_noise);
    double snr_fft   = compute_fft_snr(rx_buffer);
    double snr_corr  = compute_correlation_snr(rx_buffer, tx_waveform);

    std::cout << "\n=== Estimated SNRs ===\n";
    std::cout << "Power Ratio Method : " << snr_power << " dB\n";
    std::cout << "FFT-Based Method   : " << snr_fft   << " dB\n";
    std::cout << "Correlation Method : " << snr_corr  << " dB\n";

    std::ofstream fout("snr_log.csv");
    fout << "SNR_Power,SNR_FFT,SNR_Corr\n";
    fout << snr_power << "," << snr_fft << "," << snr_corr << "\n";
    fout.close();

    return 0;
}
