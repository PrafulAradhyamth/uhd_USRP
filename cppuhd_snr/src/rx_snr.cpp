#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/stream.hpp>
#include <iostream>
#include "barker.hpp"
#include "utils.hpp"

float correlate_and_estimate(const std::vector<std::complex<float>>& rx,
                             const std::vector<std::complex<float>>& barker) {
    int peak_idx = 0;
    float peak_val = 0;

    for (size_t i = 0; i < rx.size() - barker.size(); i++) {
        std::complex<float> sum = 0.0f;
        for (size_t j = 0; j < barker.size(); j++) {
            sum += rx[i + j] * std::conj(barker[j]);
        }
        if (std::abs(sum) > peak_val) {
            peak_val = std::abs(sum);
            peak_idx = i;
        }
    }

    std::vector<std::complex<float>> signal(rx.begin() + peak_idx,
                                            rx.begin() + peak_idx + barker.size());
    std::vector<std::complex<float>> noise(rx.begin(), rx.begin() + barker.size());

    float signal_power = compute_power(signal);
    float noise_power = compute_power(noise);

    return compute_snr_db(signal_power, noise_power);
}

int main() {
    auto usrp = uhd::usrp::multi_usrp::make("type=x4xx,addr=192.168.0.112");
    usrp->set_rx_rate(1e6);
    usrp->set_rx_freq(2.45e9);
    usrp->set_rx_gain(20);

    auto rx_streamer = usrp->get_rx_stream(uhd::stream_args_t("fc32"));

    std::vector<std::complex<float>> buffer(1024 * 10);
    uhd::rx_metadata_t md;
    rx_streamer->recv(&buffer.front(), buffer.size(), md, 3.0);

    std::vector<std::complex<float>> barker = generate_barker_bpsk();
    float snr_db = correlate_and_estimate(buffer, barker);
    std::cout << "Estimated SNR: " << snr_db << " dB\n";

    return 0;
}
