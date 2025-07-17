#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/stream.hpp>
#include <uhd/utils/thread.hpp>

#include <iostream>
#include <complex>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <fstream>

#include "barker.hpp"
#include "utils.hpp"

float correlate_and_estimate(const std::vector<std::complex<float>>& rx,
                             const std::vector<std::complex<float>>& barker) {
    int peak_idx = 0;
    float peak_val = 0;

    for (size_t i = 0; i <= rx.size() - barker.size(); i++) {
        std::complex<float> sum = 0.0f;
        for (size_t j = 0; j < barker.size(); j++) {
            sum += rx[i + j] * std::conj(barker[j]);
        }
        float mag = std::abs(sum);
        if (mag > peak_val) {
            peak_val = mag;
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



// Save received samples to binary file
void save_waveform(const std::vector<std::complex<float>>& samples, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<const char*>(samples.data()), samples.size() * sizeof(std::complex<float>));
    outfile.close();
    std::cout << "Saved " << samples.size() << " samples to " << filename << std::endl;
}


int main() {
    uhd::set_thread_priority_safe();

    auto usrp = uhd::usrp::multi_usrp::make("addr=192.168.0.112");

    size_t chan = 0;
    usrp->set_tx_subdev_spec(uhd::usrp::subdev_spec_t("A:0"));
    usrp->set_rx_subdev_spec(uhd::usrp::subdev_spec_t("A:0"));

    double rate = 10e6;
    double freq = 2.45e9;
    double tx_gain = 20;
    double rx_gain = 20;

    usrp->set_tx_rate(rate);
    usrp->set_rx_rate(rate);
    usrp->set_tx_freq(freq);
    usrp->set_rx_freq(freq);
    usrp->set_tx_gain(tx_gain);
    usrp->set_rx_gain(rx_gain);

    std::cout << "TX/RX rate: " << rate << " Hz | Freq: " << freq / 1e6 << " MHz" << std::endl;

    auto barker = generate_barker_bpsk();
    std::vector<std::complex<float>> tx_buffer;
    for (int i = 0; i < 1000; i++) {
        tx_buffer.insert(tx_buffer.end(), barker.begin(), barker.end());
    }

    // Set up RX streamer
    uhd::stream_args_t rx_args("fc32");
    rx_args.channels = {chan};
    auto rx_streamer = usrp->get_rx_stream(rx_args);
    uhd::rx_metadata_t rx_md;

    std::vector<std::complex<float>> rx_buffer(4096 * 10);

    // Issue stream command before TX
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.num_samps = rx_buffer.size();
    stream_cmd.stream_now = true;
    stream_cmd.time_spec = uhd::time_spec_t();
    rx_streamer->issue_stream_cmd(stream_cmd);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // settle before TX

    // TX setup
    uhd::stream_args_t tx_args("fc32");
    tx_args.channels = {chan};
    auto tx_streamer = usrp->get_tx_stream(tx_args);
    uhd::tx_metadata_t tx_md;
    tx_md.start_of_burst = true;
    tx_md.end_of_burst = false;
    tx_md.has_time_spec = false;

    // Send TX data
    size_t sent = tx_streamer->send(&tx_buffer.front(), tx_buffer.size(), tx_md);
    tx_md.start_of_burst = false;
    tx_md.end_of_burst = true;
    tx_streamer->send("", 0, tx_md);

    // Receive data
    size_t received = rx_streamer->recv(&rx_buffer.front(), rx_buffer.size(), rx_md, 3.0);
    rx_buffer.resize(received);
    save_waveform(rx_buffer, "rx_waveform.bin");


    std::cout << "TX sent: " << sent << " | RX received: " << received << " samples" << std::endl;

    float snr_db = correlate_and_estimate(rx_buffer, barker);
    std::cout << "Estimated SNR: " << snr_db << " dB\n";

    return 0;
}
