#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/utils/thread.hpp>

#include <iostream>
#include <vector>
#include <complex>
#include <thread>
#include <chrono>

#include "barker.hpp"

int main() {
    // Lock this thread to CPU core for consistent timing
    uhd::set_thread_priority_safe();

    // Create USRP device (X410)
    std::string args = "addr=192.168.0.112";
    auto usrp = uhd::usrp::multi_usrp::make(args);

    // Select TX subdevice/channel
    usrp->set_tx_subdev_spec(uhd::usrp::subdev_spec_t("A:0"));


    // TX settings
    double tx_rate = 10e6;
    double tx_freq = 2.45e9;
    double tx_gain = 20.0;
    size_t tx_channel = 0;

    usrp->set_tx_rate(tx_rate, tx_channel);
    usrp->set_tx_gain(tx_gain, tx_channel);
    usrp->set_tx_freq(uhd::tune_request_t(tx_freq), tx_channel);

    std::cout << "[TX] Configured with:" << std::endl;
    std::cout << "  Rate: " << usrp->get_tx_rate(tx_channel) << " Sps" << std::endl;
    std::cout << "  Freq: " << usrp->get_tx_freq(tx_channel) / 1e6 << " MHz" << std::endl;
    std::cout << "  Gain: " << usrp->get_tx_gain(tx_channel) << " dB" << std::endl;

    // Prepare Barker sequence (modulated)
    auto barker = generate_barker_bpsk();

    // TX streamer setup
    uhd::stream_args_t stream_args("fc32"); // complex float32
    stream_args.channels = {tx_channel};
    auto tx_streamer = usrp->get_tx_stream(stream_args);
    uhd::tx_metadata_t md;

    md.start_of_burst = true;
    md.end_of_burst = false;
    md.has_time_spec = false;

    // Transmit repeatedly
    std::cout << "Transmitting Barker sequence 100 times..." << std::endl;

    for (int i = 0; i < 100; i++) {
        size_t samples_sent = tx_streamer->send(&barker.front(), barker.size(), md);
        if (samples_sent != barker.size()) {
            std::cerr << "Warning: Sent " << samples_sent << " samples instead of " << barker.size() << std::endl;
        }

        md.start_of_burst = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Send end-of-burst
    md.end_of_burst = true;
    tx_streamer->send("", 0, md);

    std::cout << "Transmission complete." << std::endl;
    return 0;
}
