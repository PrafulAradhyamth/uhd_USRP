#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/thread_priority.hpp>
#include <iostream>

int main() {
    // Set high priority (optional, improves performance)
    if (!uhd::set_thread_priority_safe()) {
        std::cerr << "Warning: Failed to set thread priority." << std::endl;
    }

    try {
        // Discover devices
        uhd::device_addr_t hint;
        uhd::device_addrs_t dev_addrs = uhd::device::find(hint);

        if (dev_addrs.empty()) {
            std::cout << "No USRP found." << std::endl;
            return 0;
        }

        for (size_t i = 0; i < dev_addrs.size(); ++i) {
            std::cout << "USRP Device " << i << " Found:" << std::endl;

            auto serial = dev_addrs[i].has_key("serial") ? dev_addrs[i]["serial"] : "N/A";
            auto addr   = dev_addrs[i].has_key("addr") ? dev_addrs[i]["addr"] : "N/A";

            std::cout << "  Serial Number: " << serial << std::endl;
            std::cout << "  IP Address   : " << addr << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
// Compile with: g++ -std=c++17 testuhd_ex1.cpp -o testuhd_ex1 -luhd