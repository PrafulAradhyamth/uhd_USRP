#include "waveform_plotter.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    std::string filepath = "rx_waveform.bin";  // or pass via argv
    WaveformPlotter plotter(filepath);
    plotter.show();

    return app.exec();
}
