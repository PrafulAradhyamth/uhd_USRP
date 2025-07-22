#include "waveform_plotter.h"
#include <fstream>

WaveformPlotter::WaveformPlotter(const std::string& filepath, QWidget *parent)
    : QMainWindow(parent)
{
    resize(800, 600);
    loadData(filepath);
    plot();
}

void WaveformPlotter::loadData(const std::string &filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        QMessageBox::critical(this, "Error", "Failed to open .bin file!");
        return;
    }

    std::complex<float> sample;
    while (infile.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
        data.push_back(sample);
    }
}

void WaveformPlotter::plot() {
    QLineSeries *series = new QLineSeries();

    for (size_t i = 0; i < data.size(); ++i) {
        float magnitude = std::abs(data[i]);
        series->append(static_cast<qreal>(i), static_cast<qreal>(magnitude));
    }

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Amplitude of RX Waveform (.fc32)");
    chart->createDefaultAxes();
    chart->axes(Qt::Horizontal).first()->setTitleText("Sample Index");
    chart->axes(Qt::Vertical).first()->setTitleText("Amplitude");

    chartView = new QChartView(chart);
    setCentralWidget(chartView);
}
