#pragma once

#include <QtWidgets>
#include <QtCharts>
#include <complex>
#include <vector>

QT_CHARTS_USE_NAMESPACE

class WaveformPlotter : public QMainWindow {
    Q_OBJECT
public:
    WaveformPlotter(const std::string& filepath, QWidget *parent = nullptr);

private:
    void loadData(const std::string &filename);
    void plot();

    std::vector<std::complex<float>> data;
    QChartView *chartView;
};
