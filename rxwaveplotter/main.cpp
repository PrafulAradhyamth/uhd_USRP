
#include <QApplication>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QChart>
#include <QtCharts/QValueAxis>

#include <complex>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>



// Function to read fc32 binary data - moved before main for declaration
std::vector<std::complex<float>> read_fc32_bin(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << "\n";
        return {};
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check if file size is a multiple of complex float size (2 floats)
    if (filesize % (2 * sizeof(float)) != 0) {
        std::cerr << "File size (" << filesize << " bytes) not a multiple of complex float size (" << (2 * sizeof(float)) << " bytes).\n";
        return {};
    }

    size_t num_samples = filesize / (2 * sizeof(float));
    std::vector<std::complex<float>> data(num_samples);

    // Read real and imaginary parts for each complex number
    for (size_t i = 0; i < num_samples; ++i) {
        float real, imag;
        file.read(reinterpret_cast<char*>(&real), sizeof(float));
        file.read(reinterpret_cast<char*>(&imag), sizeof(float));
        data[i] = std::complex<float>(real, imag);
    }
    return data;
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Check for command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <fc32_bin_file>\n";
        return -1;
    }

    // Read data from the specified binary file
    const auto data = read_fc32_bin(argv[1]);
    if (data.empty()) {
        std::cerr << "No data read from file. Exiting.\n";
        return -1;
    }

    // Create a QLineSeries to hold the magnitude data
    QLineSeries *series = new QLineSeries();
    // QLineSeries does not have a reserve() method; append handles allocation.

    // Populate the series with magnitude data
    for (size_t i = 0; i < data.size(); ++i) {
        series->append(i, std::abs(data[i]));
    }

    // Create a QChart and add the series
    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Magnitude Plot of fc32 Data");

    // Configure X-axis (Sample Index)
    QValueAxis *axisX = new QValueAxis();
    axisX->setTitleText("Sample Index");
    axisX->setLabelFormat("%d"); // Integer format for sample index
    axisX->setRange(0, data.size()); // Set range from 0 to number of samples
    chart->addAxis(axisX, Qt::AlignBottom); // Add X-axis to the chart at the bottom
    series->attachAxis(axisX); // Attach the series to the X-axis

    // Configure Y-axis (Magnitude)
    QValueAxis *axisY = new QValueAxis();
    axisY->setTitleText("Magnitude");
    axisY->setLabelFormat("%.2f"); // Two decimal places for magnitude

    // Find the maximum magnitude for Y-axis range
    float max_mag = 0.0f;
    for (const auto &c : data) {
        max_mag = std::max(max_mag, std::abs(c));
    }
    // Set Y-axis range with a 10% margin above the max magnitude
    axisY->setRange(0, max_mag * 1.1f);
    chart->addAxis(axisY, Qt::AlignLeft); // Add Y-axis to the chart on the left
    series->attachAxis(axisY); // Attach the series to the Y-axis

    // Create a QChartView to display the chart
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing); // Enable antialiasing for smoother rendering
    chartView->setRubberBand(QChartView::RectangleRubberBand); // Enable drag-to-zoom
    chartView->resize(800, 600); // Set initial window size
    chartView->setWindowTitle("FC32 Wave Plotter"); // Set window title
    chartView->show(); // Show the chart window

    // Start the Qt application event loop
    return app.exec();
}