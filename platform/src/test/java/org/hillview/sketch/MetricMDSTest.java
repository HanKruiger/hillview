package org.hillview.sketch;

import org.hillview.utils.MetricMDS;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.knowm.xchart.*;

import java.io.IOException;

public class MetricMDSTest {

    private DoubleMatrix makeGrid(int sizeX, int sizeY, double scale) {
        DoubleMatrix data = new DoubleMatrix(sizeX * sizeY, 2);
        for (int i = 0; i < sizeY; i++) {
            for (int j = 0; j < sizeX; j++) {
                data.put(i * sizeX + j, 0, j);
                data.put(i * sizeX + j, 1, i);
            }
        }
        data.divi(Math.max(sizeX, sizeY));
        data.muli(scale);
        return data;
    }

    @Test
    public void testMetricMds() {
        double scale = 500;
        DoubleMatrix data = this.makeGrid(25, 4, scale);

        MetricMDS mds = new MetricMDS(data);
        DoubleMatrix proj = mds.computeEmbedding();

        XYChart chart = new XYChartBuilder().width(1000).height(1000).build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setChartTitleVisible(false);
        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setMarkerSize(16);
        XYSeries a = chart.addSeries("Original", data.getColumn(0).mul(mds.scaling).data, data.getColumn
                (1).mul(mds.scaling).data);
        chart.addSeries("MDS", proj.getColumn(0).data, proj.getColumn(1).data);
        try {
            BitmapEncoder.saveBitmap(chart, "./projection", BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
