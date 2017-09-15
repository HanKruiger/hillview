package org.hillview.utils;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.PointRange;
import org.jblas.ranges.AllRange;

import java.util.function.BiFunction;

public class MetricMDS {
    public static int maxIters = 1000;
    public static double defaultLearningRate = 0.1;
    public static double defaultStopTolerance = 0.1;
    public static BiFunction<DoubleMatrix, DoubleMatrix, Double> squaredEuclid = (x1, x2) -> MatrixFunctions.pow(x1.min(x2), 2).sum();
    public static BiFunction<DoubleMatrix, DoubleMatrix, Double> euclid = (x1, x2) -> MatrixFunctions.sqrt(MetricMDS.squaredEuclid.apply(x1, x2));

    /**
     * Number of observations in the dataset.
     */
    private int N;
    /**
     * Number of input dimensions
     */
    private int n;
    /**
     * Number of output dimensions
     */
    private int m;
    /**
     * Learning rate to use in the optimization.
     */
    public double learningRate = MetricMDS.defaultLearningRate;
    /**
     * If the magnitude of the gradient is smaller than this value, we consider the optimization converged.
     */
    public double stopTolerance = MetricMDS.defaultStopTolerance;

    /**
     * Constant term in the cost gradient that can be reused over the entire optimization.
     * It is the square root of 1 over the sum of the distances squared between all (unordered, so
     * only counted once) pairs of nD points.
     */
    private double c1;

    private BiFunction<DoubleMatrix, DoubleMatrix, Double> lowDimDist;

    /**
     * All pairwise distances d(i, j) in nD. Since it is symmetric, only the upper-triangular part is stored.
     * It is indexed as follows: d(i, j) = d(j, i) = ndDists[i * (N - (i + 3) / 2) + j - 1], with i < j < N, and N the
     * number of observations.
     * Note that the diagonal d(i, i) is not contained in the matrix, as d(i, i) = 0 always.
     */
    private final DoubleMatrix ndDists;

    /**
     * The low-dimensional embedding of the high-dimensional data.
     */
    private DoubleMatrix dataMd;

    /**
     *
     * @param dataNd High-dimensional data with observations/{data points} as rows, and dimensions/features as columns.
     * @param m The target dimensionality of the embedding. Commonly 2.
     * @param highDimDist The distance function for nD observations. Commonly Euclidean.
     * @param lowDimDist The distance function for mD observations. Commonly Euclidean.
     */
    public MetricMDS(DoubleMatrix dataNd, int m, BiFunction<DoubleMatrix, DoubleMatrix, Double> highDimDist,
                     BiFunction<DoubleMatrix, DoubleMatrix, Double> lowDimDist) {
        this.N = dataNd.rows;
        this.n = dataNd.columns;
        this.m = m;
        this.lowDimDist = lowDimDist;
        this.ndDists = this.computeNdDistances(dataNd, highDimDist);
    }

    public MetricMDS(DoubleMatrix dataNd, int m, BiFunction<DoubleMatrix, DoubleMatrix, Double> distNd) {
        this(dataNd, m, distNd, MetricMDS.euclid);
    }

    public MetricMDS(DoubleMatrix dataNd, int m) {
        this(dataNd, m, MetricMDS.euclid);
    }

    public MetricMDS(DoubleMatrix dataNd) {
        this(dataNd, 2);
    }

    private DoubleMatrix computeNdDistances(DoubleMatrix dataNd, BiFunction<DoubleMatrix, DoubleMatrix, Double> distNd) {
        DoubleMatrix dists = new DoubleMatrix(dataNd.rows);
        for (int i = 0; i < dataNd.rows - 1; i++) {
            DoubleMatrix x1 = dataNd.get(new PointRange(i), new AllRange());
            for (int j = i + 1; j < dataNd.rows; j++) {
                DoubleMatrix x2 = dataNd.get(new PointRange(j), new AllRange());
                double dist = distNd.apply(x1, x2);
                dists.put(i * (dataNd.rows - (i + 3) / 2) + j - 1, dist);
            }
        }
        this.c1 = Math.sqrt(1 / MatrixFunctions.pow(dists, 2).sum());
        return dists;
    }

    /**
     * Accessor method of the high-dimensional distances.
     * @param i Observation index
     * @param j Observation index
     * @return The distances between the high-dimensional observations.
     */
    private double getHighDimDist(int i, int j) {
        if (i < j)
            return this.ndDists.get(i * (this.N - (i + 3) / 2) + j - 1);
        else if (j < i) {
            return this.ndDists.get(j * (this.N - (j + 3) / 2) + i - 1);
        } else {
            return 0;
        }
    }

    private double getLowDimDist(int i, int j) {
        return this.lowDimDist.apply(
                this.dataMd.get(new PointRange(i), new AllRange()),
                this.dataMd.get(new PointRange(j), new AllRange())
        );
    }

    public DoubleMatrix computeEmbedding(DoubleMatrix dataMdInit) {
        int iters = 0;
        double disp;
        do {
            disp = this.doEpoch();
            iters += 1;
        } while (disp > this.stopTolerance && iters < MetricMDS.maxIters);
        if (disp > this.stopTolerance)
            System.out.println("Warning: Terminated before tolerance was met.");
        return this.dataMd;
    }

    public double doEpoch() {
        DoubleMatrix gradient = new DoubleMatrix(this.N, this.m);
        // TODO: Compute gradient.
        double magnitude = Math.sqrt(MatrixFunctions.pow(gradient, 2).sum());
        this.dataMd.addi(gradient.mul(this.learningRate));
        return magnitude;
    }

    public double cost() {
        double cost = 0.0;
        for (int i = 0; i < this.N - 1; i++) {
            for (int j = i + 1; j < this.N; j++) {
                cost += Math.pow(this.getHighDimDist(i, j) - this.getLowDimDist(i, j), 2);
            }
        }
        cost = Math.sqrt(cost);
        cost *= this.c1;
        return cost;
    }
}
