package org.hillview.utils;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.PointRange;
import org.jblas.ranges.AllRange;

import java.util.Random;
import java.util.function.BiFunction;

public class MetricMDS {
    public static int maxIterations = (int) 1e4;
    public static double defaultLearningRate = 0.2;
    public static double defaultLearningRateDecay = 0.995;
    public static double tolerance = 1e-9;
    public static int minConvergedCount = 10;

    public static BiFunction<DoubleMatrix, DoubleMatrix, Double> squaredEuclid = (x1, x2) -> MatrixFunctions.pow(x1.sub
            (x2), 2).sum();
    public static BiFunction<DoubleMatrix, DoubleMatrix, Double> euclid = (x1, x2) -> Math.sqrt(MetricMDS.squaredEuclid.apply(x1, x2));

    /**
     * Number of observations in the dataset.
     */
    private int numObservations;
    /**
     * Number of output dimensions
     */
    private int lowDims;
    /**
     * Learning rate and its decay to use in the optimization.
     */
    public double learningRate = MetricMDS.defaultLearningRate;
    public double learningRateDecay = MetricMDS.defaultLearningRateDecay;
    /**
     * If the magnitude of the gradient is smaller than this value, we consider the optimization converged.
     */
    public double stopTolerance = MetricMDS.tolerance;

    /**
     * All pairwise distances d(i, j) in nD. Since it is symmetric, only the upper-triangular part is stored.
     * It is indexed as follows: d(i, j) = d(j, i) = ndDists[i * (N - (i + 3) / 2) + j - 1], with i < j < N, and N the
     * number of observations.
     * Note that the diagonal d(i, i) is not contained in the matrix, as d(i, i) = 0 always.
     */
    private final DoubleMatrix ndDists;
    public double scaling;
    /**
     * Same format for the low-dimensional distances, but this matrix is recomputed every epoch.
     */
    private final DoubleMatrix mdDists;

    /**
     * The low-dimensional embedding of the high-dimensional data.
     */
    private final DoubleMatrix dataMd;

    /**
     * Constructs an object that calculates the MDS projection. Note that the low-dimensional distance metric is
     * always the Euclidean distance, as the gradient is calculated for this.
     * @param dataNd High-dimensional data with observations/{data points} as rows, and dimensions/features as columns.
     * @param lowDims The target dimensionality of the embedding. Commonly 2.
     * @param highDimDist The distance function for nD observations.
     */
    public MetricMDS(DoubleMatrix dataNd, int lowDims, BiFunction<DoubleMatrix, DoubleMatrix, Double> highDimDist) {
        this.numObservations = dataNd.rows;
        this.lowDims = lowDims;
        this.ndDists = this.computeHighDimDistances(dataNd, highDimDist);
        this.dataMd = new DoubleMatrix();
        this.mdDists = new DoubleMatrix();
    }

    public MetricMDS(DoubleMatrix dataNd, int lowDims) {
        this(dataNd, lowDims, MetricMDS.euclid);
    }

    public MetricMDS(DoubleMatrix dataNd) {
        this(dataNd, 2);
    }

    private int compactIndex(int i, int j) {
        return i * this.numObservations - (i * (i + 3)) / 2 + j - 1;
    }

    private DoubleMatrix computeHighDimDistances(DoubleMatrix dataNd, BiFunction<DoubleMatrix, DoubleMatrix, Double> distNd) {
        DoubleMatrix dists = new DoubleMatrix((dataNd.rows * (dataNd.rows - 1)) / 2);
        for (int i = 0; i < dataNd.rows - 1; i++) {
            DoubleMatrix x1 = dataNd.get(new PointRange(i), new AllRange());
            for (int j = i + 1; j < dataNd.rows; j++) {
                DoubleMatrix x2 = dataNd.get(new PointRange(j), new AllRange());
                double dist = distNd.apply(x1, x2);
                int idx = this.compactIndex(i, j);
                dists.put(idx, dist);
            }
        }
        /* Normalize the distances s.t. the largest is 1. */
        this.scaling = 1 / dists.max();
        dists.muli(this.scaling);

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
            return this.ndDists.get(this.compactIndex(i, j));
        else if (j < i) {
            return this.ndDists.get(this.compactIndex(j, i));
        } else {
            return 0;
        }
    }

    private DoubleMatrix computeLowDimDistances(DoubleMatrix dataMd) {
        DoubleMatrix dists = new DoubleMatrix((dataMd.rows * (dataMd.rows - 1)) / 2);
        for (int i = 0; i < dataMd.rows - 1; i++) {
            DoubleMatrix x1 = dataMd.get(new PointRange(i), new AllRange());
            for (int j = i + 1; j < dataMd.rows; j++) {
                DoubleMatrix x2 = dataMd.get(new PointRange(j), new AllRange());
                double dist = MetricMDS.euclid.apply(x1, x2);
                int idx = this.compactIndex(i, j);
                dists.put(idx, dist);
            }
        }
        return dists;
    }

    private double getLowDimDist(int i, int j) {
        if (i < j)
            return this.mdDists.get(this.compactIndex(i, j));
        else if (j < i) {
            return this.mdDists.get(this.compactIndex(j, i));
        } else {
            return 0;
        }
    }

    public DoubleMatrix computeEmbedding(DoubleMatrix dataMdInit) {
        this.dataMd.copy(dataMdInit);
        
        int iterations = 0;
        double step;
        double cost = 0.0;
        int convergedCount = 0;
        do {
            DoubleMatrix gradient = this.gradient();
            double magnitude = MatrixFunctions.pow(gradient, 2).sum();
            /* Actual learning step */
            this.dataMd.addi(gradient.mul(this.learningRate).neg());
            double newCost = this.cost();
            step = (newCost - cost) / this.learningRate;
            System.out.println(String.format("[iter %d]:\n\tcost: %6.3e\n\tstep: %6.3e\n\tmagnitude: %6.3e",
                    iterations, newCost, step, magnitude));
            iterations++;
            cost = newCost;
            this.learningRate *= this.learningRateDecay;
            if (iterations > 0 && Math.abs(step) < MetricMDS.tolerance) {
                convergedCount++;
                System.out.println(convergedCount);
            } else
                convergedCount = 0;
        } while (convergedCount < MetricMDS.minConvergedCount && iterations < MetricMDS.maxIterations);

        if (convergedCount < MetricMDS.minConvergedCount)
            System.out.println("Warning: Terminated before tolerance was met.");
        return this.dataMd;
    }

    public DoubleMatrix computeEmbedding() {
        DoubleMatrix dataMdInit = new DoubleMatrix(this.numObservations, this.lowDims);
        Random rnd = new Random();
        rnd.nextGaussian();
        for (int i = 0; i < this.numObservations; i++) {
            for (int j = 0; j < this.lowDims; j++) {
                dataMdInit.put(i, j, rnd.nextGaussian());
            }
        }
        return this.computeEmbedding(dataMdInit);
    }

    /**
     * Compute the gradient.
     * @return The gradient of the cost function w.r.t. the low-dimensional points.
     */
    private DoubleMatrix gradient() {
        DoubleMatrix gradient = DoubleMatrix.zeros(this.numObservations, this.lowDims);
        this.mdDists.copy(this.computeLowDimDistances(this.dataMd));

        /* Loop over all pairs of points. */
        for (int i = 0; i < this.numObservations - 1; i++) {
            DoubleMatrix pointI = this.dataMd.getRow(i);
            for (int j = i + 1; j < this.numObservations; j++) {
                DoubleMatrix pointJ = this.dataMd.getRow(j);
                /* Compute gradient for point i (only w.r.t. points with index > i) */

                /* Vector from (low-dim) point i to point j */
                DoubleMatrix gradientI = pointJ.sub(pointI);
                /* Make it a unit vector */
                gradientI.divi(this.getLowDimDist(j, i));

                /* Scale by the discrepancy */
                gradientI.muli(this.getHighDimDist(j, i) - this.getLowDimDist(j, i));

                /* Exploit symmetry: The gradient on i caused by j is equal to the inverse gradient on j caused by i. */
                gradient.putRow(i, gradient.getRow(i).add(gradientI));
                gradient.putRow(j, gradient.getRow(j).add(gradientI.neg()));
            }
        }

        /* Normalization factors */
        double c = 1 / Math.sqrt(MatrixFunctions.pow(this.ndDists.sub(this.mdDists), 2).sum());
        gradient.muli(c);

        return gradient;
    }

    public double cost() {
        double cost = MatrixFunctions.pow(this.ndDists.sub(this.mdDists), 2).sum();
        cost = Math.sqrt(cost);
        return cost;
    }
}
