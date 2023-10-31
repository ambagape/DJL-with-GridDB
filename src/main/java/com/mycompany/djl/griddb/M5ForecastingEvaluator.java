/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.djl.griddb;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.timeseries.Forecast;
import ai.djl.training.loss.Loss;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 * @author ambag
 */
public class M5ForecastingEvaluator {

    /**
     * An evaluator that calculates performance metrics.
     */
    private float[] quantiles;
    Map<String, Float> totalMetrics;
    Map<String, Integer> totalNum;

    public M5ForecastingEvaluator(float... quantiles) {
        this.quantiles = quantiles;
        totalMetrics = new ConcurrentHashMap<>();
        totalNum = new ConcurrentHashMap<>();
        init();
    }

    public Map<String, Float> getMetricsPerTs(
            NDArray gtTarget, NDArray pastTarget, Forecast forecast) {
        Map<String, Float> retMetrics
                = new ConcurrentHashMap<>((8 + quantiles.length * 2) * 3 / 2);
        NDArray meanFcst = forecast.mean();
        NDArray medianFcst = forecast.median();

        NDArray meanSquare = gtTarget.sub(meanFcst).square().mean();
        NDArray scaleDenom = gtTarget.get("1:").sub(gtTarget.get(":-1")).square().mean();

        NDArray rmsse = meanSquare.div(scaleDenom).sqrt();
        rmsse = NDArrays.where(scaleDenom.eq(0), rmsse.onesLike(), rmsse);

        retMetrics.put("RMSSE", rmsse.getFloat());

        retMetrics.put("MSE", gtTarget.sub(meanFcst).square().mean().getFloat());
        retMetrics.put("abs_error", gtTarget.sub(medianFcst).abs().sum().getFloat());
        retMetrics.put("abs_target_sum", gtTarget.abs().sum().getFloat());
        retMetrics.put("abs_target_mean", gtTarget.abs().mean().getFloat());
        retMetrics.put(
                "MAPE", gtTarget.sub(medianFcst).abs().div(gtTarget.abs()).mean().getFloat());
        retMetrics.put(
                "sMAPE",
                gtTarget.sub(medianFcst)
                        .abs()
                        .div(gtTarget.abs().add(medianFcst.abs()))
                        .mean()
                        .mul(2)
                        .getFloat());
        retMetrics.put("ND", retMetrics.get("abs_error") / retMetrics.get("abs_target_sum"));

        for (float quantile : quantiles) {
            NDArray forecastQuantile = forecast.quantile(quantile);
            NDArray quantileLoss
                    = Loss.quantileL1Loss(quantile)
                            .evaluate(new NDList(gtTarget), new NDList(forecastQuantile));
            NDArray quantileCoverage
                    = gtTarget.lt(forecastQuantile).toType(DataType.FLOAT32, false).mean();
            retMetrics.put(
                    String.format("QuantileLoss[%.2f]", quantile), quantileLoss.getFloat());
            retMetrics.put(
                    String.format("Coverage[%.2f]", quantile), quantileCoverage.getFloat());
        }
        return retMetrics;
    }

    public void aggregateMetrics(Map<String, Float> metrics) {
        for (Map.Entry<String, Float> entry : metrics.entrySet()) {
            totalMetrics.compute(entry.getKey(), (k, v) -> v + entry.getValue());
            totalNum.compute(entry.getKey(), (k, v) -> v + 1);
        }
    }

    public Map<String, Float> computeTotalMetrics() {
        for (Map.Entry<String, Integer> entry : totalNum.entrySet()) {
            if (!entry.getKey().contains("sum")) {
                totalMetrics.compute(entry.getKey(), (k, v) -> v / (float) entry.getValue());
            }
        }

        totalMetrics.put("RMSE", (float) Math.sqrt(totalMetrics.get("MSE")));
        totalMetrics.put(
                "NRMSE", totalMetrics.get("RMSE") / totalMetrics.get("abs_target_mean"));
        return totalMetrics;
    }

    private void init() {
        List<String> metricNames
                = new ArrayList<>(
                        Arrays.asList(
                                "RMSSE",
                                "MSE",
                                "abs_error",
                                "abs_target_sum",
                                "abs_target_mean",
                                "MAPE",
                                "sMAPE",
                                "ND"));
        for (float quantile : quantiles) {
            metricNames.add(String.format("QuantileLoss[%.2f]", quantile));
            metricNames.add(String.format("Coverage[%.2f]", quantile));
        }
        for (String metricName : metricNames) {
            totalMetrics.put(metricName, 0f);
            totalNum.put(metricName, 0);
        }
    }

}
