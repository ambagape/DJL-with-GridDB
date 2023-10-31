package com.mycompany.djl.griddb;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.distribution.DistributionLoss;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.NegativeBinomialOutput;
import ai.djl.timeseries.evaluator.Rmsse;
import ai.djl.timeseries.model.deepar.DeepARNetwork;
import ai.djl.timeseries.timefeature.TimeFeature;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.timeseries.translator.DeepARTranslator;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import com.mycompany.djl.griddb.datasets.GridDBDataset;
import com.toshiba.mwcloud.gs.GSException;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 *
 * @author ambag
 */
public class Forecaster {

    final static String FREQ = "W";
    final static int PREDICTION_LENGTH = 4;
    final static LocalDateTime START_TIME = LocalDateTime.parse("2011-01-29T00:00");
    final static String MODEL_OUTPUT_DIR = "outputs";

    public static void main(String[] args) throws Exception {
        Logger.getAnonymousLogger().info("Starting...");        
        startTraining();
        final Map<String, Float> result = predict();
        for (Map.Entry<String, Float> entry : result.entrySet()) {
            Logger.getAnonymousLogger().info(String.format("metric: %s:\t%.2f", entry.getKey(), entry.getValue()));
        }
    }

    public static Map<String, Float> predict()
            throws IOException, TranslateException, ModelException, Exception {
        try ( Model model = Model.newInstance("deepar")) {
            DeepARNetwork predictionNetwork = getDeepARModel(new NegativeBinomialOutput(), false);
            model.setBlock(predictionNetwork);
            model.load(Paths.get(MODEL_OUTPUT_DIR));

            M5Forecast testSet
                    = getDataset(
                            new ArrayList<>(),
                            predictionNetwork.getContextLength(),
                            Dataset.Usage.TEST);

            Map<String, Object> arguments = new ConcurrentHashMap<>();
            arguments.put("prediction_length", PREDICTION_LENGTH);
            arguments.put("freq", FREQ);
            arguments.put("use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(), false);
            arguments.put("use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(), true);
            arguments.put("use_" + FieldName.FEAT_STATIC_REAL.name().toLowerCase(), false);
            DeepARTranslator translator = DeepARTranslator.builder(arguments).build();

            M5ForecastingEvaluator evaluator
                    = new M5ForecastingEvaluator(0.5f, 0.67f, 0.95f, 0.99f);
            Progress progress = new ProgressBar();
            progress.reset("Inferring", testSet.size());
            try ( Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor(translator)) {
                for (Batch batch : testSet.getData(model.getNDManager().newSubManager())) {
                    NDList data = batch.getData();
                    NDArray target = data.head();
                    NDArray featStaticCat = data.get(1);

                    NDArray gt = target.get(":, {}:", -PREDICTION_LENGTH);
                    NDArray pastTarget = target.get(":, :{}", -PREDICTION_LENGTH);

                    NDList gtSplit = gt.split(batch.getSize());
                    NDList pastTargetSplit = pastTarget.split(batch.getSize());
                    NDList featStaticCatSplit = featStaticCat.split(batch.getSize());

                    List<TimeSeriesData> batchInput = new ArrayList<>(batch.getSize());
                    for (int i = 0; i < batch.getSize(); i++) {
                        TimeSeriesData input = new TimeSeriesData(10);
                        input.setStartTime(START_TIME);
                        input.setField(FieldName.TARGET, pastTargetSplit.get(i).squeeze(0));
                        input.setField(
                                FieldName.FEAT_STATIC_CAT, featStaticCatSplit.get(i).squeeze(0));
                        batchInput.add(input);
                    }
                    List<Forecast> forecasts = predictor.batchPredict(batchInput);
                    for (int i = 0; i < forecasts.size(); i++) {
                        evaluator.aggregateMetrics(
                                evaluator.getMetricsPerTs(
                                        gtSplit.get(i).squeeze(0),
                                        pastTargetSplit.get(i).squeeze(0),
                                        forecasts.get(i)));
                    }
                    progress.increment(batch.getSize());
                    batch.close();
                }
                return evaluator.computeTotalMetrics();
            }
        }
    }

    private static void startTraining() throws IOException, TranslateException, Exception {

        try ( Model model = Model.newInstance("deepar")) {
            // specify the model distribution output, for M5 case, NegativeBinomial best describe it
            DistributionOutput distributionOutput = new NegativeBinomialOutput();
            DefaultTrainingConfig config = setupTrainingConfig(distributionOutput);

            NDManager manager = model.getNDManager();
            DeepARNetwork trainingNetwork = getDeepARModel(distributionOutput, true);
            model.setBlock(trainingNetwork);

            List<TimeSeriesTransform> trainingTransformation
                    = trainingNetwork.createTrainingTransformation(manager);
            int contextLength = trainingNetwork.getContextLength();

            M5Forecast trainSet
                    = getDataset(trainingTransformation, contextLength, Dataset.Usage.TRAIN);

            try ( Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                int historyLength = trainingNetwork.getHistoryLength();
                Shape[] inputShapes = new Shape[9];
                // (N, num_cardinality)
                inputShapes[0] = new Shape(1, 5);
                // (N, num_real) if use_feat_stat_real else (N, 1)
                inputShapes[1] = new Shape(1, 1);
                // (N, history_length, num_time_feat + num_age_feat)
                inputShapes[2]
                        = new Shape(
                                1,
                                historyLength,
                                TimeFeature.timeFeaturesFromFreqStr(FREQ).size() + 1);
                inputShapes[3] = new Shape(1, historyLength);
                inputShapes[4] = new Shape(1, historyLength);
                inputShapes[5] = new Shape(1, historyLength);
                inputShapes[6]
                        = new Shape(
                                1,
                                PREDICTION_LENGTH,
                                TimeFeature.timeFeaturesFromFreqStr(FREQ).size() + 1);
                inputShapes[7] = new Shape(1, PREDICTION_LENGTH);
                inputShapes[8] = new Shape(1, PREDICTION_LENGTH);
                trainer.initialize(inputShapes);
                int epoch = 10;
                EasyTrain.fit(trainer, epoch, trainSet, null);
            }
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(DistributionOutput distributionOutput) {

        SaveModelTrainingListener listener = new SaveModelTrainingListener(MODEL_OUTPUT_DIR);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float rmsse = result.getValidateEvaluation("RMSSE");
                    model.setProperty("RMSSE", String.format("%.5f", rmsse));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(new DistributionLoss("Loss", distributionOutput))
                .addEvaluator(new Rmsse(distributionOutput))
                .optInitializer(new XavierInitializer(), Parameter.Type.WEIGHT)
                .addTrainingListeners(TrainingListener.Defaults.logging(MODEL_OUTPUT_DIR))
                .addTrainingListeners(listener);
    }

    private static DeepARNetwork getDeepARModel(DistributionOutput distributionOutput, boolean training) {

        List<Integer> cardinality = new ArrayList<>();
        cardinality.add(3);
        cardinality.add(10);
        cardinality.add(3);
        cardinality.add(7);
        cardinality.add(3049);

        DeepARNetwork.Builder builder = DeepARNetwork.builder()
                .setCardinality(cardinality)
                .setFreq(FREQ)
                .setPredictionLength(PREDICTION_LENGTH)
                .optDistrOutput(distributionOutput)
                .optUseFeatStaticCat(true);

        return training ? builder.buildTrainingNetwork() : builder.buildPredictionNetwork();
    }

    private static M5Forecast getDataset(
            List<TimeSeriesTransform> transformation, int contextLength, Dataset.Usage usage)
            throws IOException, GSException, FileNotFoundException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException, Exception {
        // In order to create a TimeSeriesDataset, you must specify the transformation of the data
        // preprocessing
        GridDBDataset.GridDBBuilder builder
                = GridDBDataset.gridDBBuilder()
                        .optUsage(usage)
                        .setTransformation(transformation)
                        .setContextLength(contextLength)
                        .setSize(32)
                        .setStartTime(START_TIME)
                        .setRandom(usage == Dataset.Usage.TRAIN)
                        .setMaxWeek(usage == Dataset.Usage.TRAIN ? 273 : 277)
                        .initData();

        M5Forecast m5Forecast = builder.build();

        m5Forecast.prepare(new ProgressBar());
        return m5Forecast;
    }   
}
