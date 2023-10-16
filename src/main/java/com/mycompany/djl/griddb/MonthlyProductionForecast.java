package com.mycompany.djl.griddb;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.dataset.TimeFeaturizers;
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
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import com.google.gson.GsonBuilder;
import com.mycompany.djl.griddb.datasets.GridDBDataset;
import com.mycompany.djl.griddb.datasets.MySQLDataset;
import com.mycompany.djl.griddb.datasets.MySQLDataset.MySQLBBuilder;
import com.mycompany.djl.griddb.db.DB.Entry;
import com.toshiba.mwcloud.gs.GridStore;
import com.toshiba.mwcloud.gs.TimeSeries;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 * @author ambag
 */
public class MonthlyProductionForecast {

    final static String FREQ = "W";
    final static int PREDICTION_LENGTH = 4;
    final static LocalDateTime START_TIME = LocalDateTime.parse("1985-01-01T00:00");
    final static String MODEL_OUTPUT_DIR = "outputs";
    final static int DATA_LENGTH = 397;
    
    public static void main(String[] args) throws Exception {
        System.out.println("Starting...");
        //GridDBDataset.connectToGridDB();
        //seedDatabase();
        MySQLDataset.connectToMySQL();
        seedMySQLDatabase();
        startTraining();
    }

    public static Map<String, Float> predict(String outputDir)
            throws IOException, TranslateException, ModelException, Exception {
        try ( Model model = Model.newInstance("deepar")) {
            DeepARNetwork predictionNetwork = getDeepARModel(new NegativeBinomialOutput(), false);
            model.setBlock(predictionNetwork);
            model.load(Paths.get(MODEL_OUTPUT_DIR));

            Dataset testSet
                    //= getDataset(Dataset.Usage.TEST, predictionNetwork.getContextLength(), new ArrayList<>());
                    = getMySQLDataset(Dataset.Usage.TEST, predictionNetwork.getContextLength(), new ArrayList<>());

            Map<String, Object> arguments = new ConcurrentHashMap<>();
            arguments.put("prediction_length", 1); // Univariate, so predict one value at a time
            arguments.put("freq", "W");
            arguments.put("use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(), false);
            arguments.put("use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(), false);
            arguments.put("use_" + FieldName.FEAT_STATIC_REAL.name().toLowerCase(), false);
            DeepARTranslator translator = DeepARTranslator.builder(arguments).build();

            M5Evaluator evaluator
                    = new M5Evaluator(0.5f, 0.67f, 0.95f, 0.99f);
            Progress progress = new ProgressBar();
            try ( Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor(translator)) {
                for (Batch batch : testSet.getData(model.getNDManager().newSubManager())) {
                    NDList data = batch.getData();
                    NDArray target = data.head();

                    NDArray gt = target.get(":, {}:", -PREDICTION_LENGTH);
                    NDArray pastTarget = target.get(":, :{}", -PREDICTION_LENGTH);

                    NDList gtSplit = gt.split(batch.getSize());
                    NDList pastTargetSplit = pastTarget.split(batch.getSize());

                    List<TimeSeriesData> batchInput = new ArrayList<>(batch.getSize());
                    for (int i = 0; i < batch.getSize(); i++) {
                        TimeSeriesData input = new TimeSeriesData(10);
                        input.setStartTime(START_TIME);
                        input.setField(FieldName.TARGET, pastTargetSplit.get(i).squeeze(0));
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
            }

            return evaluator.computeTotalMetrics();
        }
    }

    private static void startTraining() throws IOException, TranslateException, Exception {

        DistributionOutput distributionOutput = new NegativeBinomialOutput();

        Model model = null;
        Trainer trainer = null;
        NDManager manager = null;
        try {
            manager = NDManager.newBaseManager();
            model = Model.newInstance("deepar");
            DeepARNetwork trainingNetwork = getDeepARModel(distributionOutput, true);
            model.setBlock(trainingNetwork);

            List<TimeSeriesTransform> trainingTransformation = trainingNetwork.createTrainingTransformation(manager);

            //Dataset trainSet = getDataset(Dataset.Usage.TRAIN, trainingNetwork.getContextLength(), trainingTransformation);
            Dataset trainSet = getMySQLDataset(Dataset.Usage.TRAIN, trainingNetwork.getContextLength(), trainingTransformation);
            trainer = model.newTrainer(setupTrainingConfig(distributionOutput));
            trainer.setMetrics(new Metrics());

            /*int historyLength = trainingNetwork.getHistoryLength();
            Shape[] inputShapes = new Shape[9];
            inputShapes[0] = new Shape(1, 5);
            inputShapes[1] = new Shape(1, 1);

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
            inputShapes[8] = new Shape(1, PREDICTION_LENGTH);*/
            trainer.initialize(new Shape(1,1));
            int epoch = 10;
            EasyTrain.fit(trainer, epoch, trainSet, null);
        } finally {
            if (trainer != null) {
                trainer.close();
            }
            if (model != null) {
                model.close();
            }
            if (manager != null) {
                manager.close();
            }
        }
    }

    private static DeepARNetwork getDeepARModel(DistributionOutput distributionOutput, boolean training) {

        
        List<Integer> cardinality = new ArrayList<>();
       
        cardinality.add(DATA_LENGTH);

        DeepARNetwork.Builder builder = DeepARNetwork.builder()
                .setCardinality(cardinality)
                .setFreq(FREQ)
                .setPredictionLength(PREDICTION_LENGTH)
                .optDistrOutput(distributionOutput)
                .optUseFeatStaticCat(false);

        return training ? builder.buildTrainingNetwork() : builder.buildPredictionNetwork();
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

    private static Dataset getDataset(Dataset.Usage usage,
            int contextLength,
            List<TimeSeriesTransform> transformation) throws IOException {

        GridDBDataset.GridDBBuilder builder
                = GridDBDataset.gridDBBuilder()
                        .optUsage(usage)
                        .setTransformation(transformation)
                        .setContextLength(contextLength)
                        .initData()
                        .setSampling(12, usage == Dataset.Usage.TRAIN);

        int maxWeek = usage == Dataset.Usage.TRAIN ? DATA_LENGTH - 12 : DATA_LENGTH;
        for (int i = 1; i <= maxWeek; i++) {
            builder.addFeature("w_" + i, FieldName.TARGET);
        }

        GridDBDataset dataset
                = builder.addFieldFeature(FieldName.START,
                        new Feature(
                                "date",
                                TimeFeaturizers.getConstantTimeFeaturizer(START_TIME)))
                        .build();
        dataset.prepare(new ProgressBar());
        return dataset;
    }

    public static final class M5Evaluator {

        private float[] quantiles;
        Map<String, Float> totalMetrics;
        Map<String, Integer> totalNum;

        public M5Evaluator(float... quantiles) {
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

    private static Entry[] getTimeSeriesData(URL url) throws Exception {
        Reader reader = null;
        Entry[] entries;
        try {
            InputStream stream = url.openStream();
            reader = new InputStreamReader(stream, StandardCharsets.UTF_8);
            entries
                    = new GsonBuilder()
                            .setDateFormat("M/d/yyyy")
                            .create()
                            .fromJson(reader, Entry[].class);
        } finally {
            if (reader != null) {
                reader.close();
            }
        }
        return entries;
    }

    /*
    We assume the database is already containing the timeseries data
     */
    private static void seedDatabase() throws Exception {
        try ( GridStore store = GridDBDataset.connectToGridDB()) {
            Entry[] entries = getTimeSeriesData(MonthlyProductionForecast.class.getClassLoader().getResource("data/csvjson.json"));
            TimeSeries<Entry> timeSeries = store.putTimeSeries("ENTRIES", Entry.class);
            for (Entry entry : entries) {
                System.out.println(String.format("%s , %s", entry.createdAt, entry.value));
                timeSeries.put(entry);
            }
        }
    }

    private static void seedMySQLDatabase() throws Exception {
        String insertQL = "INSERT INTO `entries` (createdAt, value) VALUES (?, ?)";
        try ( Connection connection = MySQLDataset.connectToMySQL();  PreparedStatement preparedStatement = connection.prepareStatement(insertQL)) {
            Entry[] entries = getTimeSeriesData(MonthlyProductionForecast.class.getClassLoader().getResource("data/csvjson.json"));
            for (Entry entry : entries) {
                System.out.println(String.format("%s , %s", entry.createdAt, entry.value));
                preparedStatement.setDate(1, new java.sql.Date(entry.createdAt.getTime()));
                preparedStatement.setDouble(2, entry.value);
                preparedStatement.addBatch();
            }
            preparedStatement.executeBatch();
        }
    }

    private static Dataset getMySQLDataset(Dataset.Usage usage,
            int contextLength,
            List<TimeSeriesTransform> transformation) throws IOException, Exception {

        MySQLBBuilder builder
                =  ((MySQLBBuilder)MySQLDataset.gridDBBuilder()
                        .setTransformation(transformation)
                        .setContextLength(contextLength)
                        .setSampling(32, usage == Dataset.Usage.TRAIN))
                .initData();              

        builder.addFieldFeature(FieldName.START,
                        new Feature(
                                "createdAt",
                                TimeFeaturizers.getConstantTimeFeaturizer(START_TIME)));                        
        builder.addFeature("value", FieldName.TARGET);
        
        Dataset dataset = builder.buildMySQLDataset();
        dataset.prepare(new ProgressBar());
        return dataset;
    }
}
