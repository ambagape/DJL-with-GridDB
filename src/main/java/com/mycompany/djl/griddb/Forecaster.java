package com.mycompany.djl.griddb;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
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
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import com.mycompany.djl.griddb.datasets.GridDBDataset;
import com.opencsv.CSVReader;
import com.toshiba.mwcloud.gs.ColumnInfo;
import com.toshiba.mwcloud.gs.Container;
import com.toshiba.mwcloud.gs.ContainerInfo;
import com.toshiba.mwcloud.gs.ContainerType;
import com.toshiba.mwcloud.gs.GSException;
import com.toshiba.mwcloud.gs.GSType;
import com.toshiba.mwcloud.gs.GridStore;
import com.toshiba.mwcloud.gs.Row;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
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
    final static String TRAINING_COLLECTION_NAME = "NNTraining";
    final static String VALIDATION_COLLECTION_NAME = "NNValidation";

    public static void main(String[] args) throws Exception {
        Logger.getAnonymousLogger().info("Starting...");
        //GridDBDataset.connectToGridDB();
        //seedDatabase();
        GridDBDataset.connectToGridDB();
        seedDatabase();
        startTraining();
    }

    public static void predict(String outputDir)
            throws IOException, TranslateException, ModelException, Exception {
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
                    = getDatasetFomrRepository(trainingTransformation, contextLength, Dataset.Usage.TRAIN);

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
            throws IOException, GSException, FileNotFoundException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
        // In order to create a TimeSeriesDataset, you must specify the transformation of the data
        // preprocessing
        GridDBDataset.GridDBBuilder builder
                = GridDBDataset.gridDBBuilder()
                        .setContainerName(usage == Dataset.Usage.TRAIN?  TRAINING_COLLECTION_NAME: VALIDATION_COLLECTION_NAME )
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
    
    private static M5Forecast getDatasetFomrRepository(
            List<TimeSeriesTransform> transformation, int contextLength, Dataset.Usage usage)
            throws IOException {
        // In order to create a TimeSeriesDataset, you must specify the transformation of the data
        // preprocessing
        M5Forecast.Builder builder =
                M5Forecast.builder()
                        .optUsage(usage)
                        .optRepository(BasicDatasets.REPOSITORY)
                        .optGroupId(BasicDatasets.GROUP_ID)
                        .optArtifactId("m5forecast-unittest")
                        .setTransformation(transformation)
                        .setContextLength(contextLength)
                        .setSampling(32, usage == Dataset.Usage.TRAIN);

        int maxWeek = usage == Dataset.Usage.TRAIN ? 273 : 277;
        for (int i = 1; i <= maxWeek; i++) {
            builder.addFeature("w_" + i, FieldName.TARGET);
        }

        M5Forecast m5Forecast =
                builder.addFeature("state_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("store_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("cat_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("dept_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("item_id", FieldName.FEAT_STATIC_CAT)
                        .addFieldFeature(
                                FieldName.START,
                                new Feature(
                                        "date",
                                        TimeFeaturizers.getConstantTimeFeaturizer(START_TIME)))
                        .build();
        m5Forecast.prepare(new ProgressBar());
        return m5Forecast;
    }
    
   
    /*
    We assume the database is already containing the timeseries data
     */
    private static void seedDatabase() throws Exception {
        URL trainingData = Forecaster.class.getClassLoader().getResource("data/weekly_sales_train_evaluation.csv");
        URL validationData = Forecaster.class.getClassLoader().getResource("data/weekly_sales_train_validation.csv");
        String[] nextRecord;
        try ( GridStore store = GridDBDataset.connectToGridDB();  CSVReader csvReader = new CSVReader(new InputStreamReader(trainingData.openStream(), StandardCharsets.UTF_8));  CSVReader csvValidationReader = new CSVReader(new InputStreamReader(validationData.openStream(), StandardCharsets.UTF_8))) {
            store.dropContainer(TRAINING_COLLECTION_NAME);
            store.dropContainer(VALIDATION_COLLECTION_NAME);
            
            List<ColumnInfo> columnInfoList = new ArrayList<>();

            nextRecord = csvReader.readNext();
            for (int i = 0; i < nextRecord.length; i++) {
                ColumnInfo columnInfo = new ColumnInfo(nextRecord[i], GSType.STRING);
                columnInfoList.add(columnInfo);
            }

            ContainerInfo containerInfo = new ContainerInfo();
            containerInfo.setColumnInfoList(columnInfoList);
            containerInfo.setName(TRAINING_COLLECTION_NAME);
            containerInfo.setType(ContainerType.COLLECTION);
            
            Container<String, Row> container = store.putContainer(TRAINING_COLLECTION_NAME, containerInfo, false);
            
            while ((nextRecord = csvReader.readNext()) != null) {
                Row row = container.createRow();
                for (int i = 0; i < nextRecord.length; i++) {
                    row.setString(i, nextRecord[i]);
                }
                container.put(row);
            }            

            nextRecord = csvValidationReader.readNext();
            columnInfoList.clear();
            for (int i = 0; i < nextRecord.length; i++) {
                ColumnInfo columnInfo = new ColumnInfo(nextRecord[i], GSType.STRING);
                columnInfoList.add(columnInfo);
            }

            containerInfo = new ContainerInfo();
            containerInfo.setName(VALIDATION_COLLECTION_NAME);
            containerInfo.setColumnInfoList(columnInfoList);
            containerInfo.setType(ContainerType.COLLECTION);

            container = store.putContainer(VALIDATION_COLLECTION_NAME, containerInfo, false);
            while ((nextRecord = csvValidationReader.readNext()) != null) {
                Row row = container.createRow();
                for (int i = 0; i < nextRecord.length; i++) {
                    String cell = nextRecord[i];
                    row.setString(i, cell);
                }
                container.put(row);
            }
        }
    }

}
