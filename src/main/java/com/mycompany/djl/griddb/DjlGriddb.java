package com.mycompany.djl.griddb;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.repository.Repository;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
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
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.translate.TranslateException;
import com.google.gson.GsonBuilder;
import com.mycompany.djl.griddb.db.DB.Entry;
import com.toshiba.mwcloud.gs.GSException;
import com.toshiba.mwcloud.gs.GridStore;
import com.toshiba.mwcloud.gs.GridStoreFactory;
import com.toshiba.mwcloud.gs.Query;
import com.toshiba.mwcloud.gs.RowSet;
import com.toshiba.mwcloud.gs.TimeSeries;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 *
 * @author ambag
 */
public class DjlGriddb {

    final static String FREQ = "M";
    final static int PREDICTION_LENGTH = 12;
    //final static double RMSSE = 1.00;

    public static void main(String[] args) throws Exception {
        System.out.println("Starting...");
        Properties props = new Properties();
        props.setProperty("notificationMember", "127.0.0.1:10001");
        props.setProperty("clusterName", "defaultCluster");
        props.setProperty("user", "admin");
        props.setProperty("password", "admin");
        GridStore store = null;
        try {
            store = GridStoreFactory.getInstance().getGridStore(props);
            System.out.println("Connected to GridDB...");
            seedDatabase(store);
            startTraining(store);
            predict();
        } finally {
            if (store != null) {
                try {
                    store.close();
                } catch (GSException e) {
                    System.out.println("An error occurred when releasing the recsource.");
                }
            }
        }
    }

    public static float[] predict() throws Exception {
        return null;
    }

    private static void startTraining(GridStore store) throws IOException, TranslateException {

        Repository repository = Repository.newInstance("local_dataset",
                Paths.get("YOUR_PATH/m5-forecasting-accuracy"));
        NDManager manager = NDManager.newBaseManager();

        DistributionOutput distributionOutput = new NegativeBinomialOutput();

        Model model = null;
        Trainer trainer = null;
        try {            
            model = Model.newInstance("deepar");
            DeepARNetwork trainingNetwork = getDeepARModel(distributionOutput, true);
            model.setBlock(trainingNetwork);

            List<TimeSeriesTransform> trainingTransformation = trainingNetwork.createTrainingTransformation(manager);
            int contextLength = trainingNetwork.getContextLength();

            List<TimeSeriesData> data = getTrainingData(store);
            
            
            Dataset trainSet = getDataset(data);            
             
            trainer = model.newTrainer(setupTrainingConfig(distributionOutput));
            trainer.setMetrics(new Metrics());

            int historyLength = trainingNetwork.getHistoryLength();
            Shape[] inputShapes = new Shape[9];
            // (N, num_cardinality)
            inputShapes[0] = new Shape(1, 5);
            // (N, num_real) if use_feat_stat_real else (N, 1)
            inputShapes[1] = new Shape(1, 1);
            // (N, history_length, num_time_feat + num_age_feat)
            inputShapes[2] = new Shape(1, historyLength, TimeFeature.timeFeaturesFromFreqStr(FREQ).size() + 1);
            inputShapes[3] = new Shape(1, historyLength);
            inputShapes[4] = new Shape(1, historyLength);
            inputShapes[5] = new Shape(1, historyLength);
            inputShapes[6] = new Shape(1, PREDICTION_LENGTH, TimeFeature.timeFeaturesFromFreqStr(FREQ).size() + 1);
            inputShapes[7] = new Shape(1, PREDICTION_LENGTH);
            inputShapes[8] = new Shape(1, PREDICTION_LENGTH);
            trainer.initialize(inputShapes);

            int epoch = 10;
            EasyTrain.fit(trainer, epoch, trainSet, null);
        } finally {
            if (trainer != null) {
                trainer.close();
            }
            if (model != null) {
                model.close();
            }
        }

    }

    private static DeepARNetwork getDeepARModel(DistributionOutput distributionOutput, boolean training) {

        DeepARNetwork.Builder builder = DeepARNetwork.builder()
                .setFreq(FREQ)
                .setPredictionLength(PREDICTION_LENGTH)
                .optDistrOutput(distributionOutput)
                .optUseFeatStaticCat(false);
        return training ? builder.buildTrainingNetwork() : builder.buildPredictionNetwork();
    }

    private static DefaultTrainingConfig setupTrainingConfig(DistributionOutput distributionOutput) {
        return new DefaultTrainingConfig(new DistributionLoss("Loss", distributionOutput))
                .addEvaluator(new Rmsse(distributionOutput))
                .optInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
    }

    private static Dataset getDataset(
            List<TimeSeriesData> data) throws IOException {

        NDArray[] inputList = new NDArray[data.size()];
        NDArray[] targetList = new NDArray[data.size()];
        for (int i = 0; i < data.size(); i++) {
            TimeSeriesData timeSeriesData = data.get(i);
            NDArray input = timeSeriesData.get(FieldName.FEAT_DYNAMIC_REAL).toDense();
            NDArray target = timeSeriesData.get(FieldName.TARGET).toDense();
            inputList[i] = input;
            targetList[i] = target;
        }
        
        // Create the ArrayDataset
        ArrayDataset dataset = new ArrayDataset.Builder()
                .setData(inputList)
                .optLabels(targetList)
                .setSampling(1, true)
                .build();
        return dataset;

    }

    private static List<TimeSeriesData> getTrainingData(GridStore store) throws GSException {
        TimeSeries<Entry> entries = store.getTimeSeries("ENTRIES", Entry.class);
        Query<Entry> query = entries.query("Select * from ENTRIES");
        RowSet<Entry> rowSet = query.getRowSet();

        List<TimeSeriesData> timeSeriesDataList = new ArrayList<>();
        while (rowSet.hasNext()) {
            TimeSeriesData timeSeriesData = new TimeSeriesData(1);
            LocalDateTime timestamp = LocalDateTime.parse((CharSequence) rowSet.next().createdAt);
            float value = ((Number) rowSet.next().value).floatValue();
            timeSeriesData.setStartTime(timestamp);
            timeSeriesData.add(FieldName.TARGET, NDManager.newBaseManager().create(value));
            timeSeriesDataList.add(timeSeriesData);
        }

        return timeSeriesDataList;
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

    private static void seedDatabase(GridStore store) throws Exception {
        Entry[] entries = getTimeSeriesData(DjlGriddb.class.getClassLoader().getResource("data/csvjson.json"));
        TimeSeries<Entry> timeSeries = store.putTimeSeries("ENTRIES", Entry.class);
        for (Entry entry : entries) {
            System.out.println(String.format("%s , %s", entry.createdAt, entry.value));
            timeSeries.put(entry);
        }
    }
}
