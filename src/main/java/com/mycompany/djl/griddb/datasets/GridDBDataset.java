package com.mycompany.djl.griddb.datasets;

import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.dataset.TimeFeaturizers;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.training.dataset.Dataset;
import ai.djl.util.Progress;
import com.mycompany.djl.griddb.Forecaster;
import com.opencsv.CSVReader;
import com.toshiba.mwcloud.gs.ColumnInfo;
import com.toshiba.mwcloud.gs.Container;
import com.toshiba.mwcloud.gs.ContainerInfo;
import com.toshiba.mwcloud.gs.ContainerType;
import com.toshiba.mwcloud.gs.GSException;
import com.toshiba.mwcloud.gs.GSType;
import com.toshiba.mwcloud.gs.GridStore;
import com.toshiba.mwcloud.gs.GridStoreFactory;
import com.toshiba.mwcloud.gs.Query;
import com.toshiba.mwcloud.gs.Row;
import com.toshiba.mwcloud.gs.RowSet;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;
import org.apache.commons.csv.CSVParser;

/**
 *
 * @author ambag
 */
public class GridDBDataset extends M5Forecast {

    final static String TRAINING_COLLECTION_NAME = "NNTraining";
    final static String VALIDATION_COLLECTION_NAME = "NNValidation";

    private final File csvFile;
    static private M5Forecast.Builder forecastBuilder;

    protected GridDBDataset(GridDBBuilder builder) throws GSException, FileNotFoundException {
        super(initializeParent(builder));
        this.csvFile = builder.csvFile;
    }

    static M5Forecast.Builder initializeParent(GridDBBuilder builder) {
        M5Forecast.Builder newBuilder = M5Forecast.builder()
                .optUsage(builder.getUsage())
                .setTransformation(builder.getTransformation())
                .setContextLength(builder.getContextLength())
                .setSampling(builder.getSize(), builder.isRandom())
                .optCsvFile(builder.csvFile.toPath());

        for (int i = 1; i <= builder.getMaxWeek(); i++) {
            newBuilder.addFeature("w_" + i, FieldName.TARGET);
        }

        newBuilder.addFeature("state_id", FieldName.FEAT_STATIC_CAT)
                .addFeature("store_id", FieldName.FEAT_STATIC_CAT)
                .addFeature("cat_id", FieldName.FEAT_STATIC_CAT)
                .addFeature("dept_id", FieldName.FEAT_STATIC_CAT)
                .addFeature("item_id", FieldName.FEAT_STATIC_CAT)
                .addFieldFeature(
                        FieldName.START,
                        new Feature(
                                "date",
                                TimeFeaturizers.getConstantTimeFeaturizer(builder.getStartTime())));

        GridDBDataset.forecastBuilder = newBuilder;
        return newBuilder;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void prepare(Progress progress) throws IOException {
        csvUrl = this.csvFile.toURI().toURL();
        try ( Reader reader = new InputStreamReader(getCsvStream(), StandardCharsets.UTF_8)) {
            CSVParser csvParser = new CSVParser(reader, csvFormat);
            csvRecords = csvParser.getRecords();
        }
        prepareFeaturizers();
    }

    private InputStream getCsvStream() throws IOException {
        if (csvUrl.getFile().endsWith(".gz")) {
            return new GZIPInputStream(csvUrl.openStream());
        }
        return new BufferedInputStream(csvUrl.openStream());
    }

    public static GridStore connectToGridDB() throws GSException {
        Properties props = new Properties();
        props.setProperty("notificationMember", "172.18.0.2:10001");
        props.setProperty("clusterName", "defaultCluster");
        props.setProperty("user", "admin");
        props.setProperty("password", "admin");
        return GridStoreFactory.getInstance().getGridStore(props);
    }

    public static GridDBBuilder gridDBBuilder() throws Exception {
        GridDBBuilder builder = null;
        try {
            builder = new GridDBBuilder();
        } catch (GSException ex) {
            Logger.getLogger(GridDBDataset.class.getName()).log(Level.SEVERE, null, ex);
        }
        return builder;
    }

    public static class GridDBBuilder {

        GridStore store;
        public int dataLength = 0;
        File csvFile;
        private Usage usage;
        private List<TimeSeriesTransform> transformation;
        private int contextLength;
        private boolean random;
        private int size;
        private LocalDateTime startTime;
        private int maxWeek;

        GridDBBuilder() throws GSException, Exception {
            store = connectToGridDB();
            seedDatabase();
        }

        protected GridDBBuilder self() {
            return this;
        }

        public GridDBBuilder optStore(GridStore store) throws GSException {
            this.store.close();
            this.store = store;
            return this;
        }

        public GridDBBuilder setSize(int size) {
            this.size = size;
            return this;
        }

        public GridDBBuilder setTransformation(List transforms) {
            this.transformation = transforms;
            return this;
        }

        public GridDBBuilder setContextLength(int size) {
            this.contextLength = size;
            return this;
        }

        public GridDBBuilder setRandom(boolean random) {
            this.random = random;
            return this;
        }

        public GridDBBuilder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        public GridDBBuilder setStartTime(LocalDateTime startTime) {
            this.startTime = startTime;
            return this;
        }

        public String getContainerName() {
            return usage == Dataset.Usage.TRAIN ? TRAINING_COLLECTION_NAME : VALIDATION_COLLECTION_NAME;
        }

        public int getMaxWeek() {
            return maxWeek;
        }

        public GridDBBuilder setMaxWeek(int maxWeek) {
            this.maxWeek = maxWeek;
            return this;
        }

        public boolean isRandom() {
            return random;
        }

        public File getCsvFile() {
            return csvFile;
        }

        public Usage getUsage() {
            return usage;
        }

        public List<TimeSeriesTransform> getTransformation() {
            return transformation;
        }

        public int getContextLength() {
            return contextLength;
        }

        public LocalDateTime getStartTime() {
            return startTime;
        }

        public int getSize() {
            return size;
        }

        /*
    We assume the database is already containing the timeseries data
         */
        private static void seedDatabase() throws Exception {
            URL trainingData = Forecaster.class.getClassLoader().getResource("data/weekly_sales_train_validation.csv");
            URL validationData = Forecaster.class.getClassLoader().getResource("data/weekly_sales_train_evaluation.csv");
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

        private File fetchDBDataAndSaveCSV(GridStore store) throws GSException, FileNotFoundException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
            File csvOutputFile = new File(this.getContainerName()+ ".csv");
            try ( GridStore store2 = store) {
                Container container = store2.getContainer(this.getContainerName());

                Query query = container.query("Select *");
                RowSet<Row> rowSet = query.fetch();

                int columnCount = rowSet.getSchema().getColumnCount();

                List<String> csv = new LinkedList<>();
                StringBuilder builder = new StringBuilder();

                //Loan column headers
                ContainerInfo cInfo = rowSet.getSchema();
                for (int i = 0; i < cInfo.getColumnCount(); i++) {
                    ColumnInfo columnInfo = rowSet.getSchema().getColumnInfo(i);
                    builder.append(columnInfo.getName());
                    appendComma(builder, i, cInfo.getColumnCount());
                }
                csv.add(builder.toString());

                //Load each row
                while (rowSet.hasNext()) {
                    Row row = rowSet.next();
                    builder = new StringBuilder();
                    for (int i = 0; i < columnCount; i++) {
                        String val = row.getString(i);
                        builder.append(val);
                        appendComma(builder, i, columnCount);
                    }
                    csv.add(builder.toString());
                }
                try ( PrintWriter pw = new PrintWriter(csvOutputFile)) {
                    csv.stream()
                            .forEach(pw::println);
                }
            }
            return csvOutputFile;
        }

        private static void appendComma(StringBuilder builder, int columnIndex, int length) {
            if (columnIndex < length - 1) {
                builder.append(",");
            }
        }

        public GridDBBuilder initData() throws GSException, FileNotFoundException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
            this.csvFile = fetchDBDataAndSaveCSV(this.store);
            return this;
        }

        public GridDBDataset build() {
            GridDBDataset gridDBDataset = null;
            try {
                gridDBDataset = new GridDBDataset(this);
            } catch (GSException | FileNotFoundException ex) {
                Logger.getLogger(GridDBDataset.class.getName()).log(Level.SEVERE, null, ex);
            }
            return gridDBDataset;
        }
    }
}
