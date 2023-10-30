package com.mycompany.djl.griddb.datasets;

import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.dataset.TimeFeaturizers;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.util.Progress;
import com.toshiba.mwcloud.gs.ColumnInfo;
import com.toshiba.mwcloud.gs.Container;
import com.toshiba.mwcloud.gs.ContainerInfo;
import com.toshiba.mwcloud.gs.GSException;
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
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
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

    private final int dataLength;
    private final File csvFile;
    static private M5Forecast.Builder forecastBuilder;

    protected GridDBDataset(GridDBBuilder builder) throws GSException, FileNotFoundException {
        super(initializeParent(builder));
        this.csvFile = builder.csvFile;
        this.dataLength = builder.dataLength;
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

    public int getDataLength() {
        return this.dataLength;
    }

    public static GridStore connectToGridDB() throws GSException {
        Properties props = new Properties();
        props.setProperty("notificationMember", "172.18.0.3:10001");
        props.setProperty("clusterName", "defaultCluster");
        props.setProperty("user", "admin");
        props.setProperty("password", "admin");
        return GridStoreFactory.getInstance().getGridStore(props);
    }

    public static GridDBBuilder gridDBBuilder() {
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
        private String containerName;

        GridDBBuilder() throws GSException {
            store = connectToGridDB();
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
            return containerName;
        }

        public GridDBBuilder setContainerName(String containerName) {
            this.containerName = containerName;
            return this;
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

        private File fetchDBDataAndSaveCSV(GridStore store) throws GSException, FileNotFoundException, NoSuchFieldException, IllegalArgumentException, IllegalAccessException {
            File csvOutputFile = new File(this.containerName+".csv");
            try ( GridStore store2 = store) {
                Container container = store2.getContainer(this.containerName);

                Query query = container.query("Select *");
                RowSet<Row> rowSet = query.fetch();

                dataLength = rowSet.size();
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
