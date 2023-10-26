package com.mycompany.djl.griddb.datasets;

import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.dataset.TimeFeaturizers;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.util.Progress;
import com.toshiba.mwcloud.gs.ColumnInfo;
import com.toshiba.mwcloud.gs.Container;
import com.toshiba.mwcloud.gs.GSException;
import com.toshiba.mwcloud.gs.GridStore;
import com.toshiba.mwcloud.gs.GridStoreFactory;
import com.toshiba.mwcloud.gs.Query;
import com.toshiba.mwcloud.gs.RowSet;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.time.LocalDateTime;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

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
        super.prepare(progress);
    }

    public int getDataLength() {
        return this.dataLength;
    }

    public static GridStore connectToGridDB() throws GSException {
        Properties props = new Properties();
        props.setProperty("notificationMember", "127.0.0.1:10001");
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

        public void setContainerName(String containerName) {
            this.containerName = containerName;
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
            File csvOutputFile = new File("out.csv");
            try ( GridStore store2 = store) {
                Container container = store2.getContainer(this.containerName);
                Query query = container.query("Select *");
                RowSet rowSet = query.getRowSet();
                dataLength = rowSet.size();
                int columnCount = rowSet.getSchema().getColumnCount(); 
                
                List<String> csv = new LinkedList<>();
                StringBuilder builder = new StringBuilder();
               
                while (rowSet.hasNext()) {
                    Object row = rowSet.next();
                    for(int i = 0; i < columnCount; i++){
                        ColumnInfo colInfo = rowSet.getSchema().getColumnInfo(i);
                        Field field = row.getClass().getDeclaredField(colInfo.getName());
                        builder.append(field.get(row));
                        if (i < columnCount - 1)
                            builder.append(",");
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
