package com.mycompany.djl.griddb.datasets;

import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.util.Progress;
import com.mycompany.djl.griddb.db.DB;
import com.toshiba.mwcloud.gs.GSException;
import com.toshiba.mwcloud.gs.GridStore;
import com.toshiba.mwcloud.gs.GridStoreFactory;
import com.toshiba.mwcloud.gs.Query;
import com.toshiba.mwcloud.gs.RowSet;
import com.toshiba.mwcloud.gs.TimeSeries;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
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

    protected GridDBDataset(GridDBBuilder builder) throws GSException, FileNotFoundException {
        super(M5Forecast.builder()
                .optUsage(builder.usage)
                .optCsvFile(builder.csvFile.toPath()));                
        this.csvFile = builder.csvFile;
        this.dataLength = builder.dataLength;
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

    public static class GridDBBuilder extends CsvBuilder {

        GridStore store;
        public int dataLength = 0;
        File csvFile;
        Usage usage;        

        GridDBBuilder() throws GSException {
            store = connectToGridDB();
        }

        @Override
        protected GridDBBuilder self() {
            return this;
        }

        public GridDBBuilder optStore(GridStore store) throws GSException {
            this.store.close();
            this.store = store;
            return this;
        }

        @Override
        public GridDBBuilder addFieldFeature(FieldName name, Feature feature) {
            super.addFieldFeature(name, feature);
            return this;
        }
        
        @Override
        public GridDBBuilder setTransformation(List transforms) {
            super.setTransformation(transforms);
            return this;
        }      
        
        public GridDBBuilder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }
        
        public void  addFeature(String name,  FieldName fieldName){
            addFieldFeature(fieldName, new Feature(name, false));
        }        
         
        @Override
        public GridDBBuilder setSampling(int size, boolean isRandom) {
            super.setSampling(size, isRandom);
            return this;
        }

        @Override
        public GridDBBuilder setContextLength(int size) {
            super.setContextLength(size);
            return this;
        }

        private File fetchDBDataAndSaveCSV(GridStore store) throws GSException, FileNotFoundException {
            File csvOutputFile = new File("out.csv");
            try ( GridStore store2 = store) {
                TimeSeries<DB.Entry> entries = store2.getTimeSeries("ENTRIES", DB.Entry.class);
                Query<DB.Entry> query = entries.query("Select * from ENTRIES");
                RowSet<DB.Entry> rowSet = query.getRowSet();
                dataLength = rowSet.size();
                List<String> csv = new LinkedList<>();
                while (rowSet.hasNext()) {
                    csv.add(String.format("%s, %f", rowSet.next().createdAt, rowSet.next().value));
                }
                try ( PrintWriter pw = new PrintWriter(csvOutputFile)) {
                    csv.stream()
                            .forEach(pw::println);
                }
            }
            return csvOutputFile;
        }

        public GridDBBuilder initData() throws GSException, FileNotFoundException {
            this.csvFile = fetchDBDataAndSaveCSV(this.store);
            return this;
        }

        @Override
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
