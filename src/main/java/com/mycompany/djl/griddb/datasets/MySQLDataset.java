package com.mycompany.djl.griddb.datasets;

import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.basicdataset.tabular.utils.Featurizers;
import ai.djl.timeseries.dataset.CsvTimeSeriesDataset;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.training.dataset.Dataset;
import ai.djl.util.JsonUtils;
import ai.djl.util.Progress;
import com.toshiba.mwcloud.gs.GSException;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.csv.CSVFormat;
/**
 *
 * @author ambag
 */
public class MySQLDataset extends CsvTimeSeriesDataset {

    private final int dataLength;
    private final File csvFile;
    private boolean prepared;
    private List<Integer> cardinality;

    protected MySQLDataset(MySQLBBuilder builder) throws GSException, FileNotFoundException {
        super(builder);
        this.csvFile = builder.csvFile;
        this.dataLength = builder.dataLength;
        this.cardinality = builder.cardinality;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }
        csvUrl = this.csvFile.toURI().toURL();
        super.prepare(progress);
        prepared = true;
    }

    public int getDataLength() {
        return this.dataLength;
    }

    public List<Integer> getCardinality() {
        return cardinality;
    }

    public static Connection connectToMySQL() throws ClassNotFoundException, SQLException {
        Class.forName("com.mysql.cj.jdbc.Driver");
        String jdbcUrl = "jdbc:mysql://localhost:3306/dij";
        String username = "root";
        String password = "mysqlrootpass";
        return DriverManager.getConnection(jdbcUrl, username, password);
    }

    public static MySQLBBuilder gridDBBuilder() {
        MySQLBBuilder builder = null;
        try {
            builder = new MySQLBBuilder();
        } catch (ClassNotFoundException | SQLException ex) {
            Logger.getLogger(MySQLDataset.class.getName()).log(Level.SEVERE, null, ex);
        }
        return builder;
    }

    public static class MySQLBBuilder extends CsvBuilder<MySQLBBuilder> {

        Connection connection;
        public int dataLength = 0;
        File csvFile;
        M5Features mf;
        List<Integer> cardinality;

        MySQLBBuilder() throws ClassNotFoundException, SQLException {
            connection = connectToMySQL();
            csvFormat
                    = CSVFormat.DEFAULT
                            .builder()
                            .setHeader()
                            .setSkipHeaderRecord(true)
                            .setIgnoreHeaderCase(true)
                            .setTrim(true)
                            .build();
            cardinality = new ArrayList<>();

        }

        protected MySQLBBuilder self() {
            return this;
        }

        public MySQLBBuilder optStore(Connection connection) throws SQLException {
            this.connection.close();
            this.connection = connection;
            return this;
        }

        private File fetchDBDataAndSaveCSV(Connection con) throws Exception {
            File csvOutputFile = new File("out.csv");
            String sql = "Select * from entries";
            try ( Statement statement = con.createStatement();  ResultSet resultSet = statement.executeQuery(sql)) {
                List<String> csv = new LinkedList<>();
                while (resultSet.next()) {
                    csv.add(String.format("%s, %f", resultSet.getTimestamp("createdAt"), resultSet.getFloat("value")));
                }
                try ( PrintWriter pw = new PrintWriter(csvOutputFile)) {
                    csv.stream()
                            .forEach(pw::println);
                }
            }
            return csvOutputFile;
        }

        public MySQLBBuilder initData() throws FileNotFoundException, Exception {
            this.csvFile = fetchDBDataAndSaveCSV(this.connection);
            return this;
        }

        public MySQLBBuilder addFeature(String name, FieldName fieldName) {
            return addFeature(name, fieldName, false);
        }

        public MySQLBBuilder addFeature(String name, FieldName fieldName, boolean onehotEncode) {
            parseFeatures();
            if (mf.categorical.contains(name)) {
                Map<String, Integer> map = mf.featureToMap.get(name);
                if (map == null) {
                    return addFieldFeature(
                            fieldName,
                            new Feature(name, Featurizers.getStringFeaturizer(onehotEncode)));
                }
                cardinality.add(map.size());
                return addFieldFeature(fieldName, new Feature(name, map, onehotEncode));
            }
            return addFieldFeature(fieldName, new Feature(name, true));
        }

        private void parseFeatures() {
            if (mf == null) {
                try ( InputStream is
                        = M5Forecast.class.getResourceAsStream("m5forecast_parser.json"); 
                        Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                    mf = JsonUtils.GSON.fromJson(reader, M5Features.class);
                } catch (IOException e) {
                    throw new AssertionError(
                            "Failed to read m5forecast_parser.json from classpath", e);
                }
            }
        }

        public Dataset buildMySQLDataset() throws GSException {
            MySQLDataset gridDBDataset = null;
            try {
                gridDBDataset = new MySQLDataset(this);
            } catch (FileNotFoundException ex) {
                Logger.getLogger(MySQLDataset.class.getName()).log(Level.SEVERE, null, ex);
            }
            return gridDBDataset;
        }
    }

    private static final class M5Features {

        List<String> featureArray;
        Set<String> categorical;
        Map<String, Map<String, Integer>> featureToMap;
    }
}
