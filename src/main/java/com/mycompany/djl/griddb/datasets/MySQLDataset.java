package com.mycompany.djl.griddb.datasets;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import com.toshiba.mwcloud.gs.GSException;
import java.io.File;
import java.io.FileNotFoundException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
/**
 *
 * @author ambag
 */
public class MySQLDataset extends ArrayDataset {

    private boolean prepared;
    NDManager manager = NDManager.newBaseManager();

    protected MySQLDataset(MySQLBBuilder builder) throws GSException, FileNotFoundException {
        super(new ArrayDataset.Builder()
                .setData(builder.data[0])
                .optLabels(builder.data[1])
                .setSampling(builder.batchSize, builder.shuffle));
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

    public static class MySQLBBuilder {

        Connection connection;
        NDArray[] data;
        int batchSize;
        boolean shuffle;

        MySQLBBuilder() throws ClassNotFoundException, SQLException {
            connection = connectToMySQL();            
        }
        
        protected MySQLBBuilder self() {
            return this;
        }

        public MySQLBBuilder optStore(Connection connection) throws SQLException {
            this.connection.close();
            this.connection = connection;
            return this;
        }

        private NDArray[] fetchDBDataAndSaveCSV(Connection con) throws Exception {
            NDManager manager = NDManager.newBaseManager();
            List<Long> features =  new ArrayList<>();
            List<Float> labels = new ArrayList<>();
            
            String sql = "Select * from entries";
            try ( Statement statement = con.createStatement();  ResultSet resultSet = statement.executeQuery(sql)) {
                
                while (resultSet.next()) {
                    features.add(resultSet.getTimestamp("createdAt").getTime());
                    labels.add(resultSet.getFloat("value"));
                }               
            }
            float[] arrayLabels = new float[labels.size()];
            long[] arrayFeatures = new long[features.size()];
            for(int i=0; i < features.size(); i++){
                arrayLabels[i] = labels.get(i);
                arrayFeatures[i] = features.get(i);
            }
                
            return new NDArray[] {manager.create(arrayFeatures), manager.create(arrayLabels)};
        }
        
        public MySQLBBuilder setSampling(int batchSize, boolean shuffle){
            this.batchSize = batchSize;
            this.shuffle = shuffle;
            return this;
        }

        public MySQLBBuilder initData() throws FileNotFoundException, Exception {
            this.data = fetchDBDataAndSaveCSV(this.connection);
            return this;
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

}
