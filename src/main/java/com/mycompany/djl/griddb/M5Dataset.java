/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.djl.griddb;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.tabular.utils.DynamicBuffer;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

/**
 *
 * @author ambag
 */
public class M5Dataset implements Iterable<NDList>, Iterator<NDList> {

    private NDManager manager;
    private List<Feature> target;
    private List<CSVRecord> csvRecords;
    private long size;
    private long current;

    M5Dataset(Builder builder) {
        manager = builder.manager;
        target = builder.target;
        try {
            prepare(builder);
        } catch (Exception e) {
            throw new AssertionError("Failed to read files.", e);
        }
        size = csvRecords.size();
    }

    private void prepare(Builder builder) throws IOException {
        MRL mrl = builder.getMrl();
        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, null);

        Path root = mrl.getRepository().getResourceDirectory(artifact);
        Path csvFile = root.resolve("weekly_sales_train_evaluation.csv");

        URL csvUrl = csvFile.toUri().toURL();
        try ( Reader reader
                = new InputStreamReader(
                        new BufferedInputStream(csvUrl.openStream()), StandardCharsets.UTF_8)) {
            CSVParser csvParser = new CSVParser(reader, builder.csvFormat);
            csvRecords = csvParser.getRecords();
        }
    }

    @Override
    public boolean hasNext() {
        return current < size;
    }

    @Override
    public NDList next() {
        NDList data = getRowFeatures(manager, current, target);
        current++;
        return data;
    }

    public static Builder builder() {
        return new Builder();
    }

    private NDList getRowFeatures(NDManager manager, long index, List<Feature> selected) {
        DynamicBuffer bb = new DynamicBuffer();
        for (Feature feature : selected) {
            String name = feature.getName();
            String value = getCell(index, name);
            feature.getFeaturizer().featurize(bb, value);
        }
        FloatBuffer buf = bb.getBuffer();
        return new NDList(manager.create(buf, new Shape(bb.getLength())));
    }

    private String getCell(long rowIndex, String featureName) {
        CSVRecord record = csvRecords.get(Math.toIntExact(rowIndex));
        return record.get(featureName);
    }

    @Override
    public Iterator<NDList> iterator() {
        return this;
    }

    public static final class Builder {

        NDManager manager;
        List<Feature> target;
        CSVFormat csvFormat;

        Repository repository;
        String groupId;
        String artifactId;
        String version;

        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = "m5forecast-unittest";
            version = "1.0";
            csvFormat
                    = CSVFormat.DEFAULT
                            .builder()
                            .setHeader()
                            .setSkipHeaderRecord(true)
                            .setIgnoreHeaderCase(true)
                            .setTrim(true)
                            .build();
            target = new ArrayList<>();
            for (int i = 1; i <= 277; i++) {
                target.add(new Feature("w_" + i, true));
            }
        }

        public Builder optRepository(Repository repository) {
            this.repository = repository;
            return this;
        }

        public Builder setManager(NDManager manager) {
            this.manager = manager;
            return this;
        }

        public M5Dataset build() {
            return new M5Dataset(this);
        }

        MRL getMrl() {
            return repository.dataset(Application.Tabular.ANY, groupId, artifactId, version);
        }
    }
}
