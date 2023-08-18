/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.djl.griddb.db;

import com.toshiba.mwcloud.gs.RowKey;
import java.util.Date;

/**
 *
 * @author ambag
 */
public class DB {

    static class Person {

        @RowKey
        String name;
        int age;
    }

    static class HeartRate {

        @RowKey
        Date ts;
        int heartRate;
        String activity;
    }
}
