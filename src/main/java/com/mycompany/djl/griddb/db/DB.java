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
    
     public static class Entry {
        @RowKey
        public Date createdAt;
        public double value;
    }
}
