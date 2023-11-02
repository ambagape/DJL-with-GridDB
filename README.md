# djl-griddb

This project is the codebase of examples used in this article: https://gist.github.com/ambagape/c72b449cbb5c4e528361a84adec261e2. It's an example of how to integrate use a NoSQL db, such as GridDB, as the storage mechanism for a deep learning system.

# Getting Started
1. mvn clean install
2. mvn exec:java -Dexec.mainClass=com.mycompany.djl.griddb.Forecaster

# Requirements
1) You mast have GridDB running. Ensure you update the GridDBDatabase.java with values that correspond to your instance of GridDB before compiling and running this example

You can use any IDE or code editing tool for developing on any platform. Use your favorite!

