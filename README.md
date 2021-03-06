![Hillview project logo](hillview-logo.png)

# Introduction

Hillview: a big data spreadsheet.  Hillview is a cloud-based
application for browsing large datasets.  It is under active
development.  Currently the software is in pre-alpha version.  For
more information see the [Hillview user manual](docs/userManual.md).

The hillview user interface executes in a browser.  The browser can
display multiple views of the data, as in the following screenshots.
These screenshots were obtained by browsing a dataset containing more
than 120 million rows, representing all passenger airline flights in
the United States between 1997 and 2016.

The first image shows a tabular display of the data; the user can
choose a subset of the columns to display, and can sort on any
combination of columns.  In this image the data is sorted in
decreasing order on the flight distance.

![A table view](docs/hillview-table.png)

The second image shows a histogram of the data in one column, in this
case the flight departure delay.

![A one dimensional histogram](docs/hillview-histogram.png)

The third image shows a two-dimensional histogram, where the X axis is
the duration of a flight and where each bar is color-coded according
to the distance travelled.

![A two-dimensional histogram](docs/hillview-histogram2d.png)

The fourth image shows another display of a two-dimensional histogram,
where the X axis is the flight date and where each bar is color-coded
according to its delay.

![A normalized two-dimensional histogram](docs/hillview-histogram-normalized.png)

The fifth image shows a heatmap, where the X axis is the departure
delay, the Y axis is the arrival delay, and the color shows the number
of flights that fall into each pixel, with hotter colors showing more
flights.

![A heatmap](docs/hillview-heatmap.png)

Here is a [short
  video](https://1drv.ms/v/s!AlywK8G1COQ_jbF-il8AQvIkOdW2dw) of the
  system in action.

# Developing Hillview

## Software Dependences

* Back-end: Ubuntu Linux or MacOS
* Java 8, Maven build system, various Java libraries
  (Maven will manage the libraries)
* Front-end: Typescript, webpack, Tomcat app server, node.js;
  some JavaScript libraries: d3 and rx-js
* Cloud service management: Ansible
* IDEA Intellij for development (optional)

## Project structure

Hillview is currently split into two separate Maven projects.

* platform: pure Java, includes the entire back-end.  `platform` can be
developed using the free (community edition) of Intellij IDEA.

* web: the web server, web client and web services; this project links
to the result produced by the `platform` project.  To develop and
debug this we have used capabilities available only in the paid
version of Intellij, Ultimate, but only Maven is needed to build.

## Single-machine development and testing

These instructions describe how to run hillview on a single machine
using a sample dataset.

* First install all software required as described
  [below](#installing-the-software-needed).

* Download the sample data.  The download script will download and
  decompress some CSV files with flights data from FAA.  You can edit
  the script to change the range of data that will be downloaded; the
  default is to download 2 months of data.

```
$ cd data
$ ./download.sh
$ cd ..
```

If you run out of memory while doing any of the following try to
increase the java heap size as follows (you can put this in your .bashrc file):

```
$ export MAVEN_OPTS="-Xmx2048M"
```

* Install the distributed platform library:

```
$ cd platform
$ mvn install
```

* Build the web server and the front-end

```
$ cd web
$ mvn package
```

* The dataset has 110 columns; we can use them all, but for the demo
  we have stripped the dataset to 15 columns to better fit on the
  screen.  The following command creates the smaller files from the
  downloaded data; this has to be done only once, after downloading
  the data.

```
$ cd bin
$ ./demo-data-cleaner.sh
```

* Next start the back-end service which performs all the data
  processing:

```
$ ./demo-backend-start.sh &
```

* Start the web server which receives requests from clients and
dispatches them to the back-end servers; note that the folder where
this command is run is important, since the path to the data files is
relative to this folder.

```
$ ./demo-frontend-start.sh
```

* start a web browser at http://localhost:8080 and browse the data!

* when you are done stop the two services by killing the scripts.

## Deploying the Hillview service

Instructions for managing the Hillview cloud service are given
[here](deployment/README.md).

## Contributing code

### Setup IntelliJ IDEA

Download and install Intellij IDEA: https://www.jetbrains.com/idea/.
You can just untar the linux binary in a place of your choice and run
the shell script `ideaXXX/bin/idea.sh`.  The web projects uses
capabilities only available in the paid version of Intellij IDEA.

### Loading into IntelliJ IDEA

To load the project that you want to contribute to, move to the
corresponding folder: `cd platform` or `cd web` and start
intellij there.

The first time you start Intellij you must import the project: on the
welcome screen, select the "import project" option, point to the
"pom.xml" file inside the hillview folder, click "next" a few times and
you're good to go.

### Using git to contribute

* Fork the repository using the "fork" button on github, by following these instructions:
https://help.github.com/articles/fork-a-repo/
* Run IntelliJ code inspection (Analyze/Inspect code) before commit and solve all open issues.
* Submit them into your own forked repository and send us a pull request.

In more detail, here is a step-by-step guide to committing your changes:

1. Create a new branch for each fix; give it a nice suggestive name:
   - `git branch yourBranchName`
   - `git checkout yourBranchName`
   - The main benefit of using branches is that you can have multiple branches active at the same time, one for each independent fix.
2. `git add <files that changed>`
3. `git commit -m "Description of commit"`
4. `git fetch upstream`
5. `git rebase upstream/master`
6. Resolve conflicts, if any (rebase won't work if you don't; as you find conflicts you will need to `git add` the files you have merged, and then you may need to use `git rebase --continue` or `git rebase --skip`)
7. Test, analyze merged version.
8. `git push -f origin yourBranchName`
9. Create a pull request to merge your new branch into master (using the web ui).
10. Delete your branch after the merging has been done `git branch -D yourBranchName`
11. To run the program you should try the master branch:
  - `git checkout master`
  - `git fetch upstream`
  - `git rebase upstream/master`
  - `git push origin master`

### Guidance in writing code

* The pseudorandom generator is implemented in the class
  Randomness.java and uses Mersenne Twister.  Do not use the
  Java `Random` class, but this one.

* By default all pointers are assumed to be non-null; use the
  @Nullable annotation (from javax.annotation) for all pointers which
  can be null.  Use `Converters.checkNull` to cast a @Nullable to a
  @NonNull pointer.

* (optional) Use "mvn site" to generate the FindBugs report in
  target/site/findbugs.html.  Make sure any new code checked in does
  not introduce any violations.  A subset of these checks is also
  done by the IDEA code inspection tool.

## Installing the software needed

Install Maven, node.js and ansible:

```
$ sudo apt-get install maven nodejs-legacy ansible npm
```

### Installing Java

We use Java 8.

First, download a JDK for Linux x64 from here:
http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

Make sure to download the tarball version of the JDK.

Unpack the JDK, and set your `JAVA_HOME` environment variable to point
to the unpacked folder (e.g, <fully qualified path
to>/jdk/jdk1.8.0_101). To set your JAVA_HOME environment variable, add
the following to your ~/.bashrc or ~/.zshrc.

```
$ export JAVA_HOME="<path-to-jdk-folder>"
```

## Install Apache Tomcat web application server

The instructions here use version 8.5.8 (other versions may work,
but we only tested this one).

```
$ wget http://archive.apache.org/dist/tomcat/tomcat-8/v8.5.8/bin/apache-tomcat-8.5.8.tar.gz
$ tar xvfz apache-tomcat-8.5.8.tar.gz
$ cd apache-tomcat-8.5.8/webapps
$ rm -rf ROOT*
$ ln -s ../../web/target/web-1.0-SNAPSHOT.war ROOT.war
$ cd ../..
```

## Install typescript and JavaScript libraries and tools

On Mac the following command seems to work correctly only without `sudo`.

```
$ sudo npm install -g typescript ts-loader webpack
```

This installs the typescript compiler, the `webpack` tool, used to
bundle multiple JavaScript files together, the `ts-loader` tool for
webpack, which allows it to compile directly typescript into
javascript.

Then install JavaScript libraries `rx` and `d3`, together with the
typescript type definitions for these libraries:

```
$ cd web/src/main/webapp
$ npm install rx d3
$ npm install @types/d3 --save
$ cd ../../../..
```
