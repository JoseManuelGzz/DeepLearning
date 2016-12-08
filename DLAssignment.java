package org.deeplearning4j.examples.feedforward.mnist;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * Created by jmanu on 12/7/2016.
 */
public class DLAssignment {
    public static void main(String[] args) throws Exception{
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;
        int numInputs = 18;
        int numOutputs = 1;
        int numHiddenNodes = 10;

        //load the training data
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(File("full_train_csv_pre_processed.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,50,0,2);

        //load the test data
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(File("full_test_csv_pre_processed.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0,new DenseLayer.Builder()
                .nIn(numInputs)
                .nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                    .activation("relu")
                    .build())
            .layer(1,new DenseLayer.Builder()
                .nIn(numHiddenNodes)
                .nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                    .activation("relu")
                    .build())
            .layer(2,new DenseLayer.Builder()
                .nIn(numHiddenNodes)
                .nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                    .activation("relu")
                    .build())
            .layer(3,new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                    .activation("softmax")
                    .weightInit(WeightInit.XAVIER)
                    .nIn(numHiddenNodes)
                    .nOut(numOutputs)
                    .build())

            .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(1000));

            for(int i = 0; i < nEpochs; i++) {
                model.fit(trainIter);
            }

            System.out.println("Evaluating model...");
            Evaluation eval = new Evaluation(numOutputs);
            while(testIter.hasNext()) {
                DataSet t = testIter.next();
                INDArray features = t.getFeatureMatrix();
                INDArray labels = t.getLabels();
                INDArray predicted = model.output(features, false);
                eval.eval(labels, predicted);
            }

            System.out.println(eval.stats());



    }

}
