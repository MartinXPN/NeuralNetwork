
#ifndef NEURALNETWORK_FULLYCONNECTEDTESTDATAMININGPROJECT_H
#define NEURALNETWORK_FULLYCONNECTEDTESTDATAMININGPROJECT_H

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <fstream>
#include "../../library/neurons/Bias.h"
#include "../../library/layers/InputLayer.h"
#include "../../library/layers/FullyConnected.h"
#include "../../library/activations/Sigmoid.h"
#include "../../library/activations/ReLU.h"
#include "../../library/layers/LossLayer.h"
#include "../../library/lossfunctions/CrossEntropyCost.h"


using namespace std;

//////////////////////// NETWORK CONSTRUCTION//////////////////////////////////
Bias <double> *bias = new Bias <double>();
InputLayer <double> in( {143} );
FullyConnected <double> fc1( {32}, new Sigmoid <double>(), {&in}, bias );
FullyConnected <double> fc2( {8}, new ReLU <double>(), {&fc1}, bias );
LossLayer <double> out( {1}, {&fc2}, new CrossEntropyCost <double>(), new Sigmoid <double>(), bias );



vector <string> split(const string &s, char delimiter) {
    vector <string> res;
    stringstream ss;
    ss.str(s);
    string item;
    while (getline(ss, item, delimiter)) {
        res.push_back(item);
    }

    return res;
}
vector< vector <double> > readTest( string fileDirectory ) {
    ifstream file ( fileDirectory );

    /// get names of the variables in the data-frame
    string first_line;
    getline( file, first_line );
    vector <string> names = split( first_line, ',' );


    vector< vector <double> > data;
    for( int i=0; i < 20352; ++i ) {
        vector <double> variables;
        double var;
        for( int j=0; j < names.size()-1; ++j ) {
            char comma;
            file >> var >> comma;
            variables.push_back( var );
        }
        file >> var;
        variables.push_back( var );
        data.push_back( variables );
    }

    return data;
}

///        data                     labels
pair< vector< vector <double> >, vector <double> > readTrain( string fileDirectory ) {
    ifstream file ( fileDirectory );

    /// get names of the variables in the data-frame
    string first_line;
    getline( file, first_line );
    vector <string> names = split( first_line, ',' );


    vector< vector <double> > data;
    vector <double> labels;
    for( int i=0; i < 81414; ++i ) {
        vector <double> variables;
        double var;
        for( int j=0; j < names.size()-1; ++j ) {

            char comma;
            if( names[j] == "Class" ) {
                string label;
                char c;
                while( true ) {
                    file >> c;
                    if( c != ',' )  label += c;
                    else            break;
                }
                labels.push_back( label == "YES" ? 1 : 0 );
            }
            else {
                file >> var >> comma;
                variables.push_back(var);
            }
        }
        file >> var;
        variables.push_back( var );
        data.push_back( variables );
    }

    return {data, labels};
}

double evaluateOne( const vector <double> &v ) {

    for( int j=0; j < v.size(); ++j )
        ((BaseInputNeuron <double>*)in.getNeurons()[j]) -> setValue( v[j] );

    /// activate neurons
    for (auto neuron : fc1.getNeurons())    neuron->activateNeuron();
    for (auto neuron : fc2.getNeurons())    neuron->activateNeuron();
//    for (auto neuron : fc3.getNeurons())    neuron->activateNeuron();
//    for (auto neuron : fc4.getNeurons())    neuron->activateNeuron();
    for (auto neuron : out.getNeurons())    neuron->activateNeuron();
    return ((BaseOutputNeuron <double> *)out.getNeurons()[0])->getValue();
}

void evaluateTest( int epoch, const vector< vector <double> >& testData ) {

    printf( "\nEvaluating on TESTING dataset" );
    vector <double> testLabels;
    for( int i=0; i < testData.size(); ++i ) {
        testLabels.push_back( evaluateOne( testData[i] ) );
    }

    printf( "\nSaving results to file...\n" );
    ofstream fout;
    fout.open ("/home/martin/Desktop/Projects/DataMining/neural_network:(143)->(32)->(8)->(1)_epoch_" + to_string(epoch) + ".csv");
    fout << "ID,Class\n";
    for( int i=0; i < testLabels.size(); ++i ) {
        fout << i+1 << ',' << testLabels[i] << '\n';
    }
    fout.close();
}

void testDataMiningProject() {

    //////////////READ THE DATA////////////////////////////
    const double trainProportion = 0.9;
    auto testData = readTest( "/home/martin/Desktop/Projects/DataMining/filtered_scaled_test.csv" );
//    printf( "TEST:\n" );
//    for( int i=0; i < testData.size(); ++i, cout << endl )
//        for( int j=0; j < testData[i].size(); ++j ) {
//            cout << testData[i][j] << " ";
//        }

    printf( "\n\nLast row of TEST dataset-->\t" );
    for( int j=0; j < testData.back().size(); ++j ) {
        cout << testData.back()[j] << " ";
    }

    /// train
    auto trainData = readTrain( "/home/martin/Desktop/Projects/DataMining/filtered_scaled_train.csv" );
    vector< pair <vector <double>, double> > train;
    for( int i=0; i < trainData.first.size(); ++i )
        train.push_back( { trainData.first[i], trainData.second[i] } );

    random_shuffle( train.begin(), train.end() );
    vector< vector <double> > trainInputs;
    vector< vector <double> > validInputs;
    vector <double> trainLabels;
    vector <double> validLabels;
    for( int i=0; i < train.size() * trainProportion; ++i )                 trainInputs.push_back( train[i].first );
    for( int i=0; i < train.size() * trainProportion; ++i )                 trainLabels.push_back( train[i].second );
    for( int i=train.size() * trainProportion; i < train.size(); ++i )      validInputs.push_back( train[i].first );
    for( int i=train.size() * trainProportion; i < train.size(); ++i )      validLabels.push_back( train[i].second );


//    printf( "\n\nTRAIN:\n" );
//    for( int i=0; i < trainData.size(); ++i, cout << endl ) {
//        cout << trainLabels[i] << "\t";
//        for( int j=0; j < trainData[i].size(); ++j ) {
//            cout << trainData[i][j] << " ";
//        }
//    }

    printf( "\nLast row of TRAIN dataset-->\t" );
    for( int j=0; j < trainInputs.back().size(); ++j ) {
        cout << trainInputs.back()[j] << " ";
    }
    printf( "\n---> label: %lf\n", trainLabels.back() );

    //////////////////////// INITIALIZATION //////////////////////////////
    printf( "\nNetwork: (%d) -> (%d) -> (%d) -> (%d) \n\n", in.size(), fc1.size(), fc2.size(), out.size() );
    in.createNeurons();
    fc1.createNeurons();
    fc2.createNeurons();
//    fc3.createNeurons();
//    fc4.createNeurons();
    out.createNeurons();

    fc1.createWeights();
    fc2.createWeights();
//    fc3.createWeights();
//    fc4.createWeights();

    fc1.connectNeurons();
    fc2.connectNeurons();
//    fc3.connectNeurons();
//    fc4.connectNeurons();
    out.connectNeurons();

    auto inputNeurons = in.getNeurons();
    auto outputNeurons = out.getNeurons();

    ////////////////////// TRAINING //////////////////////////////////////
    const int maxEpochs = 5000;
    const int batchSize = 200;
    const double initialLearningRate = 0.1;
    const double learningRateDecay = 0.7;
    const double minLearningRate = 0.00005;
    const double restoreLearningRate = 0.01;
    double bestValidationLoss = 1e9;

    double learningRate = initialLearningRate;
    for( int epoch = 0; epoch < maxEpochs; ++epoch ) {

        double epochTrainLoss = 0;

        printf( "\n--Epoch: (%d)--Learning rate (%lf)--", epoch, learningRate );

        for (int batch = 0; batch < trainInputs.size(); batch += batchSize) {
            double batchLoss = 0;
            for (int i = batch; i < batch + batchSize && i < trainInputs.size(); ++i) {
                /// set values of input neurons
                for( int j=0; j < trainInputs[i].size(); ++j )
                    ((BaseInputNeuron <double>*)inputNeurons[j]) -> setValue( trainInputs[i][j] );

                /// activate neurons
                for (auto neuron : fc1.getNeurons())    neuron->activateNeuron();
                for (auto neuron : fc2.getNeurons())    neuron->activateNeuron();
//                for (auto neuron : fc3.getNeurons())    neuron->activateNeuron();
//                for (auto neuron : fc4.getNeurons())    neuron->activateNeuron();
                for( auto neuron : outputNeurons )      neuron->activateNeuron();

                /// calculate losses
                for( int j=0; j < outputNeurons.size(); ++j ) {
                    ((BaseOutputNeuron <double>*)outputNeurons[j]) -> calculateLoss( trainLabels[i] );
                    batchLoss += fabs( ((BaseOutputNeuron <double>*)outputNeurons[j]) -> getError( trainLabels[i] ) );
                }
//                for (auto neuron : fc4.getNeurons())    neuron->calculateLoss();
//                for (auto neuron : fc3.getNeurons())    neuron->calculateLoss();
                for (auto neuron : fc2.getNeurons())    neuron->calculateLoss();
                for (auto neuron : fc1.getNeurons())    neuron->calculateLoss();

                /// backpropagate neurons
                for (auto neuron : outputNeurons)       neuron->backpropagateNeuron();
//                for (auto neuron : fc4.getNeurons())    neuron->backpropagateNeuron();
//                for (auto neuron : fc3.getNeurons())    neuron->backpropagateNeuron();
                for (auto neuron : fc2.getNeurons())    neuron->backpropagateNeuron();
                for (auto neuron : fc1.getNeurons())    neuron->backpropagateNeuron();
            }

            epochTrainLoss += batchLoss;
//            cout << "Loss #" << batch << ": " << batchLoss / batchSize << endl;

            /// update weights
            for (auto neuron : outputNeurons)       neuron->updateWeights(learningRate, batchSize);
//            for (auto neuron : fc4.getNeurons())    neuron->updateWeights(learningRate, batchSize);
//            for (auto neuron : fc3.getNeurons())    neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : fc2.getNeurons())    neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : fc1.getNeurons())    neuron->updateWeights(learningRate, batchSize);
        }


        /// evaluate on validation dataset
        double validLoss = 0;
        for( int i=0; i < validInputs.size(); ++i ) {
            /// set values of input neurons
            for (int j = 0; j < validInputs[i].size(); ++j)
                ((BaseInputNeuron<double> *) inputNeurons[j])->setValue(trainInputs[i][j]);

            /// activate neurons
            for (auto neuron : fc1.getNeurons()) neuron->activateNeuron();
            for (auto neuron : fc2.getNeurons()) neuron->activateNeuron();
//            for (auto neuron : fc3.getNeurons()) neuron->activateNeuron();
//            for (auto neuron : fc4.getNeurons()) neuron->activateNeuron();
            for (auto neuron : outputNeurons) neuron->activateNeuron();

            /// calculate losses
            for (int j = 0; j < outputNeurons.size(); ++j) {
                ((BaseOutputNeuron<double> *) outputNeurons[j])->calculateLoss(validLabels[i]);
                validLoss += fabs( ((BaseOutputNeuron<double> *) outputNeurons[j])->getError(validLabels[i]) );
            }
        }
        if( epoch % 50 == 0 ) {
            evaluateTest( epoch, testData );

            printf( "\n\n---------- evaluate on TRAINING dataset ---------\n" );
            for( int i=0; i < 15; ++i ) {
                int id = (int) (rand() % trainInputs.size());
                printf( "Label: %lf\t Predictions: %lf\n", trainLabels[id], evaluateOne( trainInputs[id] ) );
            }
        }


        if( epoch % 25 == 0 )
            learningRate *= learningRateDecay;
        if( learningRate < minLearningRate ) {
            learningRate = restoreLearningRate;
            evaluateTest( epoch, testData );
        }

        if( bestValidationLoss > validLoss ) {
            bestValidationLoss = validLoss;
            evaluateTest( 19971997, testData );
        }

        printf( "\tValidation loss: %lf\tTrain loss: %lf", validLoss / validInputs.size(), epochTrainLoss / trainInputs.size() );
    }

    evaluateTest( 100000, testData );
}

#endif //NEURALNETWORK_FULLYCONNECTEDTESTDATAMININGPROJECT_H
