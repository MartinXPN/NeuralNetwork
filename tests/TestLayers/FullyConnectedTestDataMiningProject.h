
#ifndef NEURALNETWORK_FULLYCONNECTEDTESTDATAMININGPROJECT_H
#define NEURALNETWORK_FULLYCONNECTEDTESTDATAMININGPROJECT_H

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <fstream>
#include "../../lib/Layers/BaseLayers/BaseInputLayer.h"
#include "../../lib/Neurons/SimpleNeurons/Bias.h"
#include "../../lib/Layers/SimpleLayers/FullyConnected.h"
#include "../../lib/Activations/SimpleActivations/ReLU.h"
#include "../../lib/Layers/BaseLayers/BaseOutputLayer.h"
#include "../../lib/LossFunctions/SimpleLossFunctions/CrossEntropyCost.h"
#include "../../lib/Activations/SimpleActivations/Sigmoid.h"
#include "../../lib/LossFunctions/SimpleLossFunctions/MeanSquaredError.h"


using namespace std;

//////////////////////// NETWORK CONSTRUCTION//////////////////////////////////
Bias <double> *bias = new Bias <double>();
BaseInputLayer <double> in( {143} );
FullyConnected <double> fc1( {100}, new ReLU <double>(), {&in}, bias );
FullyConnected <double> fc2( {200}, new ReLU <double>(), {&fc1}, bias );
FullyConnected <double> fc3( {100}, new ReLU <double>(), {&fc2}, bias );
FullyConnected <double> fc4( {32}, new ReLU <double>(), {&fc3}, bias );
BaseOutputLayer <double> out( {1}, {&fc4}, new CrossEntropyCost <double>(), new Sigmoid <double>(), bias );



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
    for (auto neuron : fc3.getNeurons())    neuron->activateNeuron();
    for (auto neuron : fc4.getNeurons())    neuron->activateNeuron();
    for (auto neuron : out.getNeurons())    neuron->activateNeuron();
    return ((BaseOutputNeuron <double> *)out.getNeurons()[0])->getValue();
}

void testDataMiningProject() {

    //////////////READ THE DATA////////////////////////////
    auto testData = readTest( "/home/ubuntu/Desktop/Projects/DataMining/filtered_scaled_test.csv" );
//    printf( "TEST:\n" );
//    for( int i=0; i < testData.size(); ++i, cout << endl )
//        for( int j=0; j < testData[i].size(); ++j ) {
//            cout << testData[i][j] << " ";
//        }

    printf( "\n\nLast row of TEST dataset-->\t" );
    for( int j=0; j < testData.back().size(); ++j ) {
        cout << testData.back()[j] << " ";
    }

    auto train = readTrain( "/home/ubuntu/Desktop/Projects/DataMining/filtered_scaled_train.csv" );
    auto trainData = train.first;
    auto trainLabels = train.second;

//    printf( "\n\nTRAIN:\n" );
//    for( int i=0; i < trainData.size(); ++i, cout << endl ) {
//        cout << trainLabels[i] << "\t";
//        for( int j=0; j < trainData[i].size(); ++j ) {
//            cout << trainData[i][j] << " ";
//        }
//    }

    printf( "\nLast row of TRAIN dataset-->\t" );
    for( int j=0; j < trainData.back().size(); ++j ) {
        cout << trainData.back()[j] << " ";
    }
    printf( "\n---> label: %lf\n", trainLabels.back() );

    //////////////////////// INITIALIZATION //////////////////////////////
    printf( "\nNetwork: (%d) -> (%d) -> (%d) -> (%d) -> (%d) -> (%d)\n\n", in.size(), fc1.size(), fc2.size(), fc3.size(), fc4.size(), out.size() );
    in.createNeurons();
    fc1.createNeurons();
    fc2.createNeurons();
    fc3.createNeurons();
    fc4.createNeurons();
    out.createNeurons();

    fc1.connectNeurons();
    fc2.connectNeurons();
    fc3.connectNeurons();
    fc4.connectNeurons();
    out.connectNeurons();

    auto inputNeurons = in.getNeurons();
    auto outputNeurons = out.getNeurons();

    ////////////////////// TRAINING //////////////////////////////////////
    const int maxEpochs = 100;
    const int batchSize = 200;
    double learningRate = 0.05;
    for( int epoch = 0; epoch < maxEpochs; ++epoch ) {

        printf( "\n\nEpoch: %d --Learning rate (%lf)--\n", epoch, learningRate );

        for (int batch = 0; batch < trainData.size(); batch += batchSize) {
            double batchLoss = 0;
            for (int i = batch; i < batch + batchSize && i < trainData.size(); ++i) {
                /// set values of input neurons
                for( int j=0; j < trainData[i].size(); ++j )
                    ((BaseInputNeuron <double>*)inputNeurons[j]) -> setValue( trainData[i][j] );

                /// activate neurons
                for (auto neuron : fc1.getNeurons())    neuron->activateNeuron();
                for (auto neuron : fc2.getNeurons())    neuron->activateNeuron();
                for (auto neuron : fc3.getNeurons())    neuron->activateNeuron();
                for (auto neuron : fc4.getNeurons())    neuron->activateNeuron();
                for( auto neuron : outputNeurons )      neuron->activateNeuron();

                /// calculate losses
                for( int j=0; j < outputNeurons.size(); ++j ) {
                    ((BaseOutputNeuron <double>*)outputNeurons[j]) -> calculateLoss( trainLabels[i] );
                    batchLoss += ((BaseOutputNeuron <double>*)outputNeurons[j]) -> getError( trainLabels[i] );
                }
                for (auto neuron : fc4.getNeurons())    neuron->calculateLoss();
                for (auto neuron : fc3.getNeurons())    neuron->calculateLoss();
                for (auto neuron : fc2.getNeurons())    neuron->calculateLoss();
                for (auto neuron : fc1.getNeurons())    neuron->calculateLoss();

                /// backpropagate neurons
                for (auto neuron : outputNeurons)       neuron->backpropagateNeuron();
                for (auto neuron : fc4.getNeurons())    neuron->backpropagateNeuron();
                for (auto neuron : fc3.getNeurons())    neuron->backpropagateNeuron();
                for (auto neuron : fc2.getNeurons())    neuron->backpropagateNeuron();
                for (auto neuron : fc1.getNeurons())    neuron->backpropagateNeuron();
            }

            cout << "Loss #" << batch << ": " << batchLoss / batchSize << endl;

            /// update weights
            for (auto neuron : outputNeurons)       neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : fc4.getNeurons())    neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : fc3.getNeurons())    neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : fc2.getNeurons())    neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : fc1.getNeurons())    neuron->updateWeights(learningRate, batchSize);
        }

        learningRate *= 0.95;
    }


    printf( "\n\n\n---------- evaluate on TRAINING dataset ---------\n" );
    for( int i=0; i < 15; ++i ) {
        int id = (int) (rand() % trainData.size());
        printf( "\nLabel: %lf\t Predictions: %lf ----> ", trainLabels[id], evaluateOne( trainData[id] ) );
        for( auto item : trainData[id] )
            printf( "%lf\t", item );
    }
    printf( "\n" );



    printf( "\nEvaluating on TESTING dataset" );
    vector <double> testLabels;
    for( int i=0; i < testData.size(); ++i ) {
        testLabels.push_back( evaluateOne( testData[i] ) );
    }


    printf( "\nSaving results to file..." );
    //////////////////// SAVE RESULTS TO FILE //////////////////
    ofstream fout;
    fout.open ("/home/ubuntu/Desktop/Projects/DataMining/neural_network.csv");
    fout << "ID,Class\n";
    for( int i=0; i < testLabels.size(); ++i ) {
        fout << i+1 << ',' << testLabels[i] << '\n';
    }
    fout.close();
}

#endif //NEURALNETWORK_FULLYCONNECTEDTESTDATAMININGPROJECT_H
