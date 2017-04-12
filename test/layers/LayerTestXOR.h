
#ifndef NEURALNETWORK_LAYERTEST_H
#define NEURALNETWORK_LAYERTEST_H

#include <cstdio>
#include <vector>
#include <iostream>
#include "../../library/neurons/Bias.h"
#include "../../library/layers/InputLayer.h"
#include "../../library/layers/FullyConnected.h"
#include "../../library/activations/ReLU.h"
#include "../../library/layers/LossLayer.h"
#include "../../library/lossfunctions/CrossEntropyCost.h"
#include "../../library/activations/Sigmoid.h"

/**
 * Define a simple network with input, hidden, and output layers
 */
void layerTestXOR() {
    using namespace std;

    cout << "Starting layer test..." << endl;
    srand( 1 );

    /// construct the Network
    Bias <double>* bias = new Bias <double>();
    InputLayer <double> inputLayer( {2} );
    FullyConnected <double> hidden1( {4}, new ReLU <double>(), {&inputLayer}, bias );
    LossLayer <double> outputLayer( {1}, {&hidden1}, new CrossEntropyCost <double>(), new Sigmoid <double>(), bias );


    inputLayer.createNeurons();
    hidden1.createNeurons();
    hidden1.createWeights();
    hidden1.connectNeurons();

    outputLayer.createNeurons();
    outputLayer.connectNeurons();

    /// get neurons
    vector <BaseNeuron <double>* > inputNeurons = inputLayer.getNeurons();
    vector <BaseNeuron <double>* > hiddenNeurons = hidden1.getNeurons();
    vector <BaseNeuron <double>* > outputNeurons = outputLayer.getNeurons();
    printf( "input: %u\nhidden: %u\noutput: %u\n", inputNeurons.size(), hiddenNeurons.size(), outputNeurons.size() );


    /// learn XOR
    const int maxIterations = 10000;
    const int batchSize = 10;
    double learningRate = 0.1;
    for( int iteration = 0; iteration < maxIterations; ++iteration ) {

        double loss = 0;
        for( int batch = 0; batch < batchSize; ++batch ) {
            /// set values of input neurons
            double one = rand() % 2;
            double two = rand() % 2;
            double out = (int) one ^ (int) two;

            ((BaseInputNeuron<double> *) inputNeurons[0]) -> setValue( one );
            ((BaseInputNeuron<double> *) inputNeurons[1]) -> setValue( two );

            /// activate neurons
            for( auto neuron : hiddenNeurons )  neuron -> activateNeuron();
            for( auto neuron : outputNeurons )  neuron -> activateNeuron();

            /// calculate losses
            (( BaseOutputNeuron <double>* ) ( outputNeurons[0] ))-> calculateLoss( out );
            loss += (( BaseOutputNeuron <double>* ) ( outputNeurons[0] ))-> getError( out );
            for( auto neuron : hiddenNeurons )  neuron -> calculateLoss();


            /// backpropagate neurons
            outputNeurons[0] -> backpropagateNeuron();
            for( auto neuron : hiddenNeurons )  neuron -> backpropagateNeuron();
        }

        if( iteration % 100 == 0 )
            cout << "Loss #" << iteration << ": " << loss / batchSize << endl;

        /// update weights
        for( auto neuron : outputNeurons )  neuron -> updateWeights( learningRate, batchSize );
        for( auto neuron : hiddenNeurons )  neuron -> updateWeights( learningRate, batchSize );
    }




    for( int batch = 0; batch < 15; ++batch ) {
        /// set values of input neurons
        double one = rand() % 2;
        double two = rand() % 2;
        double out = (int) one ^ (int) two;

        ((BaseInputNeuron<double> *) inputNeurons[0])->setValue(one);
        ((BaseInputNeuron<double> *) inputNeurons[1])->setValue(two);

        /// activate neurons
        for( auto neuron : hiddenNeurons )  neuron -> activateNeuron();
        for( auto neuron : outputNeurons )  neuron -> activateNeuron();


        cout << "(" << one << "," << two << ") -> " << out << "\tout: "
             << ((BaseOutputNeuron<double> *) outputNeurons[0])->getValue() << endl;
    }

    cout << "Everything is done" << endl;
}

#endif //NEURALNETWORK_LAYERTEST_H
