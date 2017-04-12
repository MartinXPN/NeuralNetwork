
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
    FullyConnected <double> out( {1}, new Sigmoid <double>(), {&hidden1}, bias );
    LossLayer <double> lossLayer( {1}, {&out}, new CrossEntropyCost <double>());


    hidden1.createWeights();
    hidden1.connectNeurons();

    out.createWeights();
    out.connectNeurons();

    lossLayer.createWeights();
    lossLayer.connectNeurons();

    /// get neurons
    vector <BaseNeuron <double>* > inputNeurons = inputLayer.getNeurons();
    vector <BaseNeuron <double>* > hiddenNeurons1 = hidden1.getNeurons();
    vector <BaseNeuron <double>* > hiddenNeurons2 = out.getNeurons();
    vector <BaseNeuron <double>* > outputNeurons = lossLayer.getNeurons();
    printf( "input: %u\nhidden: %u\noutput: %u\n", inputNeurons.size(), hiddenNeurons1.size(), outputNeurons.size() );


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
            for( auto neuron : hiddenNeurons1 )  neuron -> activateNeuron();
            for( auto neuron : hiddenNeurons2 )  neuron -> activateNeuron();
            for( auto neuron : outputNeurons )  neuron -> activateNeuron();

            /// calculate losses
            (( BaseOutputNeuron <double>* ) ( outputNeurons[0] ))-> calculateLoss( out );
            loss += (( BaseOutputNeuron <double>* ) ( outputNeurons[0] ))-> getError( out );
            for( auto neuron : hiddenNeurons2 )  neuron -> calculateLoss();
            for( auto neuron : hiddenNeurons1 )  neuron -> calculateLoss();


            /// backpropagate neurons
            outputNeurons[0] -> backpropagateNeuron();
            for( auto neuron : hiddenNeurons2 )  neuron -> backpropagateNeuron();
            for( auto neuron : hiddenNeurons1 )  neuron -> backpropagateNeuron();
        }

        if( iteration % 100 == 0 )
            cout << "Loss #" << iteration << ": " << loss / batchSize << endl;

        /// update weights
        for( auto neuron : outputNeurons )  neuron -> updateWeights( learningRate, batchSize );
        for( auto neuron : hiddenNeurons2 )  neuron -> updateWeights( learningRate, batchSize );
        for( auto neuron : hiddenNeurons1 )  neuron -> updateWeights( learningRate, batchSize );
    }




    for( int batch = 0; batch < 15; ++batch ) {
        /// set values of input neurons
        double one = rand() % 2;
        double two = rand() % 2;
        double out = (int) one ^ (int) two;

        ((BaseInputNeuron<double> *) inputNeurons[0])->setValue(one);
        ((BaseInputNeuron<double> *) inputNeurons[1])->setValue(two);

        /// activate neurons
        for( auto neuron : hiddenNeurons1 )  neuron -> activateNeuron();
        for( auto neuron : hiddenNeurons2 )  neuron -> activateNeuron();
        for( auto neuron : outputNeurons )  neuron -> activateNeuron();


        cout << "(" << one << "," << two << ") -> " << out << "\tout: "
             << ((BaseOutputNeuron<double> *) outputNeurons[0])->getValue() << endl;
    }

    cout << "Everything is done" << endl;
}

#endif //NEURALNETWORK_LAYERTEST_H
