
#ifndef NEURALNETWORK_LAYERTEST_H
#define NEURALNETWORK_LAYERTEST_H

#include "../../lib/Layers/BaseLayers/BaseHiddenLayer.h"
#include "../../lib/Layers/BaseLayers/BaseOutputLayer.h"
#include "../../lib/Layers/BaseLayers/BaseInputLayer.h"
#include "../../lib/Layers/SimpleLayers/FullyConnected.h"

/**
 * Define a simple network with input, hidden, and output layers
 */
void layerTestXOR() {

    srand( 1 );
    /// construct the Network
    BaseBias <double>* bias = new BaseBias <double>();
    BaseInputLayer <double> inputLayer( 2 );
    inputLayer.createNeurons( 2 );

    FullyConnected <double> hidden1( 4, {&inputLayer}, new ReLU <double>(), bias );
    hidden1.createNeurons( 4, new ReLU <double>() );

    BaseOutputLayer <double> outputLayer( 1,
                                          {&hidden1},
                                          new CrossEntropyCost <double>(),
                                          new Sigmoid <double>(), bias );
    outputLayer.createNeurons( 1 );
    outputLayer.connectNeurons( hidden1 );


    /// get neurons
    vector <BaseNeuron <double>* > inputNeurons = inputLayer.getNeurons();
    vector <BaseNeuron <double>* > hiddenNeurons = hidden1.getNeurons();
    vector <BaseNeuron <double>* > outputNeurons = outputLayer.getNeurons();

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
            for( int i=hiddenNeurons.size()-1; i >= 0; --i )
                hiddenNeurons[i] -> calculateLoss();


            /// backpropagate neurons
            for( int i=outputNeurons.size()-1; i >= 0; --i )    hiddenNeurons[i] -> backpropagateNeuron();
            for( int i=hiddenNeurons.size()-1; i >= 0; --i )    hiddenNeurons[i] -> backpropagateNeuron();
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
