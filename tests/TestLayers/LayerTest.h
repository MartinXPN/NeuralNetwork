
#ifndef NEURALNETWORK_LAYERTEST_H
#define NEURALNETWORK_LAYERTEST_H

#include "../../lib/Layers/BaseLayers/BaseHiddenLayer.h"
#include "../../lib/Layers/BaseLayers/BaseOutputLayer.h"
#include "../../lib/Layers/BaseLayers/BaseInputLayer.h"
#include "../../lib/Layers/SimpleLayers/FullyConnected.h"

/**
 * Define a simple network with input, hidden, and output layers
 */
void test() {

    BaseInputLayer <double> inputLayer( 2 );                                    inputLayer.createNeurons( 2 );
    FullyConnected <double> hidden1( 5, {&inputLayer}, new ReLU <double>() );   hidden1.createNeurons( 5, new ReLU <double>() );    hidden1.connectNeurons( inputLayer );
    FullyConnected <double> hidden2( 5, {&hidden1}, new ReLU <double>() );      hidden2.createNeurons( 5, new ReLU <double>() );    hidden2.connectNeurons( hidden1 );
    BaseOutputLayer <double> outputLayer( 10, {&hidden2}, new CrossEntropyCost <double>(), new Sigmoid <double>() );
    outputLayer.createNeurons( 10 );
    outputLayer.connectNeurons( hidden2 );
}

#endif //NEURALNETWORK_LAYERTEST_H
