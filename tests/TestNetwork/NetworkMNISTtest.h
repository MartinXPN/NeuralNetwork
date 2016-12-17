
#ifndef NEURALNETWORK_NETWORKMNISTTEST_H
#define NEURALNETWORK_NETWORKMNISTTEST_H

#include <string>
#include <bits/ios_base.h>
#include <ios>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "../../lib/Neurons/SimpleNeurons/Bias.h"
#include "../../lib/Layers/BaseLayers/BaseInputLayer.h"
#include "../../lib/Layers/SimpleLayers/Convolution.h"
#include "../../lib/Activations/SimpleActivations/ReLU.h"
#include "../../lib/Layers/BaseLayers/BaseOutputLayer.h"
#include "../../lib/LossFunctions/SimpleLossFunctions/CrossEntropyCost.h"
#include "../../lib/Activations/SimpleActivations/Sigmoid.h"
#include "../../lib/Network/NeuralNetwork.h"
#include "../../lib/Layers/SimpleLayers/FullyConnected.h"
#include "../../lib/Utilities/MNIST.h"
using namespace std;

vector< vector <double> > trainImages;
vector< vector <double> > trainLabels;
//auto inputLoader( size_t item ) { return trainImages[item]; }
auto labelLoader( size_t item ) { return trainLabels[item]; }


void networkMNISTtest() {

    trainImages = MNIST::readImages("/home/martin/Desktop/MNIST_train_images.idx3-ubyte", 100000, 28 * 28);
    // vector<vector<double>> testImages = readImages("/home/ubuntu/Desktop/MNIST_test_images.idx3-ubyte", 100000, 28*28);
    vector <int> labels = MNIST::readLabels( "/home/martin/Desktop/MNIST_train_labels.idx1-ubyte", 100000 );
    trainLabels = MNIST::toLabelMatrix( labels );


    /// construct the network
    Bias <double>* bias = new Bias <double>();
    BaseInputLayer <double> inputLayer( {1, 28, 28} );
    Convolution <double> conv1( { 10, 13, 13 }, { 1, 4, 4 }, new ReLU <double>(), {&inputLayer}, {0, 2, 2}, bias );
    Convolution <double> conv2( { 5, 10, 10 }, { 10, 4, 4 }, new ReLU <double>(), {&conv1}, {0, 1, 1}, bias );
    FullyConnected <double> fc1( {100}, new ReLU <double>(), {&conv2}, bias );
    BaseOutputLayer <double> outputLayer( {10}, {&fc1}, new CrossEntropyCost <double>(), new Sigmoid <double>(), bias );


    /// initialise the network
    NeuralNetwork <double> net( {&inputLayer}, {&conv1, &conv2, &fc1}, {&outputLayer} );
    net.initializeNetwork();


    /// start training
    net.trainEpoch( trainImages.size(),
                    100,
                    0.01,
                    [&trainImages] (size_t item) { return trainImages[item]; },
                    labelLoader,
                    []() { printf( "Epoch Trained!!!" ); } );


    /// evaluate the result
    for( int i=0; i < 15; ++i ) {
        int id = (int) (rand() % trainImages.size());
        net.evaluateOne( trainImages[id], [&](const vector<double> &res) {
            MNIST::printImage( trainImages[id] );
            cout << "Prediction -> " << std::max_element(res.begin(), res.end()) - res.begin() << endl;
        } );
    }
}

#endif //NEURALNETWORK_NETWORKMNISTTEST_H
