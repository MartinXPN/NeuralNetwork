
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

vector< vector <double> > trainImages, testImages;
vector< vector <double> > trainLabels, testLabels;
//vector <double> inputLoader( size_t item ) { return trainImages[item]; }
vector <double> labelLoader( size_t item ) { return trainLabels[item]; }


void testNeuralNetworkMNIST() {

    /// load the data
    trainImages = MNIST::readImages("/home/martin/Desktop/MNIST_train_images.idx3-ubyte", 100000, 28 * 28);
    testImages = MNIST::readImages("/home/martin/Desktop/MNIST_test_images.idx3-ubyte", 100000, 28 * 28);
    trainLabels = MNIST::toLabelMatrix( MNIST::readLabels( "/home/martin/Desktop/MNIST_train_labels.idx1-ubyte" ) );
    testLabels = MNIST::toLabelMatrix( MNIST::readLabels( "/home/martin/Desktop/MNIST_test_labels.idx1-ubyte" ) );

    printf( "Train -> Images: %d\tLabels: %d\n", trainImages.size(), trainLabels.size() );
    printf( "Test ->  Images: %d\tLabels: %d\n", testImages.size(), testLabels.size() );


    /// construct the network
    Bias <double>* bias = new Bias <double>();
    BaseInputLayer <double> inputLayer( {1, 28, 28} );
    Convolution <double> conv1( { 10, 13, 13 }, { 1, 4, 4 }, new ReLU <double>(), {&inputLayer}, {0, 2, 2}, bias );
    Convolution <double> conv2( { 5, 10, 10 }, { 10, 4, 4 }, new ReLU <double>(), {&conv1}, {0, 1, 1}, bias );
    FullyConnected <double> fc1( {50}, new ReLU <double>(), {&conv2}, bias );
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
    for( int i=0; i < 20; ++i ) {
        int id = (int) (rand() % testImages.size());
        net.evaluateOne( testImages[id], [&](const vector<double> &res) {
            MNIST::printImage( testImages[id] );
            cout << "Prediction -> " << std::max_element(res.begin(), res.end()) - res.begin() << endl;
        } );
    }
}

#endif //NEURALNETWORK_NETWORKMNISTTEST_H
