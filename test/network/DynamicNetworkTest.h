
#ifndef NEURALNETWORK_DYNAMICNETWORKTEST_H
#define NEURALNETWORK_DYNAMICNETWORKTEST_H

#include <algorithm>
#include "../../library/util/MNIST.h"
#include "../../library/neurons/Bias.h"
#include "../../library/layers/InputLayer.h"
#include "../../library/layers/Convolution.h"
#include "../../library/activations/ReLU.h"
#include "../../library/layers/FullyConnected.h"
#include "../../library/layers/LossLayer.h"
#include "../../library/lossfunctions/CrossEntropyCost.h"
#include "../../library/activations/Sigmoid.h"
#include "../../library/network/DynamicNetwork.h"

using namespace std;

vector< vector <double> > trainImages, testImages;
vector< vector <double> > trainLabels, testLabels;

void testDynamicNetworkMNIST() {

    /// load the data
    trainImages = MNIST::readImages("/home/martin/Desktop/MNIST_train_images.idx3-ubyte", 100000, 28 * 28);
    testImages = MNIST::readImages("/home/martin/Desktop/MNIST_test_images.idx3-ubyte", 100000, 28 * 28);
    trainLabels = MNIST::toLabelMatrix( MNIST::readLabels( "/home/martin/Desktop/MNIST_train_labels.idx1-ubyte" ) );
    testLabels = MNIST::toLabelMatrix( MNIST::readLabels( "/home/martin/Desktop/MNIST_test_labels.idx1-ubyte" ) );

    printf( "Train -> Images: %d\tLabels: %d\n", trainImages.size(), trainLabels.size() );
    printf( "Test ->  Images: %d\tLabels: %d\n", testImages.size(), testLabels.size() );


    /// construct the network
    Bias <double>* bias = new Bias <double>();
    InputLayer <double> inputLayer( {1, 28, 28} );
    Convolution <double> conv1( { 2, 13, 13 }, { 1, 4, 4 }, new ReLU <double>(), {&inputLayer}, {0, 2, 2}, bias );
    Convolution <double> conv2( { 1, 10, 10 }, { 2, 4, 4 }, new ReLU <double>(), {&conv1}, {0, 1, 1}, bias );
    FullyConnected <double> fc1( {10}, new Sigmoid <double>(), {&conv2}, bias );
    LossLayer <double> outputLayer( {10}, {&fc1}, new CrossEntropyCost <double>() );


    /// initialise the network
    DynamicNetwork <double> net( {&inputLayer}, {&conv1, &conv2, &fc1}, {&outputLayer} );
    net.initializeNetwork();
    printf( "Before starting number of weights: %d\n", (int) net.getSmallWeightsNumber( 1111 ) );

    /// start training
    for( int epoch=0; epoch < 2; ++epoch )
        net.trainEpoch( trainImages.size(),
                        100,
                        0.01,
                        [&trainImages] (size_t item) -> vector <double>& { return trainImages[item]; },
                        [&trainLabels] (size_t item) -> vector <double>& { return trainLabels[item]; },
                        [&]() {
                            /// get accuracy on test data-set
                            int correct = 0;
                            for( size_t i=0; i < testImages.size(); ++i ) {
                                net.evaluateOne( testImages[i], [&](const vector<double> &res) {
                                    correct += int( std::max_element( res.begin(), res.end()) - res.begin() ) ==
                                               int( std::max_element( testLabels[i].begin(), testLabels[i].end()) - testLabels[i].begin() );
                                } );
                            }

                            printf("\n\nEpoch Trained, ACC: %lf\n", double(correct) / testImages.size() );
                            if( epoch % 2 == 0 ) {
                                printf("Current small weights -> %d\n", (int) net.getSmallWeightsNumber(0.1));
                                net.pruneNetwork(0.1);
                            }
                        },
                        [&]() {
                            printf( "Current Batch Loss: %lf\n", net.getBatchLoss() );
                        } );


    /// evaluate the result
    for( int i=0; i < 20; ++i ) {
        int id = (int) (rand() % testImages.size());
        net.evaluateOne( testImages[id], [&](const vector<double> &res) {
            MNIST::printImage( testImages[id] );
            cout << "Prediction -> " << std::max_element(res.begin(), res.end()) - res.begin() << endl;
        } );
    }
}

#endif //NEURALNETWORK_DYNAMICNETWORKTEST_H
