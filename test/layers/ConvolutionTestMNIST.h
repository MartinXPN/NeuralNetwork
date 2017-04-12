
#ifndef NEURALNETWORK_CONVOLUTIONTESTMNIST_H
#define NEURALNETWORK_CONVOLUTIONTESTMNIST_H

#include <iostream>
#include <vector>
#include <fstream>
#include <zconf.h>
#include "../../library/neurons/Bias.h"
#include "../../library/layers/InputLayer.h"
#include "../../library/layers/LossLayer.h"
#include "../../library/layers/Convolution.h"
#include "../../library/activations/ReLU.h"
#include "../../library/activations/Sigmoid.h"
#include "../../library/lossfunctions/CrossEntropyCost.h"
#include "../../library/util/MNIST.h"
#include "../../library/layers/FullyConnected.h"
#include "../../library/activations/ELU.h"
#include "../../library/lossfunctions/MeanSquaredError.h"
#include "../../library/initializers/neuron/SimpleNeuronInitializer.h"


Bias <double>* bias = new Bias <double>();
InputLayer <double> inputLayer( {1, 28, 28} );
Convolution <double> conv1( { 10, 13, 13 }, { 1, 4, 4 }, {&inputLayer}, {0, 2, 2}, new SimpleNeuronInitializer <double>( new ReLU<double>() ), bias );
Convolution <double> conv2( { 5, 10, 10 }, { 10, 4, 4 }, {&conv1}, {0, 1, 1}, new SimpleNeuronInitializer <double>( new ReLU<double>() ), bias );
FullyConnected <double> outputLayer( {10}, {&conv2}, new SimpleNeuronInitializer <double> ( new Sigmoid <double> ), bias );
LossLayer <double> loss( {10}, {&outputLayer}, new OutputNeuronInitializer<double>(new CrossEntropyCost<double>()) );

using namespace std;

void evaluateOne(vector<double> image) {

    /// print iamge
    MNIST::printImage( image );

    /// set values of input neurons
    for( int j=0; j < image.size(); ++j )
        ((BaseInputNeuron <double>*)inputLayer.getNeurons()[j]) -> setValue( image[j] );

    /// activate neurons
    for (auto neuron : conv1.getNeurons())          neuron->activateNeuron();
    for (auto neuron : conv2.getNeurons())          neuron->activateNeuron();
    for( auto neuron : outputLayer.getNeurons() )   neuron->activateNeuron();
    for( auto neuron : loss.getNeurons() )          neuron->activateNeuron();

    int maxId = 0;
    for( int i=1; i < loss.getNeurons().size(); ++i )
        if( ((BaseOutputNeuron <double>*)loss.getNeurons()[i]) -> getValue() >
            ((BaseOutputNeuron <double>*)loss.getNeurons()[maxId]) -> getValue() )
            maxId = i;

    printf( "\nNetwork prediction: %d\n", maxId );
}

void testConvolutionMNIST() {

    vector<vector<double>> trainImages = MNIST::readImages("/home/ubuntu/Desktop/MNIST_train_images.idx3-ubyte", 100000, 28 * 28);
    // vector<vector<double>> testImages = readImages("/home/ubuntu/Desktop/MNIST_test_images.idx3-ubyte", 100000, 28*28);
    vector <int> labels = MNIST::readLabels( "/home/ubuntu/Desktop/MNIST_train_labels.idx1-ubyte" );

//    cout << "Image: \n";
//    for( int i=0; i < 28; ++i, printf( "\n" ) )
//        for( int j=0; j < 28; ++j ) {
//            double current_number = trainImages[7][ i * 28 + j ];
//            if( current_number != 0. )  printf("%.1lf  ", current_number);
//            else                        printf( "    " );
//        }

    /// construct the network
    conv1.createWeights();
    conv2.createWeights();
    outputLayer.createWeights();
    loss.createWeights();

    conv1.connectNeurons();
    printf( "Sample connection (%d)(%d)(%d)(%d):\n", inputLayer.size(), conv1.size(), conv2.size(), outputLayer.size() );
    for( auto item : conv1.getNeurons()[1689] -> getPreviousConnections() ) {
        printf( "%lf\t", item -> getWeight() );
        fflush( stdout );
    }
    fflush( stdout );

//    sleep(10000);
    conv2.connectNeurons();
    outputLayer.connectNeurons();
    loss.connectNeurons();


    printf( "Sample connection:\n" );
    for( auto item : conv1.getNeurons()[0] -> getPreviousConnections() ) {
        printf( "%lf\t", item -> getWeight() );
    }
    printf( "\n" );
    for( auto item : conv1.getNeurons()[1] -> getPreviousConnections() ) {
        printf( "%lf\t", item -> getWeight() );
    }
    printf( "\n" );
    for( auto item : conv2.getNeurons()[2] -> getPreviousConnections() ) {
        printf( "%lf\t", item -> getWeight() );
    }
    printf( "\n" );




    /// learn to classify digits
    const int maxEpochs = 1;
    const int batchSize = 50;
    double learningRate = 0.01;
    for( int epoch = 0; epoch < maxEpochs; ++epoch ) {

        for (int batch = 0; batch < trainImages.size(); batch += batchSize) {
            double batchLoss = 0;
            for (int i = batch; i < batch + batchSize && i < trainImages.size(); ++i) {
                /// set values of input neurons
                for( int j=0; j < trainImages[i].size(); ++j )
                    ((BaseInputNeuron <double>*)inputLayer.getNeurons()[j]) -> setValue( trainImages[i][j] );

                /// activate neurons
                for (auto neuron : conv1.getNeurons())          neuron->activateNeuron();
                for (auto neuron : conv2.getNeurons())          neuron->activateNeuron();
                for( auto neuron : outputLayer.getNeurons() )   neuron->activateNeuron();
                for( auto neuron : loss.getNeurons() )          neuron->activateNeuron();

                /// calculate losses
                for( int j=0; j < loss.getNeurons().size(); ++j ) {
                    BaseOutputNeuron <double>* outputNeuron = (BaseOutputNeuron<double> *) loss.getNeurons()[j];
                    outputNeuron -> calculateLoss( j == labels[i] );
//                    cout << "LOSS = " << outputNeuron -> getError( j == labels[i] ) << endl;
                    batchLoss += fabs( outputNeuron -> getError( j == labels[i] ) );
                }
                for (auto neuron : outputLayer.getNeurons())    neuron->calculateLoss();
                for (auto neuron : conv2.getNeurons())          neuron->calculateLoss();
                for (auto neuron : conv1.getNeurons())          neuron->calculateLoss();

                /// backpropagate neurons
                for (auto neuron : loss.getNeurons())           neuron->backpropagateNeuron();
                for (auto neuron : outputLayer.getNeurons())    neuron->backpropagateNeuron();
                for (auto neuron : conv2.getNeurons())          neuron->backpropagateNeuron();
                for (auto neuron : conv1.getNeurons())          neuron->backpropagateNeuron();
            }

            cout << "Loss #" << batch << ": " << batchLoss / batchSize << endl;

            /// update weights
            for (auto neuron : loss.getNeurons())               neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : outputLayer.getNeurons())        neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : conv2.getNeurons())              neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : conv1.getNeurons())              neuron->updateWeights(learningRate, batchSize);
        }
    }

    cout << "Training is done" << endl;
    for( int i=0; i < 15; ++i ) {
        int id = (int) (rand() % trainImages.size());
        evaluateOne(trainImages[id]);
    }


//    /// calculate number of very small weights
//    int smallWeights = 0;
//    for( auto neuron : conv1.getNeurons() )
//        for( auto edge : neuron -> getPreviousConnections() )
//            if( fabs( edge -> getWeight() ) < 0.001 )
//                ++smallWeights;
//
//    for( auto neuron : conv2.getNeurons() )
//        for( auto edge : neuron -> getPreviousConnections() )
//            if( fabs( edge -> getWeight() ) < 0.001 )
//                ++smallWeights;
//
//    for( auto neuron : outputLayer.getNeurons() )
//        for( auto edge : neuron -> getPreviousConnections() )
//            if( fabs( edge -> getWeight() ) < 0.001 )
//                ++smallWeights;
//
//    cout << "Number of edges smaller than 0.001: " << smallWeights;
}

#endif //NEURALNETWORK_CONVOLUTIONTESTMNIST_H
