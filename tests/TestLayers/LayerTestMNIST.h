
#ifndef NEURALNETWORK_LAYERTESTMNIST_H
#define NEURALNETWORK_LAYERTESTMNIST_H

#include <iostream>
#include <vector>
#include <fstream>
#include "../../lib/Layers/BaseLayers/BaseInputLayer.h"
#include "../../lib/Layers/SimpleLayers/FullyConnected.h"
#include "../../lib/Activations/SimpleActivations/ReLU.h"
#include "../../lib/Layers/BaseLayers/BaseOutputLayer.h"
#include "../../lib/LossFunctions/SimpleLossFunctions/CrossEntropyCost.h"
#include "../../lib/Activations/SimpleActivations/Sigmoid.h"
#include "../../lib/Utilities/MNIST.h"


Bias <double>* bias = new Bias <double>();
BaseInputLayer <double> inputLayer( {28*28} );
FullyConnected <double> fc1( {100}, new ReLU <double>(), {&inputLayer}, bias );
FullyConnected <double> fc2( {100}, new ReLU <double>(), {&fc1}, bias );
BaseOutputLayer <double> outputLayer( {10}, {&fc2}, new CrossEntropyCost <double>(), new Sigmoid <double>(), bias );

using namespace std;

void evaluateOne(vector<double> image) {

    /// print iamge
    MNIST::printImage( image );

    /// set values of input neurons
    for( int j=0; j < image.size(); ++j )
        ((BaseInputNeuron <double>*)inputLayer.getNeurons()[j]) -> setValue( image[j] );

    /// activate neurons
    for (auto neuron : fc1.getNeurons())            neuron->activateNeuron();
    for (auto neuron : fc2.getNeurons())            neuron->activateNeuron();
    for( auto neuron : outputLayer.getNeurons() )   neuron->activateNeuron();

    int maxId = 0;
    for( int i=1; i < outputLayer.getNeurons().size(); ++i )
        if( ((BaseOutputNeuron <double>*)outputLayer.getNeurons()[i]) -> getValue() >
            ((BaseOutputNeuron <double>*)outputLayer.getNeurons()[maxId]) -> getValue() )
            maxId = i;

    printf( "\nNetwork prediction: %d\n", maxId );
}

void testMNIST() {

    vector<vector<double>> trainImages = MNIST::readImages("/home/ubuntu/Desktop/MNIST_train_images.idx3-ubyte", 100000, 28 * 28);
    // vector<vector<double>> testImages = readImages("/home/ubuntu/Desktop/MNIST_test_images.idx3-ubyte", 100000, 28*28);
    vector <int> labels = MNIST::readLabels( "/home/ubuntu/Desktop/MNIST_train_labels.idx1-ubyte", 100000 );

    cout << "Image: \n";
    for( int i=0; i < 28; ++i, printf( "\n" ) )
        for( int j=0; j < 28; ++j ) {
            double current_number = trainImages[7][ i * 28 + j ];
            if( current_number != 0. )  printf("%.1lf  ", current_number);
            else                        printf( "    " );
        }


    cout << endl << endl;
    cout << "Labels: ";
    for( int i=0; i < 10; ++i )
        cout << labels[i] << endl;

    /// construct the network

    inputLayer.createNeurons();
    fc1.createNeurons();
    fc2.createNeurons();
    outputLayer.createNeurons();

    fc1.createWeights();
    fc2.createWeights();

    fc1.connectNeurons();
    fc2.connectNeurons();
    outputLayer.connectNeurons();



    auto inputNeurons = inputLayer.getNeurons();
    auto outputNeurons = outputLayer.getNeurons();


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
                    ((BaseInputNeuron <double>*)inputNeurons[j]) -> setValue( trainImages[i][j] );

                /// activate neurons
                for (auto neuron : fc1.getNeurons())    neuron->activateNeuron();
                for (auto neuron : fc2.getNeurons())    neuron->activateNeuron();
                for( auto neuron : outputNeurons )      neuron->activateNeuron();

                /// calculate losses
                for( int j=0; j < outputNeurons.size(); ++j ) {
                    ((BaseOutputNeuron <double>*)outputNeurons[j]) -> calculateLoss( j == labels[i] );
                    batchLoss += ((BaseOutputNeuron <double>*)outputNeurons[j]) -> getError( j == labels[i] );
                }
                for (auto neuron : fc2.getNeurons())    neuron->calculateLoss();
                for (auto neuron : fc1.getNeurons())    neuron->calculateLoss();

                /// backpropagate neurons
                for (auto neuron : outputNeurons)       neuron->backpropagateNeuron();
                for (auto neuron : fc2.getNeurons())    neuron->backpropagateNeuron();
                for (auto neuron : fc1.getNeurons())    neuron->backpropagateNeuron();
            }

            cout << "Loss #" << batch << ": " << batchLoss / batchSize << endl;

            /// update weights
            for (auto neuron : outputNeurons)       neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : fc2.getNeurons())    neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : fc1.getNeurons())    neuron->updateWeights(learningRate, batchSize);
        }
    }

    cout << "Training is done" << endl;
    for( int i=0; i < 5; ++i ) {
        int id = (int) (rand() % trainImages.size());
        evaluateOne(trainImages[id]);
    }


    /// calculate number of very small weights
    int smallWeights = 0;
    for( auto neuron : fc1.getNeurons() )
        for( auto edge : neuron -> getPreviousConnections() )
            if( fabs( edge -> getWeight() ) < 0.001 )
                ++smallWeights;

    for( auto neuron : fc2.getNeurons() )
        for( auto edge : neuron -> getPreviousConnections() )
            if( fabs( edge -> getWeight() ) < 0.001 )
                ++smallWeights;

    for( auto neuron : outputLayer.getNeurons() )
        for( auto edge : neuron -> getPreviousConnections() )
            if( fabs( edge -> getWeight() ) < 0.001 )
                ++smallWeights;
    cout << "Number of edges smaller than 0.001: " << smallWeights;


    /// prune fc2
    for( auto neuron : fc2.getNeurons() ) {
        cout << "Before pruning the neuron size of previous connections: " << neuron -> getPreviousConnections().size() << endl;
        for (auto edge : neuron->getPreviousConnections())
            if (fabs(edge->getWeight()) < 0.001) {
                double& weight = (double &) edge -> getWeight();
                neuron->removePreviousLayerConnection( &weight );
            }
        cout << "After pruning the neuron size of previous connections: " << neuron -> getPreviousConnections().size() << endl;
    }
}

#endif //NEURALNETWORK_LAYERTESTMNIST_H
