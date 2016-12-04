
#ifndef NEURALNETWORK_CONVOLUTIONTESTMNIST_H
#define NEURALNETWORK_CONVOLUTIONTESTMNIST_H

#include <iostream>
#include <vector>
#include <fstream>
#include <zconf.h>
#include "../../lib/Layers/BaseLayers/BaseInputLayer.h"
#include "../../lib/Layers/SimpleLayers/FullyConnected.h"
#include "../../lib/Activations/SimpleActivations/ReLU.h"
#include "../../lib/Layers/BaseLayers/BaseOutputLayer.h"
#include "../../lib/LossFunctions/SimpleLossFunctions/CrossEntropyCost.h"
#include "../../lib/Activations/SimpleActivations/Sigmoid.h"
#include "../../lib/Layers/SimpleLayers/Convolution.h"


Bias <double>* bias = new Bias <double>();
BaseInputLayer <double> inputLayer( {1, 28, 28} );
Convolution <double> conv1( { 10, 13, 13 }, { 1, 4, 4 }, new ReLU <double>(), {&inputLayer}, {0, 2, 2}, bias );
Convolution <double> conv2( { 5, 10, 10 }, { 10, 4, 4 }, new ReLU <double>(), {&conv1}, {0, 1, 1}, bias );
BaseOutputLayer <double> outputLayer( {10}, {&conv2}, new CrossEntropyCost <double>(), new Sigmoid <double>(), bias );

using namespace std;
int reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1= (unsigned char) (i & 255);
    ch2= (unsigned char) ((i >> 8) & 255);
    ch3= (unsigned char) ((i >> 16) & 255);
    ch4= (unsigned char) ((i >> 24) & 255);
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
vector< vector <double> > readImages(string directory, size_t numberOfImages, size_t imageSize) {

    vector< vector <double> > images(numberOfImages, vector<double>(imageSize));
    ifstream file (directory,ios::binary);
    if (file.is_open()) {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read( (char*)&magic_number, sizeof(magic_number) );            magic_number= reverseInt( magic_number );
        file.read( (char*)&number_of_images, sizeof(number_of_images) );    number_of_images= reverseInt( number_of_images );
        file.read( (char*)&n_rows, sizeof(n_rows) );                        n_rows= reverseInt( n_rows );
        file.read( (char*)&n_cols, sizeof(n_cols) );                        n_cols= reverseInt( n_cols );

        images.resize( (size_t) number_of_images );
        for(int i=0;i < min(number_of_images, (const int &) numberOfImages); ++i) {
            for (int r = 0; r < n_rows; ++r)
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char *) &temp, sizeof(temp));
                    images[i][(n_rows * r) + c] = (double) temp;
                }
        }
        cout << "Finished reading images" << endl;
    }
    else {
        cout << "Couldn't find the directory: " << directory << endl;
    }
    file.close();
    return images;
}
vector <int> readLabels(string directory, int numberOfLabels) {

    vector <int> labels( (size_t)numberOfLabels );
    ifstream file(directory, ios::binary);
    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if(magic_number != 2049)
            throw runtime_error("Invalid MNIST label file!");
        file.read((char *)&numberOfLabels, sizeof(numberOfLabels)), numberOfLabels = reverseInt(numberOfLabels);
        for(int i = 0; i < numberOfLabels; i++)
            file.read((char*)&labels[i], 1);
    }
    else {
        cout << "Couldn't find the directory: " << directory << endl;
    }

    file.close();
    return labels;
}

void evaluateOne(vector<double> image) {

    /// print iamge
    for( int i=0; i < 28; ++i, printf( "\n" ) )
        for( int j=0; j < 28; ++j ) {
            double current_number = image[ i * 28 + j ];
            if( current_number != 0. )  printf("%.1lf  ", current_number);
            else                        printf( "    " );
        }

    /// set values of input neurons
    for( int j=0; j < image.size(); ++j )
        ((BaseInputNeuron <double>*)inputLayer.getNeurons()[j]) -> setValue( image[j] );

    /// activate neurons
    for (auto neuron : conv1.getNeurons())            neuron->activateNeuron();
    for (auto neuron : conv2.getNeurons())            neuron->activateNeuron();
    for( auto neuron : outputLayer.getNeurons() )   neuron->activateNeuron();

    int maxId = 0;
    for( int i=1; i < outputLayer.getNeurons().size(); ++i )
        if( ((BaseOutputNeuron <double>*)outputLayer.getNeurons()[i]) -> getValue() >
            ((BaseOutputNeuron <double>*)outputLayer.getNeurons()[maxId]) -> getValue() )
            maxId = i;

    printf( "\nNetwork prediction: %d\n", maxId );
}

void testConvolutionMNIST() {


    vector<vector<double>> trainImages = readImages("/home/martin/Desktop/MNIST_train_images.idx3-ubyte", 100000, 28 * 28);
    // vector<vector<double>> testImages = readImages("/home/ubuntu/Desktop/MNIST_test_images.idx3-ubyte", 100000, 28*28);
    vector <int> labels = readLabels( "/home/martin/Desktop/MNIST_train_labels.idx1-ubyte", 100000 );

    for( int i=0; i < trainImages.size(); ++i )
        for( int j=0; j < trainImages[i].size(); ++j )
            trainImages[i][j] /= 255.;

//    cout << "Image: \n";
//    for( int i=0; i < 28; ++i, printf( "\n" ) )
//        for( int j=0; j < 28; ++j ) {
//            double current_number = trainImages[7][ i * 28 + j ];
//            if( current_number != 0. )  printf("%.1lf  ", current_number);
//            else                        printf( "    " );
//        }

    /// construct the network

    inputLayer.createNeurons();
    conv1.createNeurons();
    conv2.createNeurons();
    outputLayer.createNeurons();

    conv1.createWeights();
    conv2.createWeights();

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



    auto inputNeurons = inputLayer.getNeurons();
    auto outputNeurons = outputLayer.getNeurons();


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
                    ((BaseInputNeuron <double>*)inputNeurons[j]) -> setValue( trainImages[i][j] );

                /// activate neurons
                for (auto neuron : conv1.getNeurons())    neuron->activateNeuron();
                for (auto neuron : conv2.getNeurons())    neuron->activateNeuron();
                for( auto neuron : outputNeurons )      neuron->activateNeuron();

                /// calculate losses
                for( int j=0; j < outputNeurons.size(); ++j ) {
                    ((BaseOutputNeuron <double>*)outputNeurons[j]) -> calculateLoss( j == labels[i] );
                    batchLoss += ((BaseOutputNeuron <double>*)outputNeurons[j]) -> getError( j == labels[i] );
                }
                for (auto neuron : conv2.getNeurons())    neuron->calculateLoss();
                for (auto neuron : conv1.getNeurons())    neuron->calculateLoss();

                /// backpropagate neurons
                for (auto neuron : outputNeurons)       neuron->backpropagateNeuron();
                for (auto neuron : conv2.getNeurons())    neuron->backpropagateNeuron();
                for (auto neuron : conv1.getNeurons())    neuron->backpropagateNeuron();
            }

            cout << "Loss #" << batch << ": " << batchLoss / batchSize << endl;

            /// update weights
            for (auto neuron : outputNeurons)       neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : conv2.getNeurons())    neuron->updateWeights(learningRate, batchSize);
            for (auto neuron : conv1.getNeurons())    neuron->updateWeights(learningRate, batchSize);
        }
    }

    cout << "Training is done" << endl;
    for( int i=0; i < 10; ++i ) {
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
