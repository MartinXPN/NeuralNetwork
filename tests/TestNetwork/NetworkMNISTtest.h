
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
void printMNISTImage( const vector <double>& image ) {
    cout << endl;
    for( int i=0; i < 28; ++i, printf( "\n" ) )
        for( int j=0; j < 28; ++j )
            if( image[ i*28 + j ] == 0 )    printf( "   " );
            else                            printf( "%.1lf ", image[ i*28 + j ] );
}


vector< vector <double> > trainImages;
vector< vector <double> > trainLabels;
//auto inputLoader( size_t item ) { return trainImages[item]; }
auto labelLoader( size_t item ) { return trainLabels[item]; }


void networkMNISTtest() {

    trainImages = readImages("/home/ubuntu/Desktop/MNIST_train_images.idx3-ubyte", 100000, 28 * 28);
    // vector<vector<double>> testImages = readImages("/home/ubuntu/Desktop/MNIST_test_images.idx3-ubyte", 100000, 28*28);
    vector <int> labels = readLabels( "/home/ubuntu/Desktop/MNIST_train_labels.idx1-ubyte", 100000 );
    for( int i=0; i < labels.size(); ++i ) {
        trainLabels.push_back( vector <double>( 10, 0 ) );
        trainLabels[i][ labels[i] ] = 1;
    }

    for( int i=0; i < trainImages.size(); ++i )
        for( int j=0; j < trainImages[i].size(); ++j )
            trainImages[i][j] /= 255.;



    /// construct the network
    Bias <double>* bias = new Bias <double>();
    BaseInputLayer <double> inputLayer( {1, 28, 28} );
//    Convolution <double> conv1( {10, 13, 13}, {1, 4, 4}, new ReLU <double>(), {&inputLayer}, {0, 2, 2}, bias );
    FullyConnected <double> fc1( {100}, new ReLU <double>(), {&inputLayer}, bias );
    FullyConnected <double> fc2( {100}, new ReLU <double>(), {&fc1}, bias );
    BaseOutputLayer <double> outputLayer( {10}, {&fc2}, new CrossEntropyCost <double>(), new Sigmoid <double>(), bias );


    /// initialise the network
    NeuralNetwork <double> net( {&inputLayer}, {&fc1, &fc2}, {&outputLayer} );
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
            printMNISTImage( trainImages[id] );
            cout << "Prediction -> " << std::max_element(res.begin(), res.end()) - res.begin() << "\tLabel -> " << labels[id] << endl;
        } );
    }
}

#endif //NEURALNETWORK_NETWORKMNISTTEST_H
