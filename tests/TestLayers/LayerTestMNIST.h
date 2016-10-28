
#ifndef NEURALNETWORK_LAYERTESTMNIST_H
#define NEURALNETWORK_LAYERTESTMNIST_H

// /home/ubuntu/Desktop/MNIST_train_images.idx3-ubyte
#include <iostream>
#include <vector>
#include <fstream>
#include "../../lib/Layers/BaseLayers/BaseInputLayer.h"
#include "../../lib/Layers/SimpleLayers/FullyConnected.h"
#include "../../lib/Activations/SimpleActivations/ReLU.h"
#include "../../lib/Layers/BaseLayers/BaseOutputLayer.h"
#include "../../lib/LossFunctions/SimpleLossFunctions/CrossEntropyCost.h"
#include "../../lib/Activations/SimpleActivations/Sigmoid.h"


BaseBias <double>* bias = new BaseBias <double>();
BaseInputLayer <double> inputLayer( 28*28 );
FullyConnected <double> fc1( 100, {&inputLayer}, new ReLU <double>(), bias );
FullyConnected <double> fc2( 100, {&fc1}, new ReLU <double>(), bias );
BaseOutputLayer <double> outputLayer( 10, {&fc2}, new CrossEntropyCost <double>(), new Sigmoid <double>(), bias );

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
        file.read((char*)&magic_number,sizeof(magic_number));           printf( "magic number: %d\n", magic_number );           magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));   printf( "number of images: %d\n", number_of_images );   number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));                       printf( "number of rows: %d\n", n_rows );               n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));                       printf( "number of columns: %d\n", n_cols );            n_cols= reverseInt(n_cols);

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

vector <int> readLabels(string directory, int number_of_labels) {

    vector <int> labels( (size_t)number_of_labels );
    ifstream file(directory, ios::binary);
    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if(magic_number != 2049)
            throw runtime_error("Invalid MNIST label file!");
        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);
        for(int i = 0; i < number_of_labels; i++)
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

    vector<vector<double>> trainImages = readImages("/home/ubuntu/Desktop/MNIST_train_images.idx3-ubyte", 100000, 28 * 28);
    // vector<vector<double>> testImages = readImages("/home/ubuntu/Desktop/MNIST_test_images.idx3-ubyte", 100000, 28*28);
    vector <int> labels = readLabels( "/home/ubuntu/Desktop/MNIST_train_labels.idx1-ubyte", 100000 );

    for( int i=0; i < trainImages.size(); ++i )
        for( int j=0; j < trainImages[i].size(); ++j )
            trainImages[i][j] /= 255.;

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

    inputLayer.createNeurons( 28*28 );
    fc1.createNeurons( 100, new ReLU <double>() );
    fc2.createNeurons( 100, new ReLU <double>() );
    outputLayer.createNeurons( 10 );

    fc1.connectNeurons( inputLayer );
    fc2.connectNeurons( fc1 );
    outputLayer.connectNeurons( fc2 );



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

    cout << "Everything is done" << endl;

    for( int i=0; i < 5; ++i ) {
        int id = (int) (rand() % trainImages.size());
        evaluateOne(trainImages[id]);
    }
}

#endif //NEURALNETWORK_LAYERTESTMNIST_H
