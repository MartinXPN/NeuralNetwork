
#ifndef NEURALNETWORK_NEURONTEST_H
#define NEURALNETWORK_NEURONTEST_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include "../../library/neurons/base/BaseInputNeuron.h"
#include "../../library/neurons/base/BaseOutputNeuron.h"
#include "../../library/neurons/Bias.h"
#include "../../library/activations/ReLU.h"
#include "../../library/activations/Sigmoid.h"
#include "../../library/lossfunctions/CrossEntropyCost.h"
#include "../../library/util/NeuronOperations.h"

using namespace std;


void testNeuronsXOR() {

    const int numberOfEdges = 13;

    vector< BaseNeuron <double>* > neurons;
    neurons.push_back( new BaseInputNeuron <double>() );                    // layer{1}     [0]
    neurons.push_back( new BaseInputNeuron <double>() );                    // layer{1}     [1]
    neurons.push_back( new Bias <double>() );                               // [2] bias

    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );    // layer{2}     [3]
    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );    // layer{2}     [4]
    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );    // layer{2}     [5]

    neurons.push_back( new BaseOutputNeuron <double>( new CrossEntropyCost <double>(), //   [6]
                                                      new Sigmoid <double> () ) );

/*
/// network with weights
8 10
0 3 0.5
0 4 0.1
1 3 1.0
1 4 2.2
2 3 0.2
2 4 0.3
3 5 0.4
3 7 0.8
4 6 1.1
4 7 1.2


/// XOR network without weights
0 3
0 4
0 5
1 3
1 4
1 5
2 3
2 4
2 5

2 6
3 6
4 6
5 6
*/
    srand( 1 );
    cout << "Get the edges..." << endl;
    /// get values of all edges
    for( int i=0; i < numberOfEdges; ++i ) {
        int from, to;
        cin >> from >> to;
        NeuronOperations::connectNeurons( neurons[from], neurons[to] );
    }

    cout << "Finished initialization..." << endl;


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

            ((BaseInputNeuron<double> *) neurons[0]) -> setValue( one );
            ((BaseInputNeuron<double> *) neurons[1]) -> setValue( two );

            /// activate neurons
            for (int i = 3; i < neurons.size(); ++i) {
                neurons[i] -> activateNeuron();
            }


            /// calculate losses
            ((BaseOutputNeuron<double> *) neurons.back())->calculateLoss(out);
            double currentLoss = ((BaseOutputNeuron<double> *) neurons.back())->getError(out);
            loss += currentLoss;
            if( iteration == maxIterations - 1 ) {
                cout << "(" << one << "," << two << ") -> " << out << "\tout: "
                     << ((BaseOutputNeuron<double> *) neurons.back())->getValue() << "   \t"
                     << "loss #" << iteration << ": " << currentLoss
                     << endl;
            }

            for( size_t i = neurons.size() - 2; i >= 3; --i ) {
                neurons[i] -> calculateLoss();
            }

            /// backpropagate neurons
            for( size_t i = neurons.size() - 1; i >= 3; --i ) {
                neurons[i] -> backpropagateNeuron();
            }
        }

        if( iteration % 100 == 0 )
            cout << "Loss #" << iteration << ": " << loss / batchSize << endl;

        /// update weights
        for (size_t i = neurons.size() - 1; i >= 3; --i) {
            neurons[i] -> updateWeights( learningRate, batchSize );
        }
    }




    for( int batch = 0; batch < 15; ++batch ) {
        /// set values of input neurons
        double one = rand() % 2;
        double two = rand() % 2;
        double out = (int) one ^ (int) two;

        ((BaseInputNeuron<double> *) neurons[0])->setValue(one);
        ((BaseInputNeuron<double> *) neurons[1])->setValue(two);

        /// activate neurons
        for (int i = 3; i < neurons.size(); ++i) {
            neurons[i]->activateNeuron();
        }

        cout << "(" << one << "," << two << ") -> " << out << "\tout: "
             << ((BaseOutputNeuron<double> *) neurons.back())->getValue() << endl;
    }

    cout << "Everything is done" << endl;
}

#endif //NEURALNETWORK_NEURONTEST_H
