#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;


#include "lib/BaseClasses/Neurons/BaseInputNeuron.h"
#include "lib/BaseClasses/Neurons/BaseOutputNeuron.h"
#include "lib/Activations/SimpleActivations/ReLU.h"
#include "lib/Activations/SimpleActivations/Sigmoid.h"


int main() {

    const int numberOfNeurons = 10;
    const int numberOfEdges = 13;

    vector< BaseNeuron <double>* > neurons;
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>() );     // layer{1} first input neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>() );     // layer{1} second input neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{1} bias neuron

    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );            // layer{2} first neuron
    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );            // layer{2} second neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{2} bias neuron


    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );            // layer{3} first neuron
    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );            // layer{3} second neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{3} bias neuron

    neurons.push_back( (BaseNeuron <double>*) new BaseOutputNeuron <double>( new Sigmoid <double> () ) );    // output layer

/*
0 3
0 4
1 3
1 4
2 3
2 4
3 6
3 7
4 6
4 7
5 6
5 7
6 9
7 9
8 9
*/
    srand( 1 );
    cout << "Get the edges..." << endl;
    /// get values of all edges
    for( int i=0; i < numberOfEdges; ++i ) {
        int from, to;
        double *weight = new double( ( rand()%10 - 5 ) / 5. );
        cin >> from >> to;

        BaseEdge <double>* edge = new BaseEdge <double>( neurons[from], neurons[to], weight );
        neurons[from] -> addNextLayerConnection( edge );
        neurons[to] -> addPreviousLayerConnection( edge );
    }


    const int maxIterations = 50000;
    const int batchSize = 5;
    double learningRate = 0.01;
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
            for (int i = 3; i < numberOfNeurons; ++i) {
                neurons[i] -> activateNeuron();
            }


            /// calculate losses
            double currentLoss = ((BaseOutputNeuron<double> *) neurons.back())->calculateLoss(out);
            loss += currentLoss;
            if( iteration == maxIterations - 1 ) {
                cout << "(" << one << "," << two << ") -> " << out << "\tout: "
                     << ((BaseOutputNeuron<double> *) neurons.back())->getValue() << "   \t"
                     << "loss #" << iteration << ": " << currentLoss
                     << endl;
            }

            for( int i = numberOfNeurons - 2; i >= 3; --i ) {
                neurons[i] -> calculateLoss();
            }

            /// backpropagate neurons
            for( int i = numberOfNeurons - 2; i >= 0; --i ) {
                neurons[i] -> backpropagateNeuron( learningRate, batchSize );
            }
        }

//        cout << "Loss #" << iteration << ": " << loss / batchSize << endl;

        /// update weights
        for (int i = numberOfNeurons - 2; i >= 0; --i) {
            neurons[i] -> updateWeights();
        }
    }




    for( int batch = 0; batch < 10; ++batch ) {
        /// set values of input neurons
        double one = rand() % 2;
        double two = rand() % 2;
        double out = (int) one ^(int) two;

        ((BaseInputNeuron<double> *) neurons[0])->setValue(one);
        ((BaseInputNeuron<double> *) neurons[1])->setValue(two);

        /// activate neurons
        for (int i = 3; i < numberOfNeurons; ++i) {
            neurons[i]->activateNeuron();
        }

        cout << "(" << one << "," << two << ") -> " << out << "\tout: "
             << ((BaseOutputNeuron<double> *) neurons.back())->getValue() << endl;
    }

    cout << "Everything is done" << endl;
    return 0;
}