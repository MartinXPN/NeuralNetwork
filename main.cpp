#include <iostream>
#include <vector>
using namespace std;


#include "lib/BaseClasses/Neurons/BaseInputNeuron.h"
#include "lib/BaseClasses/Neurons/BaseOutputNeuron.h"
#include "lib/Activations/SimpleActivations/Sigmoid.h"


int main() {

    const int numberOfNeurons = 7;
    const int numberOfEdges = 9;

    cout << "start";

    vector< BaseNeuron <double>* > neurons;
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{1} first input neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 0 ) );  // layer{1} second input neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{1} bias neuron

    neurons.push_back( new BaseNeuron <double>( new Sigmoid <double>() ) );         // layer{2} first neuron
    neurons.push_back( new BaseNeuron <double>( new Sigmoid <double>() ) );         // layer{2} second neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{2} bias neuron

    neurons.push_back( (BaseNeuron <double>*) new BaseOutputNeuron <double>( new Sigmoid <double> () ) );    // layer{3} output neuron


/*
0 3
0 4
1 3
1 4
2 3
2 4
3 6
4 6
5 6
*/
    cout << "Get the edges..." << endl;
    /// get values of all edges
    for( int i=0; i < numberOfEdges; ++i ) {
        int from, to;
        double *weight = new double( 0.5 );
        cin >> from >> to;

        BaseEdge <double>* edge = new BaseEdge <double>( neurons[from], neurons[to], weight );
        neurons[from] -> addNextLayerConnection( edge );
        neurons[to] -> addPreviousLayerConnection( edge );
    }


    srand((unsigned int) time(0));
    for( int iteration = 0; iteration < 10000; ++iteration ) {
        /// set values of input neurons
        double one = rand() % 2;
        double two = rand() % 2;
        double out = (int)one ^ (int)two;

        ((BaseInputNeuron<double> *) neurons[0]) -> setValue( one );
        ((BaseInputNeuron<double> *) neurons[1]) -> setValue( two );

        /// activate neurons
        for (int i = 3; i < numberOfNeurons; ++i) {
            neurons[i] -> activateNeuron();
        }


        /// calculate losses
        cout << "(" << one << "," << two << ") -> " << out << "\tout: " << ((BaseOutputNeuron<double> *) neurons[6]) -> getValue() << "\t"
             << "loss #" << iteration << ": " << ((BaseOutputNeuron<double> *) neurons[6]) -> calculateLoss( out ) << endl;

        for (int i = 5; i >= 3; --i) {
            neurons[i] -> calculateLoss();
        }

        /// backpropagate neurons
        for (int i = numberOfNeurons - 1; i >= 0; --i) {
            neurons[i] -> backpropagateNeuron(0.01, 1);
        }

        /// update weights
        for (int i = numberOfNeurons - 1; i >= 0; --i) {
            neurons[i] -> updateWeights();
        }
    }

    cout << "Everything is done" << endl;
    cout << "Time for garbage collection" << endl;
    return 0;
}