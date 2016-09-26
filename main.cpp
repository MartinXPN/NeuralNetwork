#include <iostream>
#include <vector>
using namespace std;


#include "lib/Neurons/SigmoidNeuron.h"
#include "lib/BaseClasses/Neurons/BaseInputNeuron.h"
#include "lib/BaseClasses/Neurons/BaseOutputNeuron.h"


int main() {

    const int numberOfNeurons = 7;
    const int numberOfEdges = 9;


    vector< BaseNeuron <double>* > neurons;
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{1} first input neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 0 ) );  // layer{1} second input neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{1} bias neuron

    neurons.push_back( (BaseNeuron <double>*) new SigmoidNeuron <double>() );       // layer{2} first neuron
    neurons.push_back( (BaseNeuron <double>*) new SigmoidNeuron <double>() );       // layer{2} second neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{2} bias neuron

    neurons.push_back( (BaseNeuron <double>*) new BaseOutputNeuron <double>() );    // layer{3} output neuron


/*
0 3 -30
0 4 10
1 3 20
1 4 -20
2 3 20
2 4 -20
3 6 20
4 6 20
5 6 -10
*/
    /// get values of all edges
    for( int i=0; i < numberOfEdges; ++i ) {
        int from, to;
        double *weight = new double;
        cin >> from >> to >> (*weight);

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
        cout << "input: (" << one << "," << two << ") -> " << out << "\t"
             << "loss in #" << iteration << " iteration: " << ((BaseOutputNeuron<double> *) neurons[6]) -> calculateLoss( out )
             << "\tout: " << ((BaseOutputNeuron<double> *) neurons[6]) -> getValue() << endl;
        for (int i = 5; i >= 0; --i) {
            neurons[i] -> calculateLoss();
        }
        /// backpropagate neurons
        for (int i = numberOfNeurons - 1; i >= 0; --i) {
            neurons[i] -> backpropagateNeuron(0.001, 1);
        }
    }

    cout << "Everything is done" << endl;
    cout << "Time for garbage collection" << endl;
    return 0;
}