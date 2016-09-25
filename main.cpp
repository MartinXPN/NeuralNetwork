#include <iostream>
#include <vector>
using namespace std;


#include "lib/Neurons/SigmoidNeuron.h"


int main() {

    int numberOfNeurons;
    int numberOfEdges;
    cin >> numberOfNeurons >> numberOfEdges;


    vector< SigmoidNeuron <double> > neurons((size_t) numberOfNeurons);

    /// get values of all edges
    for( int i=0; i < numberOfEdges; ++i ) {
        int from, to;
        double *weight = new double;
        cin >> from >> to >> (*weight);

        BaseEdge <double>* edge = new BaseEdge <double>( &neurons[from], &neurons[to], weight );
        neurons[from].addNextLayerConnection( edge );
        neurons[to].addPreviousLayerConnection( edge );
    }


    /// set values of input neurons
    neurons[0].setValue( 3 );
    neurons[1].setValue( 4 );
    neurons[2].setValue( 1 );

    /// activate neurons
    for( int i=3; i < numberOfNeurons; ++i ) {
        neurons[i].onActivation();
    }

    for( int i=0; i < numberOfNeurons; ++i ) cout << "preactivation[" << i << "] = " << neurons[i].getPreActivatedValue() << endl;
    for( int i=0; i < numberOfNeurons; ++i ) cout << "activated[" << i << "] = " << neurons[i].getActivatedValue() << endl;


    cout << "Everything is done" << endl;
    cout << "Time for garbage collection" << endl;
    return 0;
}