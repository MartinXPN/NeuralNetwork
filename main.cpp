#include <iostream>
#include <cstdio>
#include <vector>
using namespace std;

#include "SigmoidNeuron.h"


int main() {

    int numberOfNeurons;
    int numberOfEdges;
    scanf("%d %d", &numberOfNeurons, &numberOfEdges );


    vector< SigmoidNeuron <double> > neurons;
    for( int i=0; i < numberOfNeurons; ++ i ) {
        SigmoidNeuron <double> neuron;
        neurons.push_back( neuron );
        cout << "# " << i << " Neuron created" << endl;
    }

    for( int i=0; i < numberOfEdges; ++i ) {
        int from;
        int to;
        double *weight = new double;
        scanf( "%d %d %lf", &from, &to, weight );

        --from;
        --to;
        BaseEdge <double> ( &neurons[from], &neurons[to], weight );
        BaseEdge <double>* edge = new BaseEdge <double>( &neurons[from], &neurons[to], weight );
        neurons[from].addNextLayerConnection( edge );
        neurons[to].addPreviousLayerConnection( edge );

        cout << from << "->" << to << " (" << *weight << ") " << endl;
    }


    cout << "Done";

    return 0;
}