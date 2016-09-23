#include <iostream>
#include <cstdio>
#include <vector>
using namespace std;

#include "BaseNeuron.h"

int main() {
    freopen( "simple_network.txt", "r", stdin );


    int numberOfNeurons;
    int numberOfEdges;
    scanf("%d %d", &numberOfNeurons, &numberOfEdges );

    //vector <BaseNeuron <double> > neurons( static_cast <unsigned int> (numberOfNeurons) );
    // compile error as Base Neuron is an abstract class


    for( int i=0; i < numberOfEdges; ++i ) {
        int from;
        int to;
        double weight;
        scanf( "%d %d %lf", &from, &to, &weight );
    }
    return 0;
}