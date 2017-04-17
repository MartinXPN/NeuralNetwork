
#ifndef NEURALNETWORK_NEURONOPERATIONS_H
#define NEURALNETWORK_NEURONOPERATIONS_H


#include <cstdlib>
#include "../neurons/base/BaseNeuron.h"
#include "../edges/SharedEdge.h"

namespace NeuronOperations {

    template<class Type>
    void connectConvolutionalNeurons(BaseNeuron<Type> *from,
                                     BaseNeuron<Type> *to,
                                     int *numberOfUsages,
                                     Type *weight,
                                     Type *deltaWeight) {

//        printf( "connect -> %d %d %d(%d) %d(%lf) %d(%lf)\n", from, to, numberOfUsages, *numberOfUsages, weight, *weight, deltaWeight, *deltaWeight );
        SharedEdge <Type>* edge = new SharedEdge <Type> ( from, to, numberOfUsages, weight, deltaWeight );
        from -> addNextLayerConnection( edge );
        to -> addPreviousLayerConnection( edge );
    }
}

#endif //NEURALNETWORK_NEURONOPERATIONS_H
