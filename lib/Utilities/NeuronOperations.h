
#ifndef NEURALNETWORK_NEURONOPERATIONS_H
#define NEURALNETWORK_NEURONOPERATIONS_H


#include "../Neurons/BaseNeurons/BaseNeuron.h"
#include "../Edges/SimpleEdges/SharedEdge.h"

namespace NeuronOperations {

    /**
     * connect From neuron to To with a weight and deltaWeight
     */
    template<class Type>
    void connectNeurons(BaseNeuron<Type> *from,
                        BaseNeuron<Type> *to,
                        Type *weight = nullptr,
                        Type *deltaWeight = nullptr) {

        if( weight == nullptr )         weight = new Type(rand() / Type(RAND_MAX) - 0.5);
        if( deltaWeight == nullptr )    deltaWeight = new Type( 0 );

        BaseEdge<Type> *edge = new BaseEdge<Type>(from, to, weight, deltaWeight );

        from->addNextLayerConnection(edge);
        to->addPreviousLayerConnection(edge);
    }

    template<class Type>
    void connectConvolutionalNeurons(BaseNeuron<Type> *from,
                                     BaseNeuron<Type> *to,
                                     int *numberOfUsages,
                                     Type *weight,
                                     Type *deltaWeight) {

        SharedEdge <Type>* edge = new SharedEdge <Type> ( from, to, numberOfUsages, weight, deltaWeight );
        from -> addNextLayerConnection( edge );
        to -> addPreviousLayerConnection( edge );
    }
}

#endif //NEURALNETWORK_NEURONOPERATIONS_H
