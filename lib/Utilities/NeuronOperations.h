
#ifndef NEURALNETWORK_NEURONOPERATIONS_H
#define NEURALNETWORK_NEURONOPERATIONS_H


#include "../Neurons/BaseNeurons/BaseNeuron.h"

namespace NeuronOperations {

    template<class Type>
    void connectNeurons(BaseNeuron<Type> *from, BaseNeuron<Type> *to, Type *weight) {

        BaseEdge<Type> *edge = new BaseEdge<Type>(from, to, weight);

        from->addNextLayerConnection(edge);
        to->addPreviousLayerConnection(edge);
    }
}

#endif //NEURALNETWORK_NEURONOPERATIONS_H