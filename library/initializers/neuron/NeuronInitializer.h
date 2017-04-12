
#ifndef NEURALNETWORK_BASENEURONINITIALIZER_H
#define NEURALNETWORK_BASENEURONINITIALIZER_H


#include <cstddef>
#include "../../neurons/base/BaseNeuron.h"


/**
 * Interface for creating neurons
 * Layers get NeuronInitializer as a constructor parameter or just a collection of neurons
 */
template <class NeuronType>
class NeuronInitializer {

public:
    virtual std::vector <BaseNeuron<NeuronType>* > createNeurons(size_t numberOfNeurons) = 0;
};


#endif //NEURALNETWORK_BASENEURONINITIALIZER_H
