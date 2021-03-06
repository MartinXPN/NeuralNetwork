
#ifndef NEURALNETWORK_BASEBIAS_H
#define NEURALNETWORK_BASEBIAS_H


#include "base/BaseInputNeuron.h"
#include <vector>

/**
 * Bias class is extended from BaseInputNeuron and has a constant value 1
 * The whole network can be supported with only one bias connected with different edges to the neurons
 * So, it's more efficient to keep only one instance of Bias for the whole network and connect this one bias
 *      to every neuron that needs a bias
 */
template <class NeuronType>
class Bias : public BaseInputNeuron <NeuronType> {

public:
    Bias( std::vector < BaseEdge <NeuronType>* > next = {} );
    using BaseInputNeuron <NeuronType> :: addNextLayerConnection;
    virtual ~Bias();
};

#include "Bias.tpp"

#endif //NEURALNETWORK_BASEBIAS_H
