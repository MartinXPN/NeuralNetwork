
#ifndef NEURALNETWORK_BASEINPUTNEURON_H
#define NEURALNETWORK_BASEINPUTNEURON_H


#include "BaseNeuron.h"

/**
 * Base class for input neurons
 * privately inherited from BaseNeuron <>
 * Contains:
 *      1. next (connections to the next layer)
 *      2. activatedValue
 */
template <class NeuronType>
class BaseInputNeuron : private BaseNeuron <NeuronType> {


protected:
    using BaseNeuron <NeuronType> :: activatedValue;
    using BaseNeuron <NeuronType> :: next;

public:

    BaseInputNeuron();
    BaseInputNeuron( NeuronType inputValue, std::vector < BaseEdge <NeuronType>* > next = {} );
    virtual ~BaseInputNeuron() {}

    using BaseNeuron <NeuronType> :: backpropagateNeuron;
    using BaseNeuron <NeuronType> :: addNextLayerConnection;

    inline const NeuronType& getValue() const   { return activatedValue; }
    inline void setValue( NeuronType value )    { activatedValue = value; }
};


#include "BaseInputNeuron.tpp"


#endif //NEURALNETWORK_BASEINPUTNEURON_H
