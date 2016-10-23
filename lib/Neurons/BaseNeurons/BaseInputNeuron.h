
#ifndef NEURALNETWORK_BASEINPUTNEURON_H
#define NEURALNETWORK_BASEINPUTNEURON_H


#include "BaseNeuron.h"

/**
 * Base class for input neurons
 * Inherited from BaseNeuron
 * Contains:
 *      1. next (connections to the next layer)
 *      2. activatedValue
 */
template <class NeuronType>
class BaseInputNeuron : public BaseNeuron <NeuronType> {

private:
    using BaseNeuron <NeuronType> :: loss;                          /// loss is not needed for input neurons
    using BaseNeuron <NeuronType> :: preActivatedValue;             /// input neuron only one value -> activatedValue
    using BaseNeuron <NeuronType> :: activationFunction;            /// there is no need in having an activation function
    using BaseNeuron <NeuronType> :: activateNeuron;                /// we don't activate the input neuron
    using BaseNeuron <NeuronType> :: getPreActivatedValue;          /// we don't use preactivatedValue
    using BaseNeuron <NeuronType> :: getLoss;                       /// we don't use loss
    using BaseNeuron <NeuronType> :: calculateLoss;                 /// we don't use loss
    using BaseNeuron <NeuronType> :: addPreviousLayerConnection;    /// hide this function
    using BaseNeuron <NeuronType> :: backpropagateNeuron;
    using BaseNeuron <NeuronType> :: updateWeights;


protected:
    using BaseNeuron <NeuronType> :: activatedValue;
    using BaseNeuron <NeuronType> :: next;

public:

    BaseInputNeuron();
    BaseInputNeuron( NeuronType inputValue, std::vector < BaseEdge <NeuronType>* > next = {} );
    virtual ~BaseInputNeuron() {}

    using BaseNeuron <NeuronType> :: addNextLayerConnection;
    inline const NeuronType& getValue() const   { return activatedValue; }
    inline void setValue( NeuronType value )    { activatedValue = value; }
};


#include "BaseInputNeuron.tpp"


#endif //NEURALNETWORK_BASEINPUTNEURON_H
