
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
    /// hide all these functions as we don't need them
    using BaseNeuron <NeuronType> :: loss;
    using BaseNeuron <NeuronType> :: preActivatedValue;
    using BaseNeuron <NeuronType> :: activationFunction;
    using BaseNeuron <NeuronType> :: activateNeuron;
    using BaseNeuron <NeuronType> :: getPreActivatedValue;
    using BaseNeuron <NeuronType> :: getLoss;
    using BaseNeuron <NeuronType> :: calculateLoss;
    using BaseNeuron <NeuronType> :: addPreviousLayerConnection;
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
