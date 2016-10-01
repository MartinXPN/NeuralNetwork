
#ifndef NEURALNETWORK_BASEOUTPUTNEURON_H
#define NEURALNETWORK_BASEOUTPUTNEURON_H


#include "BaseNeuron.h"
#include "../../Activations/SimpleActivations/Identitiy.h"


/**
 * Base class for output neurons
 * privately inherited from BaseNeuron <>
 * Conteins:
 *      1. activatedValue
 *      2. preActivatedValue
 *      3. loss
 *      4. activationFunction
 */
template <class NeuronType>
class BaseOutputNeuron : private BaseNeuron <NeuronType> {


protected:
    using BaseNeuron <NeuronType> :: activatedValue;
    using BaseNeuron <NeuronType> :: preActivatedValue;
    using BaseNeuron <NeuronType> :: loss;
    using BaseNeuron <NeuronType> :: activationFunction;


public:
    BaseOutputNeuron();
    BaseOutputNeuron( BaseActivationFunction <NeuronType> * activationFunction );
    BaseOutputNeuron( BaseActivationFunction <NeuronType> * activationFunction,
                      std::vector < BaseEdge <NeuronType>* > previous );
    virtual ~BaseOutputNeuron() {}


    virtual NeuronType calculateLoss( NeuronType targetValue );
    using BaseNeuron <NeuronType> :: activateNeuron;
    using BaseNeuron <NeuronType> :: addPreviousLayerConnection;

    inline NeuronType getValue() const { return activatedValue; }
};


#include "BaseOutputNeuron.tpp"

#endif //NEURALNETWORK_BASEOUTPUTNEURON_H
