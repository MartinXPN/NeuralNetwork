
#ifndef NEURALNETWORK_BASEOUTPUTNEURON_H
#define NEURALNETWORK_BASEOUTPUTNEURON_H


#include "BaseNeuron.h"
#include "../../Activations/SimpleActivations/Identitiy.h"
#include "../../LossFunctions/BaseLoss/BaseLossFunction.h"


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
class BaseOutputNeuron : public BaseNeuron <NeuronType> {

private:
    using BaseNeuron <NeuronType> :: addNextLayerConnection;    /// there is no next layer
    using BaseNeuron <NeuronType> :: updateWeights;             /// last layer is not responsible for updating the weights
    using BaseNeuron <NeuronType> :: calculateLoss;             /// the loss function for outputNeuron is different

protected:
    using BaseNeuron <NeuronType> :: activatedValue;
    using BaseNeuron <NeuronType> :: preActivatedValue;
    using BaseNeuron <NeuronType> :: loss;
    using BaseNeuron <NeuronType> :: activationFunction;
    BaseLossFunction <NeuronType>* lossFunction;


public:
    BaseOutputNeuron( BaseLossFunction <NeuronType>* lossFunction );
    BaseOutputNeuron( BaseLossFunction <NeuronType>* lossFunction,
                      BaseActivationFunction <NeuronType> * activationFunction,
                      std::vector < BaseEdge <NeuronType>* > previous = {} );
    virtual ~BaseOutputNeuron() {}


    virtual void calculateLoss( NeuronType targetValue );
    virtual NeuronType getError( NeuronType targetValue );
    using BaseNeuron <NeuronType> :: activateNeuron;
    using BaseNeuron <NeuronType> :: addPreviousLayerConnection;

    inline const NeuronType& getValue() const { return activatedValue; }
};


#include "BaseOutputNeuron.tpp"

#endif //NEURALNETWORK_BASEOUTPUTNEURON_H
