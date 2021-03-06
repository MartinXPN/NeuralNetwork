
#ifndef NEURALNETWORK_BASEOUTPUTNEURON_H
#define NEURALNETWORK_BASEOUTPUTNEURON_H


#include "BaseNeuron.h"
#include "../../lossfunctions/base/BaseLossFunction.h"
#include "../../activations/Identitiy.h"


/**
 * Base class for output neurons
 * Inherited from BaseNeuron
 * @Lifecycle:
 *      1. activateNeuron
 *      2. calculateLoss
 *      3. backpropagateNeuron
 *      4. updateWeights (called after the batch is processed)
 * Conteins:
 *      1. activatedValue
 *      2. preActivatedValue
 *      3. loss
 *      4. activationFunction
 *      5. lossFunction
 */
template <class NeuronType>
class BaseOutputNeuron : public BaseNeuron <NeuronType> {

private:
    using BaseNeuron <NeuronType> :: addNextLayerConnection;    /// there is no next layer
    using BaseNeuron <NeuronType> :: calculateLoss;             /// the loss function for outputNeuron is different

protected:
    using BaseNeuron <NeuronType> :: activatedValue;
    using BaseNeuron <NeuronType> :: preActivatedValue;
    using BaseNeuron <NeuronType> :: loss;
    using BaseNeuron <NeuronType> :: activationFunction;
    BaseLossFunction <NeuronType>* lossFunction;


public:
    BaseOutputNeuron( BaseLossFunction <NeuronType>* lossFunction,
                      BaseActivationFunction <NeuronType> * activationFunction = new Identity <NeuronType>(),
                      std::vector < BaseEdge <NeuronType>* > previous = {} );
    virtual ~BaseOutputNeuron() {}


    virtual void calculateLoss( NeuronType targetValue );
    virtual NeuronType getError( NeuronType targetValue );
    using BaseNeuron <NeuronType> :: activateNeuron;
    using BaseNeuron <NeuronType> :: addPreviousLayerConnection;
    using BaseNeuron <NeuronType> :: backpropagateNeuron;
    using BaseNeuron <NeuronType> :: updateWeights;


    inline const NeuronType& getValue() const { return activatedValue; }
};


#include "BaseOutputNeuron.tpp"

#endif //NEURALNETWORK_BASEOUTPUTNEURON_H
