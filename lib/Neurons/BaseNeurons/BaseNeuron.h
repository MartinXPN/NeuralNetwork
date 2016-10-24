
#ifndef NEURALNETWORK_BASENEURON_H
#define NEURALNETWORK_BASENEURON_H

#include <vector>
#include "../../Edges/BaseEdge.h"
#include "../../Activations/BaseActivation/BaseActivationFunction.h"


template <class EdgeType> class BaseEdge;   /// say that this class exists but don't declare what's inside
                                            /// this is needed in order to be able to keep a pointer inside BaseNeuron
                                            /// and to keep a pointer of BaseNeuron inside BaseEdge as without this it's a compile error

/**
 * Base class for a neuron
 * @Lifecycle:
 *      1. activateNeuron
 *      2. calculateLoss
 *      3. backpropagateNeuron
 *      4. updateWeights (called after the batch is processed)
 *
 * @Contains:
 *      1. edges to the next layer
 *      2. edges to the previous layer
 *      3. activatedValue
 *      4. preActivatedValue
 *      5. loss
 *      6. activation function (interface) -> inherited from BaseActivationFunction
 */
template <class NeuronType>
class BaseNeuron {

protected:
    std::vector < BaseEdge <NeuronType>* > next;        /// all the connections to the neuron from the next layer
    std::vector < BaseEdge <NeuronType>* > previous;    /// all the connections to the neuron from the previous layer

    BaseActivationFunction <NeuronType>* activationFunction; /// inherface for activation function ( contains activation() and activationDerivative() )
    NeuronType activatedValue;      /// value after activating the neuron
    NeuronType preActivatedValue;   /// value before activating the neuron
    NeuronType loss;                /// loss calculated during backpropagation


public:
    BaseNeuron( BaseActivationFunction <NeuronType>* activationFunction,
                std::vector < BaseEdge <NeuronType>* > next = {},
                std::vector < BaseEdge <NeuronType>* > previous = {} );
    virtual ~BaseNeuron();

    inline const NeuronType& getActivatedValue() const      { return activatedValue; }
    inline const NeuronType& getPreActivatedValue() const   { return preActivatedValue; }   /// get pre activated value (i.e. sum of [values of neurons from previous layer * weights connected to them ] )
    inline const NeuronType& getLoss() const                { return loss; }
    inline const std::vector < BaseEdge <NeuronType>* >& getNextConnections() const         { return next; }
    inline const std::vector < BaseEdge <NeuronType>* >& getPreviousConnections() const     { return previous; }

    virtual void addNextLayerConnection( BaseEdge <NeuronType>* edge )      { next.push_back( edge ); }
    virtual void addPreviousLayerConnection( BaseEdge <NeuronType> * edge ) { previous.push_back( edge ); }



    /**
     * Called to activate the neuron
     * calculates the sum of [values of neurons from previous layer * weights connected to them ]
     * updates the values activatedValue and preActivatedValue
     */
    virtual void activateNeuron();

    /**
     * Called to calculate the loss for the neuron
     */
    virtual void calculateLoss();

    /**
     * Called to beckpropagating the neuron
     */
    virtual void backpropagateNeuron(NeuronType coefficient = 1);

    /**
     * Called to update the weights connecting the neuron to the next layer
     */
    virtual void updateWeights(NeuronType learningRate, int batchSize);
};

#include "BaseNeuron.tpp"

#endif //NEURALNETWORK_BASENEURON_H
