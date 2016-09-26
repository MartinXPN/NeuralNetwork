
#ifndef NEURALNETWORK_BASENEURON_H
#define NEURALNETWORK_BASENEURON_H

#include <vector>
#include "BaseEdge.h"

template <class EdgeType> class BaseEdge;   /// say that this class exists but don't declare what's inside
                                            /// this is needed in order to be able to keep a pointer inside BaseNeuron
                                            /// and to keep a pointer of BaseNeuron inside BaseEdge as without this it's a compile error

/**
 * Base ABSTRACT class for a neuron
 * @Lifecycle:
 *      1. onPreActivation
 *      2. onActivation
 *      3. onPostActivation
 *      4. onCalculateLoss
 *      5. onPreBackpropagation
 *      6. onBackpropagation
 *      7. onPostBackpropagation
 *
 * @Contains:
 *      1. edges to the next layer
 *      2. edges to the previous layer
 *      3. activatedValue
 *      4. preActivatedValue
 *      5. loss
 */
template <class NeuronType>
class BaseNeuron {

protected:
    std::vector < BaseEdge <NeuronType>* > next;        /// all the connections to the neuron from the next layer
    std::vector < BaseEdge <NeuronType>* > previous;    /// all the connections to the neuron from the previous layer

    NeuronType activatedValue;      /// value after activating the neuron
    NeuronType preActivatedValue;   /// value before activating the neuron
    NeuronType loss;                /// loss calculated during backpropagation


public:
    BaseNeuron();
    BaseNeuron( const std::vector < BaseEdge <NeuronType>* >& next, const std::vector < BaseEdge <NeuronType>* >& previous );
    virtual ~BaseNeuron();

    virtual NeuronType activation( NeuronType x ) = 0;              /// activation function of the neuron
    virtual NeuronType activationDerivative( NeuronType x ) = 0;    /// derivative of the activation function of the neuron


    inline NeuronType getActivatedValue() const     { return activatedValue; }      /// get the activated value
    inline NeuronType getPreActivatedValue() const  { return preActivatedValue; }   /// get pre activated value (i.e. sum of [values of neurons from previous layer * weights connected to them ] )
    inline NeuronType getLoss() const               { return loss; }

    virtual void addNextLayerConnection( BaseEdge <NeuronType>* edge ) { next.push_back( edge ); }
    virtual void addPreviousLayerConnection( BaseEdge <NeuronType> * edge ) { previous.push_back( edge ); }



    /**
     * Called right before activating the neuron
     */
    virtual void onPreActivation() {};

    /**
     * Called to activate the neuron
     * calculates the sum of [values of neurons from previous layer * weights connected to them ]
     * updates the values activatedValue and preActivatedValue
     */
    virtual void onActivation();

    /**
     * called after activating the neuron
     */
    virtual void onPostActivation() {};


    /**
     * called to calculate the loss for the neuron
     */
    virtual void onCalculateLoss() {};


    /**
     * Called right before beckpropagating the neuron
     */
    virtual void onPreBackpropagation() {};

    /**
     * Called to beckpropagating the neuron
     */
    virtual void onBackpropagation( NeuronType learningRate, int batchSize );

    /**
     * Called after beckpropagating the neuron
     */
    virtual void onPostBackpropagation() {};
};

#include "BaseNeuron.tpp"

#endif //NEURALNETWORK_BASENEURON_H
