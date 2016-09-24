
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
 *
 * @Contains:
 *      1. edges to the next layer
 *      2. edges to the previous layer
 *      3. activatedValue
 *      4. preActivatedValue
 */
template <class NeuronType> class BaseNeuron {

protected:
    std::vector < BaseEdge <NeuronType>* > next;        /// all the connections to the neuron from the next layer
    std::vector < BaseEdge <NeuronType>* > previous;    /// all the connections to the neuron from the previous layer

    NeuronType activatedValue;      /// value after activating the neuron
    NeuronType preActivatedValue;   /// value before activating the neuron


public:
    BaseNeuron();
    BaseNeuron( const std::vector < BaseEdge <NeuronType>* >& next, const std::vector < BaseEdge <NeuronType>* >& previous );
    virtual ~BaseNeuron();

    virtual NeuronType activation( NeuronType x ) = 0;              /// activation function of the neuron
    virtual NeuronType activationDerivative( NeuronType x ) = 0;    /// derivative of the activation function of the neuron


    inline NeuronType getActivatedValue() const    { return activatedValue; }      /// get the activated value
    inline NeuronType getPreActivatedValue() const { return preActivatedValue; }   /// get pre activated value (i.e. sum of [values of neurons from previous layer * weights connected to them ] )

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
};



template <class NeuronType>
BaseNeuron <NeuronType> :: BaseNeuron() {}


template <class NeuronType>
BaseNeuron <NeuronType> :: BaseNeuron( const std::vector < BaseEdge <NeuronType>* >& next,
                                       const std::vector < BaseEdge <NeuronType>* >& previous ):
        next( next ), previous( previous ) {}


template <class NeuronType>
BaseNeuron <NeuronType> :: ~BaseNeuron() {

    for( auto edge : next )     delete edge;
    for( auto edge : previous ) delete edge;
}

template <class NeuronType>
void BaseNeuron <NeuronType> :: onActivation() {

    /// preactivatedValue = sum of [values of neurons from previous layer * weights connected to them ]
    preActivatedValue = 0;
    for( auto edge : previous ) {
        preActivatedValue += *(edge -> getWeight()) *
                             edge -> getFrom() -> getActivatedValue();
    }
    activatedValue = activation( preActivatedValue );
}

#endif //NEURALNETWORK_BASENEURON_H
