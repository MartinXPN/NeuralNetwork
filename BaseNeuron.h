
#ifndef NEURALNETWORK_BASENEURON_H
#define NEURALNETWORK_BASENEURON_H

#include <vector>
using namespace std;

#include "BaseEdge.h"

template <class Type> class BaseEdge;   /// say that this class exists but don't declare what's inside
                                        /// this is needed in order to be able to keep a pointer inside BaseNeuron
                                        /// and to keep a pointer of BaseNeuron inside BaseEdge as without this it's a compile error

/**
 * Base class for a neuron
 * @Lifecycle:
 *     1. onPreActivation
 *     2. onActivation
 *     3. onPostActivation
 *
 * @Contains:
 *      1. edges to the layer before it
 *      2. edges to the layer after it
 *      3. activatedValue
 *      4. preActivatedValue
 */
template <class Type> class BaseNeuron {

protected:
    vector < BaseEdge <Type>* > after;     /// all the connections to the neuron from the next layer
    vector < BaseEdge <Type>* > before;    /// all the connections to the neuron from the previous layer

    Type activatedValue;            /// value after activating the neuron
    Type preActivatedValue;         /// value before activating the neuron


public:

    BaseNeuron( const vector < BaseEdge <Type>* >& after, const vector < BaseEdge <Type>* >& before );
    virtual ~BaseNeuron();

    inline Type getActivatedValue() const    { return activatedValue; }      /// get the activated value
    inline Type getPreActivatedValue() const { return preActivatedValue; }   /// get pre activated value (i.e. sum of [values of neurons from previous layer * weights connected to them ] )

    virtual Type activation( Type x ) = 0;              /// activation function of the neuron
    virtual Type activationDerivative( Type x ) = 0;    /// derivative of the activation function of the neuron


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

#endif //NEURALNETWORK_BASENEURON_H
