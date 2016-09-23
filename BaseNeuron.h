
#ifndef NEURALNETWORK_BASENEURON_H
#define NEURALNETWORK_BASENEURON_H

#include <vector>
using namespace std;

#include "BaseEdge.h"

template <class Type> class BaseNeuron {

protected:
    vector < BaseEdge* > after;     /// all the connections to the neuron from the next layer
    vector < BaseEdge* > before;    /// all the connections to the neuron from the previous layer

    Type activatedValue;            /// value after activating the neuron
    Type preActivatedValue;         /// value before activating the neuron


public:

    Type getActivatedValue()    { return activatedValue; }      /// get the activated value
    Type getPreActivatedValue() { return preActivatedValue; }   /// get pre activated value (i.e. sum of [values of neurons from previous layer * weights connecting to them ] )

    virtual Type activation() = 0;              /// activation function of the neuron
    virtual Type activationDerivative() = 0;    /// derivative of the activation function of the neuron

    /// Lifecycle of the neuron is defined in these 3 steps
    virtual void onPreActivation() = 0;         /// called before activating the neuron
    virtual void onActivation() = 0;            /// called to activate the neuron
    virtual void onPostActivation() = 0;        /// called after activating the neuron
};


#endif //NEURALNETWORK_BASENEURON_H
