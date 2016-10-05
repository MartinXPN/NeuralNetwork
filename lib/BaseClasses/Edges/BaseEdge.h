
#ifndef NEURALNETWORK_BASEEDGE_H
#define NEURALNETWORK_BASEEDGE_H


#include "../Neurons/BaseNeuron.h"

template <class NeuronType>
class BaseNeuron;   /// say that this class exists but don't declare what's inside
                    /// this is needed in order to be able to keep a pointer inside BaseEdge
                    /// and to keep a pointer of BaseEdge inside BaseNeuron as without this it's a compile error

/**
 *            Weight
 * (From) --------------> (To)
 *
 * Base class for keeping the edges of the network
 */
template <class WeightType>
class BaseEdge {

protected:
    const BaseNeuron <WeightType> *from;  /// pointer to Neuron | there are no setters for the Neuron from as it has to be given in the constructor
    const BaseNeuron <WeightType> *to;    /// pointer to Neuron | there are no setters for the Neuron to as it has to be given in the constructor
    WeightType *weight;         /// weight of the Edge
    WeightType *deltaWeight;    /// how much to update weight on backpropagation

public:
    BaseEdge( BaseNeuron <WeightType> * from, BaseNeuron <WeightType>* to, WeightType* weight );
    virtual ~BaseEdge();

    /**
     * Get const reference of the 'from' neuron
     */
    inline const BaseNeuron <WeightType>& getFrom() const { return *from; }
    /**
     * Get const reference of the 'to' neuron
     */
    inline const BaseNeuron <WeightType>& getTo() const   { return *to; }


    /**
     * Get instance of the weight
     */
    inline const WeightType& getWeight() const      { return *weight; }
    /**
     * Set the value of the weight
     */
    inline void setWeight( WeightType weight )      { *(this -> weight) = weight; }


    /**
     * Get instance of the delta weight
     */
    inline const WeightType& getDeltaWeight() const     { return *deltaWeight; }
    /**
     * Set the value of the weight
     */
    inline void setDeltaWeight( WeightType deltaWeight ) { *(this -> deltaWeight) = deltaWeight; }


    virtual void updateWeight();
};

#include "BaseEdge.tpp"

#endif //NEURALNETWORK_BASEEDGE_H
