
#ifndef NEURALNETWORK_BASEEDGE_H
#define NEURALNETWORK_BASEEDGE_H


#include "../Neurons/BaseNeurons/BaseNeuron.h"

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
    WeightType *weight;         /// weight of the Edge                              | we keep pointer because weights can be shared (in convolutions for instance)
    WeightType *deltaWeight;    /// how much to update weight on backpropagation    | we keep pointer because weights can be shared (in convolutions for isntance)

public:
    /**
     * @param from: Neuron which is connected by this edge to To
     * @param to: Neuron which is connected by this edge to From
     * @param weight: weight of this edge. It's pointer because weights can be shared sometimes (Convolutions for example)
     * @param deltaWeight: buffer that is responsible for the update of the weight. It's pointer because weights can be shared sometimes (Convolutions for example)
     */
    BaseEdge( BaseNeuron <WeightType> * from, BaseNeuron <WeightType>* to, WeightType* weight, WeightType* deltaWeight = nullptr );
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


    /**
     * Update the weight by subtracting deltaWeight from the current weight
     */
    virtual void updateWeight();
};

#include "BaseEdge.tpp"

#endif //NEURALNETWORK_BASEEDGE_H
