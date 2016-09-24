
#ifndef NEURALNETWORK_BASEEDGE_H
#define NEURALNETWORK_BASEEDGE_H


#include "BaseNeuron.h"

template <class NeuronType> class BaseNeuron;   /// say that this class exists but don't declare what's inside
                                                /// this is needed in order to be able to keep a pointer inside BaseEdge
                                                /// and to keep a pointer of BaseEdge inside BaseNeuron as without this it's a compile error

/**
 *            Weight
 * (From) --------------> (To)
 *
 * Base class for keeping the edges of the network
 */
template <class WeightType> class BaseEdge {

protected:
    BaseNeuron <WeightType> *from;  /// pointer to Neuron | there are no setters for the Neuron from as it has to be given in the constructor
    BaseNeuron <WeightType> *to;    /// pointer to Neuron | there are no setters for the Neuron to as it has to be given in the constructor
    WeightType *weight;             /// weight of the Edge

public:
    BaseEdge( BaseNeuron <WeightType> * from, BaseNeuron <WeightType>* to, WeightType* weight );
    virtual ~BaseEdge();

    inline auto getFrom() const { return from; }
    inline auto getTo() const   { return to; }

    inline WeightType* getWeight() const              { return weight; }
    inline void setWeight( WeightType *weight )       { this -> weight = weight; }
    inline void setWeight( const WeightType& weight ) { *(this -> weight) = weight; }
 };



template <class WeightType>
BaseEdge <WeightType> :: BaseEdge(BaseNeuron<WeightType> *from,
                                  BaseNeuron<WeightType> *to,
                                  WeightType *weight) :
        from(from), to(to), weight(weight) {}


template <class WeightType>
BaseEdge <WeightType> :: ~BaseEdge() {
    delete weight;
}

#endif //NEURALNETWORK_BASEEDGE_H
