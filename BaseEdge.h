
#ifndef NEURALNETWORK_BASEEDGE_H
#define NEURALNETWORK_BASEEDGE_H


#include "BaseNeuron.h"

template <class Type> class BaseNeuron; /// say that this class exists but don't declare what's inside
                                        /// this is needed in order to be able to keep a pointer inside BaseEdge
                                        /// and to keep a pointer of BaseEdge inside BaseNeuron as without this it's a compile error

/**
 *            weight
 * (from) --------------> (to)
*/

template <class Type> class BaseEdge {

protected:
    BaseNeuron <Type> *from;    /// pointer to Neuron | there are no setters for the Neuron from as it has to be given in the constructor
    BaseNeuron <Type> *to;      /// pointer to Neuron | there are no setters for the Neuron to as it has to be given in the constructor
    Type *weight;               /// weight of the Edge

public:
    BaseEdge( BaseNeuron <Type> * from, BaseNeuron <Type>* to, const Type& weight );
    BaseEdge( BaseNeuron <Type> from, BaseNeuron <Type> to, const Type& weight );
    virtual ~BaseEdge();


    inline BaseNeuron <Type>* getFrom() const   { return from; }
    inline BaseNeuron <Type>* getTo() const     { return to; }

    inline Type* getWeight() const              { return weight; }
    void setWeight( Type weight )               { *(this -> weight) = weight; }
    void setWeight( const Type& weight )        { *(this -> weight) = weight; }
    void setWeight( Type *weight )              { this -> weight = weight; }
 };

#endif //NEURALNETWORK_BASEEDGE_H
