
#ifndef NEURALNETWORK_BASEEDGE_H
#define NEURALNETWORK_BASEEDGE_H


#include "BaseNeuron.h"


/**
 *            weight
 * (from) --------------> (to)
*/
template <class Type> class BaseEdge {

protected:
    BaseNeuron *from;   /// pointer to Neuron
    BaseNeuron *to;     /// pointer to Neuron
    Type weight;        /// weight of the Edge

public:

    BaseEdge( BaseNeuron* from, BaseNeuron* to, const Type& weight );
    BaseEdge( BaseNeuron from, BaseNeuron to, const Type& weight );


    BaseNeuron* getFrom()               { return from; }
    void setFrom( BaseNeuron* from )    { this -> from = from; }

    BaseNeuron* getTo()                 { return to; }
    void setTo( BaseNeuron* from )      { this -> to = to; }

    Type getWeight()                    { return weight; }
    void setWeight( Type weight )       { this -> weight = weight; }
};


#endif //NEURALNETWORK_BASEEDGE_H
