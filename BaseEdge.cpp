
#include "BaseEdge.h"



template <class Type> BaseEdge::BaseEdge(BaseNeuron *from, BaseNeuron *to, const Type &weight){

    this -> from = from;
    this -> to = to;
    this -> weight = weight;
}

template <class Type> BaseEdge::BaseEdge(BaseNeuron from, BaseNeuron to, const Type &weight) {

    *( this -> from ) = from;
    *( this -> to ) = to;
    this -> weight = weight;
}

