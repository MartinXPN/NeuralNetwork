
#include "BaseEdge.h"


template <class Type>
BaseEdge <Type> :: BaseEdge(BaseNeuron<Type> *from, BaseNeuron<Type> *to, const Type &weight) {
    this -> from = from;
    this -> to = to;
    this -> weight = weight;
}

template <class Type>
BaseEdge <Type> :: BaseEdge(BaseNeuron<Type> from, BaseNeuron<Type> to, const Type &weight) {
    *( this -> from ) = from;
    *( this -> to ) = to;
    this -> weight = weight;
}

template <class Type>
BaseEdge <Type> :: ~BaseEdge() {
    delete weight;
}
