
#include "BaseNeuron.h"


template <class Type>
void BaseNeuron <Type> :: onActivation() {

    /// preactivatedValue = sum of [values of neurons from previous layer * weights connected to them ]
    preActivatedValue = 0;
    for( auto edge : before ) {
        preActivatedValue += edge -> getWeight() *
                             edge -> getFrom() -> getActivatedValue();
    }
    activatedValue = activation( preActivatedValue );
}


template <class Type>
BaseNeuron <Type> :: BaseNeuron(const vector< BaseEdge<Type>* > &after, const vector< BaseEdge<Type>* > &before) {

    this -> after = after;
    this -> before = before;
}


template <class Type>
BaseNeuron <Type> :: ~BaseNeuron() {

    for( auto edge : after )    delete edge;
    for( auto edge : before )   delete edge;
}
