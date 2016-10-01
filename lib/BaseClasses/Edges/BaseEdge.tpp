
#include "BaseEdge.h"


template <class WeightType>
BaseEdge <WeightType> :: BaseEdge(BaseNeuron<WeightType> *from,
                                  BaseNeuron<WeightType> *to,
                                  WeightType *weight) :
        from(from), to(to) {

    this -> weight = weight;
    this -> deltaWeight = new WeightType( 0 );
}


template <class WeightType>
BaseEdge <WeightType> :: ~BaseEdge() {
    delete weight;
    delete deltaWeight;
}


template <class WeightType>
void BaseEdge <WeightType> :: updateWeight() {

    *weight -= *deltaWeight;
    *deltaWeight = 0;
}
