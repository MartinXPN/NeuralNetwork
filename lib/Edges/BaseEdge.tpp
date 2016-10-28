
#include "BaseEdge.h"


template <class WeightType>
BaseEdge <WeightType> :: BaseEdge(BaseNeuron<WeightType> *from,
                                  BaseNeuron<WeightType> *to,
                                  WeightType *weight,
                                  WeightType *deltaWeight ) :
        from(from), to(to) {

    this -> weight = weight;
    if( deltaWeight == nullptr )    this -> deltaWeight = new WeightType( 0 );
    else                            this -> deltaWeight = deltaWeight;
}


template <class WeightType>
BaseEdge <WeightType> :: ~BaseEdge() {
    /// can't delete weigh and deltaWeight in cases they are shared
//    delete weight;
//    delete deltaWeight;
}


template <class WeightType>
void BaseEdge <WeightType> :: updateWeight() {

    *weight -= *deltaWeight;
    *deltaWeight = 0;
}
