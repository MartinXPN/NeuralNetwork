
#include "SharedEdge.h"

template <class WeightType>
SharedEdge::SharedEdge(BaseNeuron<WeightType> *from,
                       BaseNeuron<WeightType> *to,
                       int *numberOfUsages,
                       WeightType *weight,
                       WeightType *deltaWeight)
        : BaseEdge(from, to, weight, deltaWeight),
          numberOfUsages(numberOfUsages) {

}


template <class WeightType>
void SharedEdge :: updateWeight( WeightType coefficient ) {
    BaseEdge <WeightType> :: updateWeight( ( 1 / *numberOfUsages ) * coefficient );
}


template <class WeightType>
SharedEdge <WeightType> :: ~SharedEdge() {

    if( -- *numberOfUsages == 0 ) {
        delete numberOfUsages;
        delete weight;
        delete deltaWeight;
    }
}
