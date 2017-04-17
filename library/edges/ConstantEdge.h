
#ifndef NEURALNETWORK_CONSTANTEDGE_H
#define NEURALNETWORK_CONSTANTEDGE_H


#include "base/BaseEdge.h"

template <class EdgeType>
class ConstantEdge : public BaseEdge <EdgeType> {

public:
    ConstantEdge <EdgeType> ( BaseNeuron<EdgeType> *from, BaseNeuron<EdgeType> *to, EdgeType* weight )
            : BaseEdge <EdgeType> (from, to, weight, new EdgeType( 0 ) ) {}

    void updateWeight(EdgeType coefficient) override {
        /// do nothing
    }
};


#endif //NEURALNETWORK_CONSTANTEDGE_H
