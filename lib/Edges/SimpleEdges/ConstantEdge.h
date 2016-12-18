
#ifndef NEURALNETWORK_CONSTANTEDGE_H
#define NEURALNETWORK_CONSTANTEDGE_H


#include "../BaseEdges/BaseEdge.h"

template <class EdgeType>
class ConstantEdge : public BaseEdge <EdgeType> {

public:
    ConstantEdge <EdgeType> ( BaseNeuron<EdgeType> *from, BaseNeuron<EdgeType> *to, EdgeType weight = 1 )
            : BaseEdge(from, to, new EdgeType( weight ), nullptr) {}

    void updateWeight(EdgeType coefficient) override {
        /// do nothing
    }
};


#endif //NEURALNETWORK_CONSTANTEDGE_H
