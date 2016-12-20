
#ifndef NEURALNETWORK_CONSTANTEDGE_H
#define NEURALNETWORK_CONSTANTEDGE_H


#include "../BaseEdges/BaseEdge.h"

template <class EdgeType>
class ConstantEdge : public BaseEdge <EdgeType> {

public:
    ConstantEdge <EdgeType> ( BaseNeuron<EdgeType> *from, BaseNeuron<EdgeType> *to, EdgeType weight = 1 )
            : BaseEdge(from, to, new EdgeType( weight ), new EdgeType( 0 ) ) {}

    void updateWeight(EdgeType coefficient) override {
        /// do nothing
    }

    ~ConstantEdge() override {

        delete weight;
        delete deltaWeight;
    }
};


#endif //NEURALNETWORK_CONSTANTEDGE_H
