
#ifndef NEURALNETWORK_CONVOLUTIONALEDGE_H
#define NEURALNETWORK_CONVOLUTIONALEDGE_H


#include "../BaseEdges/BaseEdge.h"

template <class WeightType>
class ConvolutionalEdge : public BaseEdge <WeightType> {
public:
    ConvolutionalEdge(BaseNeuron<WeightType> *from,
                      BaseNeuron<WeightType> *to,
                      WeightType *weight = nullptr,
                      WeightType *deltaWeight = nullptr)
            : BaseEdge(from, to, weight, deltaWeight) {

    }

};


#endif //NEURALNETWORK_CONVOLUTIONALEDGE_H
