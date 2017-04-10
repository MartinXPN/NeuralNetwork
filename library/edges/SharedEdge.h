
#ifndef NEURALNETWORK_CONVOLUTIONALEDGE_H
#define NEURALNETWORK_CONVOLUTIONALEDGE_H


#include <cstdio>
#include "base/BaseEdge.h"


/**
 * Class for supporting connections that share weights
 * for example in convolutions
 */
template <class WeightType>
class SharedEdge : public BaseEdge <WeightType> {

protected:
    /// number of usages of the same weight in the network
    int* numberOfUsages;
    using BaseEdge <WeightType> :: weight;
    using BaseEdge <WeightType> :: deltaWeight;


public:
    /**
     * @param from: neuron connected to To with this edge
     * @param to: neuron connected to From with this edge
     * @param numberOfUsages: number of usages of the *weight in the network (i.e. how many times the weight was shared)
     * @param weight: weight of the edge
     * @param deltaWeight: buffer which helps to update the weight
     */
    SharedEdge(BaseNeuron<WeightType> *from,
               BaseNeuron<WeightType> *to,
               int* numberOfUsages,
               WeightType *weight = nullptr,
               WeightType *deltaWeight = nullptr);


    /**
     * Update the weight by subtracting deltaWeight from the current weight
     * @param coefficient may include regularization terms, learning rate etc...
     * We override updateWeight of the base class to be able to handle the update of shared weights
     */
    virtual void updateWeight(WeightType coefficient = 1) override;


    /**
     * Deletes all variables if the numberOfUsages of the weight is 0
     */
    virtual ~SharedEdge();
};

#include "SharedEdge.tpp"

#endif //NEURALNETWORK_CONVOLUTIONALEDGE_H
