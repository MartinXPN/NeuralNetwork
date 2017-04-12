
#ifndef NEURALNETWORK_ONETOONE_H
#define NEURALNETWORK_ONETOONE_H

#include <cstdlib>
#include "base/BaseHiddenLayer.h"


template <class LayerType>
class OneToOne : public BaseHiddenLayer <LayerType> {
protected:
    using BaseHiddenLayer <LayerType> :: neurons;
    using BaseHiddenLayer <LayerType> :: weights;
    using BaseHiddenLayer <LayerType> :: deltaWeights;
    using BaseHiddenLayer <LayerType> :: previousLayers;


public:
    using BaseHiddenLayer <LayerType> :: BaseHiddenLayer;

    virtual void createWeights();

    void connectNeurons() override;
};


#include "OneToOne.tpp"

#endif //NEURALNETWORK_ONETOONE_H
