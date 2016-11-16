
#ifndef NEURALNETWORK_BASEINPUTLAYER_H
#define NEURALNETWORK_BASEINPUTLAYER_H


#include "BaseLayer.h"
#include "../../Neurons/BaseNeurons/BaseInputNeuron.h"


/**
 * Base class of Input layers
 * Its only responsibility is to create neurons and populate the collection - vector <> neurons
 */
template <class LayerType>
class BaseInputLayer : public BaseLayer <LayerType> {

private:
    /// Hide this method from subclasses
    virtual void connectNeurons() override {}

protected:
    using BaseLayer <LayerType> :: neurons;
    using BaseLayer <LayerType> :: numberOfNeurons;

public:
    BaseInputLayer(const std :: vector <unsigned>& dimensions ) :
            BaseLayer <LayerType> ( dimensions, {} ) {

    }


    virtual void createNeurons() override;

};

#include "BaseInputLayer.tpp"

#endif //NEURALNETWORK_BASEINPUTLAYER_H
