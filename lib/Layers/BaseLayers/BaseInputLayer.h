
#ifndef NEURALNETWORK_BASEINPUTLAYER_H
#define NEURALNETWORK_BASEINPUTLAYER_H


#include "BaseLayer.h"
#include "../../Neurons/BaseNeurons/BaseInputNeuron.h"

template <class LayerType>
class BaseInputLayer : public BaseLayer <LayerType> {

private:
    virtual void connectNeurons( const BaseLayer<LayerType>& previous) override {}

protected:
    using BaseLayer <LayerType> :: neurons;

public:
    BaseInputLayer(unsigned int numberOfNeurons ) :
            BaseLayer <LayerType> ( numberOfNeurons, {} ) {

    }


    virtual void createNeurons(unsigned numberOfNeurons) override;

};

#include "BaseInputLayer.tpp"

#endif //NEURALNETWORK_BASEINPUTLAYER_H
