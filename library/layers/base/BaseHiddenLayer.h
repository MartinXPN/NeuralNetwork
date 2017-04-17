
#ifndef NEURALNETWORK_BASEHIDDENLAYER_H
#define NEURALNETWORK_BASEHIDDENLAYER_H


#include "BaseLayer.h"
#include "../../neurons/Bias.h"
#include "../../activations/base/BaseActivationFunction.h"
#include "../../initializers/neuron/NeuronInitializer.h"

/**
 * Base ABSTRACT class for all hidden layers
 * Any class inherited from BaseHiddenLayer has to implement two functions:
 *      1. createNeurons
 *      2. createWeights
 *      3. connectNeurons
 */
template <class LayerType>
class BaseHiddenLayer : public BaseLayer <LayerType> {

private:

protected:
    using BaseLayer <LayerType> :: neurons;
    using BaseLayer <LayerType> :: previousLayers;
    std :: vector <LayerType*> weights;
    std :: vector <LayerType*> deltaWeights;
    Bias <LayerType>* bias;


    virtual void connectNeurons(BaseNeuron<LayerType>* source,
                                BaseNeuron<LayerType>* target,
                                LayerType* weight,
                                LayerType* deltaWeight = nullptr);

public:
    BaseHiddenLayer(const std :: vector <unsigned>& dimensions,
                    const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                    NeuronInitializer <LayerType>* neuronInitializer,
                    Bias <LayerType>* bias = nullptr );

    BaseHiddenLayer(const std :: vector <unsigned>& dimensions,
                    BaseActivationFunction <LayerType>* activationFunction,
                    const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                    Bias <LayerType>* bias = nullptr );


    BaseHiddenLayer(const std :: vector <unsigned>& dimensions,
                    const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                    const std :: vector< BaseNeuron <LayerType>* >& neurons,
                    Bias <LayerType>* bias = nullptr );

    using BaseLayer <LayerType> :: connectNeurons;

    virtual void createWeights() = 0;

    virtual ~BaseHiddenLayer();
};

#include "BaseHiddenLayer.tpp"

#endif //NEURALNETWORK_BASEHIDDENLAYER_H
