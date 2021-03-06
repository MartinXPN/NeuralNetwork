
#ifndef NEURALNETWORK_CONVOLUTION_H
#define NEURALNETWORK_CONVOLUTION_H


#include <cassert>
#include <cstdio>
#include <cmath>
#include "base/BaseHiddenLayer.h"
#include "../initializers/neuron/SimpleNeuronInitializer.h"
#include "../activations/ReLU.h"

template <class LayerType>
class Convolution : public BaseHiddenLayer <LayerType> {

protected:
    using BaseHiddenLayer <LayerType> :: neurons;
    using BaseHiddenLayer <LayerType> :: bias;
    using BaseHiddenLayer <LayerType> :: previousLayers;
    using BaseHiddenLayer <LayerType> :: dimensions;

    std :: vector <unsigned> kernel;        /// size of the sliding window      | example {2, 3}
    std :: vector <unsigned> stride;        /// step of the sliding window      | example {1, 3} -> step 1 in direction of rows and 3 in direction of columns
    std :: vector <LayerType*> weights;
    std :: vector <LayerType*> deltaWeights;
    std :: vector <int*> numberOfUsages;     /// number of the usages of the same weight

    virtual void connectTwoNeurons( BaseNeuron <LayerType>* previousNeuron,
                                    BaseNeuron <LayerType>* neuron,
                                    int weightIndex );

    virtual void connectOne( BaseNeuron <LayerType>* neuron,
                             const BaseLayer <LayerType>* previousLayer,
                             int previousLayerStart,
                             int currentDimension,
                             int weightIndex );

    virtual void connectLayer( const BaseLayer <LayerType>* previousLayer,
                               int currentLayerStart,
                               int previousLayerStart,
                               int currentDimension,
                               int weightIndex = 0 );

public:
    Convolution( std :: vector <unsigned> dimensions,
                 std :: vector <unsigned> kernel,
                 const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                 std :: vector <unsigned> stride = {},
                 NeuronInitializer<LayerType> *neuronInitializer = new SimpleNeuronInitializer <LayerType>( new ReLU <LayerType>() ),
                 Bias<LayerType> *bias = nullptr );


    Convolution( std :: vector <unsigned> dimensions,
                 std :: vector <unsigned> kernel,
                 BaseActivationFunction<LayerType> *activationFunction,
                 const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                 std :: vector <unsigned> stride = {},
                 Bias<LayerType> *bias = nullptr );


    Convolution( std :: vector <unsigned> dimensions,
                 std :: vector <unsigned> kernel,
                 std :: vector <BaseNeuron <LayerType>* > neurons,
                 const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                 std :: vector <unsigned> stride = {},
                 Bias<LayerType> *bias = nullptr );


    virtual void connectNeurons() override;

    virtual void createWeights() override;

    virtual ~Convolution() {}
};

#include "Convolution.tpp"

#endif //NEURALNETWORK_CONVOLUTION_H
