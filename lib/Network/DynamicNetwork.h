
#ifndef NEURALNETWORK_DYNAMICNETWORK_H
#define NEURALNETWORK_DYNAMICNETWORK_H


#include "NeuralNetwork.h"

template <class NetworkType>
class DynamicNetwork : public NeuralNetwork <NetworkType> {

protected:
    using NeuralNetwork <NetworkType> :: inputLayers;
    using NeuralNetwork <NetworkType> :: hiddenLayers;
    using NeuralNetwork <NetworkType> :: outputLayers;

    using NeuralNetwork <NetworkType> :: inputNeurons;
    using NeuralNetwork <NetworkType> :: buckets;
    using NeuralNetwork <NetworkType> :: outputNeurons;

    virtual void pruneNeuronPreviousLayerConnections( BaseNeuron<NetworkType> *neuron, NetworkType threshold );
    virtual void pruneNeuronNextLayerConnections( BaseNeuron<NetworkType> *neuron, NetworkType threshold );

public:
    using NeuralNetwork <NetworkType> :: NeuralNetwork;

    virtual size_t getSmallWeightsNumber( NetworkType threshold );
    virtual void pruneNetwork( NetworkType threshold );
    virtual void pruneLayers( NetworkType threshold, std::vector <BaseLayer <NetworkType>* > layers );
};

#include "DynamicNetwork.tpp"

#endif //NEURALNETWORK_DYNAMICNETWORK_H
