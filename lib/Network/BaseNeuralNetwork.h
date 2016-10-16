
#ifndef NEURALNETWORK_BASENEURALNETWORK_H
#define NEURALNETWORK_BASENEURALNETWORK_H


#include <vector>
#include "../Neurons/BaseNeurons/BaseInputNeuron.h"
#include "../Neurons/BaseNeurons/BaseOutputNeuron.h"
#include "../Layers/BaseLayers/BaseLayer.h"

template <class NetworkType>
class BaseNeuralNetwork {

protected:
    std :: vector< BaseInputNeuron <NetworkType>* > inputNeurons;
    std :: vector< BaseNeuron <NetworkType>* > hiddenNeurons;
    std :: vector< BaseOutputNeuron <NetworkType>* > outputNeurons;

public:
    BaseNeuralNetwork( const std :: vector< BaseInputNeuron <NetworkType>* >& inputNeurons,
                       const std :: vector< BaseNeuron <NetworkType>* >& hiddenNeurons,
                       const std :: vector< BaseOutputNeuron <NetworkType>* >& outputNeurons );

    BaseNeuralNetwork( const BaseLayer <NetworkType>& inputLayer,
                       const std :: vector< BaseLayer <NetworkType> >& hiddenLayers,
                       const BaseLayer <NetworkType>& outputLayer );
};


#include "BaseNeuralNetwork.tpp"

#endif //NEURALNETWORK_BASENEURALNETWORK_H
