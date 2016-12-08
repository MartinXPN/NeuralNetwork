
#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H


#include <vector>
#include "../Neurons/BaseNeurons/BaseInputNeuron.h"
#include "../Neurons/BaseNeurons/BaseOutputNeuron.h"
#include "../Layers/BaseLayers/BaseLayer.h"
#include "../Layers/BaseLayers/BaseInputLayer.h"
#include "../Layers/BaseLayers/BaseHiddenLayer.h"
#include "../Layers/BaseLayers/BaseOutputLayer.h"


/**
 * Base class for Neural Network
 * Lifecycle
 *      1. create neurons
 *      2. connect neurons
 *      3. fill in buckets by time to visit
 *      4. start training
 */
template <class NetworkType>
class NeuralNetwork {

protected:
    std :: vector <BaseInputLayer> inputLayers;
    std :: vector <BaseHiddenLayer> hiddenLayers;
    std :: vector <BaseOutputLayer> outputLayer;

public:
    NeuralNetwork( std :: vector <BaseInputLayer> inputLayers,
                   std :: vector <BaseHiddenLayer> hiddenLayers,
                   std :: vector <BaseOutputLayer> outputLayers );


};



#include "NeuralNetwork.tpp"

#endif //NEURALNETWORK_NEURALNETWORK_H
