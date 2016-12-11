
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
    std :: vector <BaseInputLayer <NetworkType>* > inputLayers;
    std :: vector <BaseHiddenLayer <NetworkType>* > hiddenLayers;
    std :: vector <BaseOutputLayer <NetworkType>* > outputLayers;

    std :: vector< std :: vector <BaseNeuron <NetworkType>* > > buckets;

public:
    NeuralNetwork( std :: vector <BaseInputLayer <NetworkType>* > inputLayers,
                   std :: vector <BaseHiddenLayer <NetworkType>* > hiddenLayers,
                   std :: vector <BaseOutputLayer <NetworkType>* > outputLayers );


    const std::vector< std :: vector <BaseNeuron <NetworkType>* > >& getBuckets() {
        return buckets;
    }
    virtual void initializeNetwork();
    virtual void calculatePropagationOrder();

    virtual std :: vector< BaseNeuron <NetworkType>* > getInputNeuronsAndBiases();
    virtual std :: vector< std :: vector <BaseNeuron <NetworkType>* > > divideIntoBuckets( std :: vector< BaseNeuron <NetworkType>* > startOffNeurons );
    virtual void train();
};



#include "NeuralNetwork.tpp"

#endif //NEURALNETWORK_NEURALNETWORK_H
