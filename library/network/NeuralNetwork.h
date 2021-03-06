
#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H


#include <vector>
#include <functional>
#include "../layers/InputLayer.h"
#include "../layers/base/BaseHiddenLayer.h"
#include "../layers/LossLayer.h"


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
    std :: vector <InputLayer <NetworkType>* > inputLayers;
    std :: vector <BaseHiddenLayer <NetworkType>* > hiddenLayers;
    std :: vector <LossLayer <NetworkType>* > outputLayers;

    std :: vector <BaseInputNeuron <NetworkType>* > inputNeurons;
    std :: vector< std :: vector <BaseNeuron <NetworkType>* > > buckets;
    std :: vector <BaseOutputNeuron <NetworkType>* > outputNeurons;

    double learningRate;
    double batchLoss;
    double epochLoss;


public:
    NeuralNetwork( std :: vector <InputLayer <NetworkType>* > inputLayers,
                   std :: vector <BaseHiddenLayer <NetworkType>* > hiddenLayers,
                   std :: vector <LossLayer <NetworkType>* > outputLayers );


    const std::vector< std :: vector <BaseNeuron <NetworkType>* > >& getBuckets() {
        return buckets;
    }
    virtual double getLearningRate()    { return learningRate; }
    virtual double getBatchLoss()       { return batchLoss; }
    virtual double getEpochLoss()       { return epochLoss; }


    virtual void initializeNetwork();
    virtual void calculatePropagationOrder();

    virtual std :: vector< BaseNeuron <NetworkType>* > getInputNeuronsAndBiases();
    virtual std :: vector< std :: vector <BaseNeuron <NetworkType>* > > divideIntoBuckets( std :: vector< BaseNeuron <NetworkType>* > startOffNeurons );
    virtual void trainEpoch( size_t numberOfInputs,
                             size_t batchSize,
                             double learningRate,
                             std::function< std::vector <NetworkType> (size_t itemNumber) > inputLoader,
                             std::function< std::vector <NetworkType> (size_t itemNumber) > labelLoader,
                             std::function< void() > onEpochTrained = nullptr,
                             std::function< void() > onBatchProcessed = nullptr );

    virtual std::vector <NetworkType> evaluateOne( std::vector <NetworkType> input,
                                                   std::function< void (const std::vector <NetworkType>&) > onEvaluated );
};



#include "NeuralNetwork.tpp"

#endif //NEURALNETWORK_NEURALNETWORK_H
