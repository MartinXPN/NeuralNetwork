
#include "NeuralNetwork.h"

template <class NetworkType>
NeuralNetwork <NetworkType> ::NeuralNetwork(std::vector<BaseInputLayer> inputLayers,
                                            std::vector<BaseHiddenLayer> hiddenLayers,
                                            std::vector<BaseOutputLayer> outputLayers) {

    this -> inputLayers = inputLayers;
    this -> hiddenLayers = hiddenLayers;
    this -> outputLayer = outputLayers;
}