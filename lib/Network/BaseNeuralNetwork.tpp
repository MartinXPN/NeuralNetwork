
#include "BaseNeuralNetwork.h"


template <class NetworkType>
BaseNeuralNetwork <NetworkType> :: BaseNeuralNetwork( const std::vector<BaseInputNeuron<NetworkType> *> &inputNeurons,
                                                      const std::vector<BaseNeuron<NetworkType> *> &hiddenNeurons,
                                                      const std::vector<BaseOutputNeuron<NetworkType> *> &outputNeurons ) :
        inputNeurons(inputNeurons),
        hiddenNeurons(hiddenNeurons),
        outputNeurons(outputNeurons) {

}

template <class NetworkType>
BaseNeuralNetwork <NetworkType> :: BaseNeuralNetwork(const BaseLayer<NetworkType> &inputLayer,
                                                     const std::vector<BaseLayer<NetworkType> > &hiddenLayers,
                                                     const BaseLayer<NetworkType> &outputLayer) {

}
