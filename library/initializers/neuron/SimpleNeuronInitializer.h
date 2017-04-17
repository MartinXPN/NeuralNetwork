

#ifndef NEURALNETWORK_SIMPLENEURONINITIALIZER_H
#define NEURALNETWORK_SIMPLENEURONINITIALIZER_H


#include "NeuronInitializer.h"

template <class NeuronType>
class SimpleNeuronInitializer : public NeuronInitializer<NeuronType> {

protected:
    BaseActivationFunction <NeuronType>* activationFunction;

public:

    SimpleNeuronInitializer(BaseActivationFunction <NeuronType>* activationFunction)
            : activationFunction(activationFunction) {
    }

    std::vector< BaseNeuron<NeuronType>* > createNeurons(size_t numberOfNeurons) override {
        std::vector< BaseNeuron<NeuronType>* > res;
        for( auto i=0; i < numberOfNeurons; ++i ) {
            res.push_back( new BaseNeuron<NeuronType>(activationFunction) );
        }
        return res;
    }
};


#endif //NEURALNETWORK_SIMPLENEURONINITIALIZER_H
