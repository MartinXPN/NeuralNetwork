
#ifndef NEURALNETWORK_OUTPUTNEURONINITIALIZER_H
#define NEURALNETWORK_OUTPUTNEURONINITIALIZER_H


#include "NeuronInitializer.h"
#include "../../lossfunctions/base/BaseLossFunction.h"
#include "../../neurons/base/BaseOutputNeuron.h"

template <class NeuronType>
class OutputNeuronInitializer : public NeuronInitializer <NeuronType> {

protected:
    BaseLossFunction <NeuronType>* lossFunction;


public:
    OutputNeuronInitializer(BaseLossFunction <NeuronType>* lossFunction) : lossFunction(lossFunction) {
    }

    std::vector<BaseNeuron<NeuronType> *> createNeurons(size_t numberOfNeurons) override {
        std::vector< BaseNeuron<NeuronType>* > res;
        for( auto i=0; i < numberOfNeurons; ++i ) {
            res.push_back( new BaseOutputNeuron<NeuronType>(lossFunction) );
        }
        return res;
    }
};


#endif //NEURALNETWORK_OUTPUTNEURONINITIALIZER_H
