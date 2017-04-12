
#ifndef NEURALNETWORK_INPUTNEURONINITIALIZER_H
#define NEURALNETWORK_INPUTNEURONINITIALIZER_H


#include "NeuronInitializer.h"
#include "../../neurons/base/BaseInputNeuron.h"

template <class NeuronType>
class InputNeuronInitializer : public NeuronInitializer<NeuronType> {

public:
    std::vector< BaseNeuron<NeuronType>* > createNeurons(size_t numberOfNeurons) override {
        std::vector< BaseNeuron<NeuronType>* > res;
        for( auto i=0; i < numberOfNeurons; ++i ) {
            res.push_back( new BaseInputNeuron<NeuronType>() );
        }
        return res;
    }
};

#endif //NEURALNETWORK_INPUTNEURONINITIALIZER_H
