
#ifndef NEURALNETWORK_CONVOLUTIONTESTCONNECTIONS_H
#define NEURALNETWORK_CONVOLUTIONTESTCONNECTIONS_H

#include "../../lib/Layers/SimpleLayers/Convolution.h"
#include "../../lib/Activations/SimpleActivations/ReLU.h"

void testConvolutionConnections() {
    Convolution <double> first( {2, 9, 9}, {3, 3, 3}, new ReLU <double>(), {} );
    Convolution <double> second( {2, 4, 4}, {2, 3, 3}, new ReLU <double>(), {&first}, {0, 2, 2} );

    first.createNeurons();
    second.createNeurons();

    second.connectNeurons();
}

#endif //NEURALNETWORK_CONVOLUTIONTESTCONNECTIONS_H
