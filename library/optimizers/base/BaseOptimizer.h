

#ifndef NEURALNETWORK_BASEOPTIMIZER_H
#define NEURALNETWORK_BASEOPTIMIZER_H


#include "../../edges/base/BaseEdge.h"


/**
 * Base ABSTRACT class for optimizers
 * Optimizer is responsible for updating weights of edges
 * So in its getUpdate( edge ) function the update rule of the optimizer has to be defined
 */
template <class OptimizerType>
class BaseOptimizer {

public:
    virtual ~BaseOptimizer() = 0;

    /**
     * Defines the update rule of the optimizer
     * It has access to the specific edge on which it's making an update
     *
     */
    virtual OptimizerType getUpdate(BaseEdge <OptimizerType> edge) = 0;
};


#endif //NEURALNETWORK_BASEOPTIMIZER_H
