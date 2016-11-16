
#ifndef NEURALNETWORK_CONVOLUTION2D_H
#define NEURALNETWORK_CONVOLUTION2D_H


#include <cassert>
#include <cstdio>
#include <cmath>
#include "../BaseLayers/BaseHiddenLayer.h"

template <class LayerType>
class Convolution : public BaseHiddenLayer <LayerType> {

protected:
    using BaseHiddenLayer <LayerType> :: neurons;
    using BaseHiddenLayer <LayerType> :: numberOfNeurons;
    using BaseHiddenLayer <LayerType> :: activationFunction;
    using BaseHiddenLayer <LayerType> :: bias;
    using BaseHiddenLayer <LayerType> :: previousLayers;

    unsigned numberOfFilters;               /// number of filters in this layer | example -> 10
    std :: vector <unsigned> dimensions;    /// number of rows, columns, etc.   | example {100, 200} -> 100 rows 200 columns
    std :: vector <unsigned> windowSize;    /// size of the sliding window      | example {2, 3}
    std :: vector <unsigned> stride;        /// step of the sliding window      | example {1, 3} -> step 1 in direction of rows and 3 in direction of columns

    void connectOne( int dimension );

public:
    Convolution( std :: vector <unsigned> dimensions,
                 std :: vector <unsigned> windowSize,
                 BaseActivationFunction<LayerType> *activationFunction,
                 const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                 std :: vector <unsigned> stride = {},
                 std :: vector <unsigned> padding = {},
                 Bias<LayerType> *bias = nullptr );


    virtual void createNeurons() override;

    virtual void connectNeurons() override;

    virtual ~Convolution() {}
};

#include "Convolution.tpp"

#endif //NEURALNETWORK_CONVOLUTION2D_H
