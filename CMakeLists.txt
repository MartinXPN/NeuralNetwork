cmake_minimum_required(VERSION 3.6)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

set(SOURCE_FILES main.cpp lib/Neurons/BaseNeurons/BaseNeuron.h lib/Edges/BaseEdge.h lib/Edges/BaseEdge.tpp lib/Neurons/BaseNeurons/BaseNeuron.tpp lib/Neurons/BaseNeurons/BaseInputNeuron.tpp lib/Neurons/BaseNeurons/BaseInputNeuron.h lib/Neurons/BaseNeurons/BaseOutputNeuron.tpp lib/Neurons/BaseNeurons/BaseOutputNeuron.h lib/Activations/BaseActivation/BaseActivationFunction.h lib/Activations/SimpleActivations/Sigmoid.h lib/Activations/SimpleActivations/ReLU.h lib/Activations/SimpleActivations/Tanh.h lib/Activations/SimpleActivations/Identitiy.h lib/Activations/SimpleActivations/ELU.h lib/LossFunctions/BaseLoss/BaseLossFunction.h lib/LossFunctions/SimpleLossFunctions/MeanSquaredError.h lib/LossFunctions/SimpleLossFunctions/CrossEntropyCost.h tests/TestNeurons/NeuronTest.h lib/Network/BaseNeuralNetwork.tpp lib/Network/BaseNeuralNetwork.h lib/Layers/BaseLayers/BaseLayer.tpp lib/Layers/BaseLayers/BaseLayer.h lib/Layers/SimpleLayers/FullyConnected.cpp lib/Layers/SimpleLayers/FullyConnected.h)
add_executable(NeuralNetwork ${SOURCE_FILES})
