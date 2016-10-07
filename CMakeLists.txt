cmake_minimum_required(VERSION 3.6)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

set(SOURCE_FILES main.cpp lib/BaseClasses/Neurons/BaseNeuron.h lib/BaseClasses/Edges/BaseEdge.h lib/BaseClasses/Edges/BaseEdge.tpp lib/BaseClasses/Neurons/BaseNeuron.tpp lib/BaseClasses/Neurons/BaseInputNeuron.tpp lib/BaseClasses/Neurons/BaseInputNeuron.h lib/BaseClasses/Neurons/BaseOutputNeuron.tpp lib/BaseClasses/Neurons/BaseOutputNeuron.h lib/Activations/BaseActivation/BaseActivationFunction.h lib/Activations/SimpleActivations/Sigmoid.h lib/Activations/SimpleActivations/ReLU.h lib/Activations/SimpleActivations/Tanh.h lib/Activations/SimpleActivations/Identitiy.h lib/Activations/SimpleActivations/ELU.h lib/LossFunctions/BaseLoss/BaseLossFunction.h lib/LossFunctions/SimpleLossFunctions/MeanSquaredError.h lib/LossFunctions/SimpleLossFunctions/CrossEntropyCost.h)
add_executable(NeuralNetwork ${SOURCE_FILES})
