cmake_minimum_required(VERSION 3.6)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

set(SOURCE_FILES main.cpp)
add_executable(NeuralNetwork ${SOURCE_FILES})