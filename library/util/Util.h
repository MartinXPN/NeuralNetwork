
#ifndef NEURALNETWORK_UTIL_H
#define NEURALNETWORK_UTIL_H

template <typename Base, typename T>
inline bool instanceof(const T*) {
    return std::is_base_of<Base, T>::value;
}

#endif //NEURALNETWORK_UTIL_H
