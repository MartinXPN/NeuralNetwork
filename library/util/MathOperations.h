
#ifndef NEURALNETWORK_MATHOPERATIONS_H
#define NEURALNETWORK_MATHOPERATIONS_H

#include <vector>
#include <cstddef>

namespace math {

    unsigned multiply( const std::vector<unsigned> v, size_t start, size_t end ) {
        unsigned res = 1;
        for( auto i=start; i < end; ++i )
            res *= v[i];
        return res;
    }
    unsigned multiply( const std::vector<unsigned> v, size_t start = 0 ) {
        return multiply(v, start, v.size());
    }
}

#endif //NEURALNETWORK_MATHOPERATIONS_H
