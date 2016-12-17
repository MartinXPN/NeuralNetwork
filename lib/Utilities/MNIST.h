
#ifndef NEURALNETWORK_MNIST_H
#define NEURALNETWORK_MNIST_H

#include <iostream>
#include <vector>
#include <fstream>


namespace MNIST {
    using namespace std;

    int reverseInt(int i) {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = (unsigned char) (i & 255);
        ch2 = (unsigned char) ((i >> 8) & 255);
        ch3 = (unsigned char) ((i >> 16) & 255);
        ch4 = (unsigned char) ((i >> 24) & 255);
        return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
    }

    vector<vector<double> > readImages(string directory, size_t numberOfImages, size_t imageSize) {

        vector<vector<double> > images(numberOfImages, vector<double>(imageSize));
        ifstream file(directory, ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read((char *) &magic_number, sizeof(magic_number));            magic_number = reverseInt(magic_number);
            file.read((char *) &number_of_images, sizeof(number_of_images));    number_of_images = reverseInt(number_of_images);
            file.read((char *) &n_rows, sizeof(n_rows));                        n_rows = reverseInt(n_rows);
            file.read((char *) &n_cols, sizeof(n_cols));                        n_cols = reverseInt(n_cols);

            images.resize((size_t) number_of_images);
            for (int i = 0; i < min(number_of_images, (const int &) numberOfImages); ++i) {
                for (int r = 0; r < n_rows; ++r)
                    for (int c = 0; c < n_cols; ++c) {
                        unsigned char temp = 0;
                        file.read((char *) &temp, sizeof(temp));
                        images[i][(n_rows * r) + c] = (double) temp;
                    }
            }
            cout << "Finished reading images" << endl;
        }
        else {
            cout << "Couldn't find the directory: " << directory << endl;
        }
        file.close();

        /// make the range of the data [0;1]
        for( int i=0; i < images.size(); ++i )
            for( int j=0; j < images[i].size(); ++j )
                images[i][j] /= 255.;

        return images;
    }

    vector<int> readLabels(string directory, int numberOfLabels) {

        vector<int> labels((size_t) numberOfLabels);
        ifstream file(directory, ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            file.read((char *) &magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);
            if (magic_number != 2049)
                throw runtime_error("Invalid MNIST label file!");
            file.read((char *) &numberOfLabels, sizeof(numberOfLabels)), numberOfLabels = reverseInt(numberOfLabels);
            for (int i = 0; i < numberOfLabels; i++)
                file.read((char *) &labels[i], 1);
        }
        else {
            cout << "Couldn't find the directory: " << directory << endl;
        }

        file.close();
        return labels;
    }

    vector <double> toLabelVector( int label ) {
        vector <double> res( 10, 0 );
        res[label] = 1;
        return res;
    }

    vector< vector <double> > toLabelMatrix( vector <int> labels ) {
        vector< vector <double> > res( labels.size(), vector <double> ( 10, 0 ) );
        for( int i=0; i < labels.size(); ++i )
            res[i][ labels[i] ] = 1;
        return res;
    }

    void printImage( const vector <double>& image ) {

        for( int i=0; i < 28; ++i, printf( "\n" ) )
            for( int j=0; j < 28; ++j ) {
                double current_number = image[ i * 28 + j ];
                if( current_number != 0. )  printf("%.1lf  ", current_number);
                else                        printf( "    " );
            }

        fflush( stdout );
    }
}

#endif //NEURALNETWORK_MNIST_H
