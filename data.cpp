// data procssing functions: split train and test, load data, save data, etc.

#include "data.h"
#include <iostream>
using namespace std;



void splitTrainTest(const vector<vector<double>>& data, const vector<int>& labels, vector<vector<double>>& trainData, vector<int>& trainLabels, vector<vector<double>>& testData, vector<int>& testLabels, double ratio) {
    // Split the data into train and test sets
    // The ratio is the percentage of the data that will be used for training
    // The rest will be used for testing
    // The data and labels are split in the same way
    // The data and labels are split in the same way
    int n = data.size();
    int nTrain = n * ratio;
    for (int i = 0; i < nTrain; i++) {
        trainData.push_back(data[i]);
        trainLabels.push_back(labels[i]);
    }
    for (int i = nTrain; i < n; i++) {
        testData.push_back(data[i]);
        testLabels.push_back(labels[i]);
    }
}

void saveData(const vector<vector<double>>& data, const vector<int>& labels, const string& filename) {
    // Save the data and labels to a file
    // The file format is as follows:
    // The first line contains the number of data points and the number of features
    // Each subsequent line contains the features of a data point followed by the label
    // The features and label are separated by a space
    ofstream file(filename);
    if (file.is_open()) {
        file << data.size() << " " << data[0].size() << endl;
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[i].size(); j++) {
                file << data[i][j] << " ";
            }
            file << labels[i] << endl;
        }
        file.close();
    }
    else {
        cout << "Unable to open file";
    }
}


