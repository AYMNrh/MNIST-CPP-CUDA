// data.h for data.cpp

#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>

void splitTrainTest(const std::vector<std::vector<double>> &data, const std::vector<int> &labels, std::vector<std::vector<double>> &trainData, std::vector<int> &trainLabels, std::vector<std::vector<double>> &testData, std::vector<int> &testLabels, double ratio);
void saveData(const std::vector<std::vector<double>> &data, const std::vector<int> &labels, const std::string &filename);