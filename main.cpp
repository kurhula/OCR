//
//  main.cpp
//  Vision_project
//
//  Created by Mahmoud Khodary on 12/1/15.
//  Copyright (c) 2015 Mahmoud Khodary. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "CCL_code.h"
#include <vector>

#define LINE_THRESH 50

using namespace std;

struct Font{
	string name;
	int ch_count;
};

struct bvec{
	vector<Blob> v;
	bool operator < (const bvec& x){
		return (v[0].minY < x.v[0].minY);
	}
};

vector<cv::Mat> visionize(cv::Mat, bool, string, int);
int matchWindows(cv::Mat&, cv::Mat&);
void matchLetters(vector<cv::Mat>&, vector<cv::Mat>&, string&, int);
int minIndex(int[], int);
void init_templates(vector<cv::Mat>&, string, int);
void read_from_file(string, vector<Font>&);
void write_to_file(string, vector<Font>&);

void sortLines(vector<Blob>& blobs){
	vector<bvec> sorted;
	for(int i=0; i<blobs.size(); i++){

		int liny = blobs[i].maxY;
		bvec line_vec;						//the vector which has all objects on the same line
		line_vec.v.push_back(blobs[i]);
		blobs.erase(blobs.begin() + i);
		i--;

		for(int j=0; j<blobs.size(); j++){			//removes blobs in the same line from the blobs vector and inserts them in the line vec
			if(abs(blobs[j].maxY - liny) <= LINE_THRESH){
				line_vec.v.push_back(blobs[j]);
				blobs.erase(blobs.begin()+j);
				j--;
			}
		}

		sort(line_vec.v.begin(), line_vec.v.end());
		//for(int j=0; j<line_vec.size(); j++)
			//sorted.push_back(line_vec[j]);
		sorted.push_back(line_vec);

	}
	vector<Blob> results;
	sort(sorted.begin(), sorted.end());
	for(int i=0; i<sorted.size(); i++){
		for(int j=0; j<sorted[i].v.size(); j++){
			results.push_back(sorted[i].v[j]);
		}
	}
	blobs = results;
}

int main(int argc, const char * argv[]) {
	
	vector<cv::Mat> letters;
	vector<cv::Mat> templates;
	vector<Font> fonts;
	cv::Mat imColor;
	string result;

	if( argc != 2)
    {
		cout <<" No image argument is given to the program" << endl;
		system("pause");
		return -1;
	}
    cout << "Program has started!\n";

	imColor = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if(! imColor.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
		system("pause");
        return -1;
    }

	cout << "Please choose one of the following options:\n"<<
			"1. Carry out OCR using one of the existing fonts\n"<<
			"2. Import a new font for OCR\n";

	int choice;
	cin >> choice;
	bool import = choice-1;
	
	read_from_file("templates/Fonts.txt", fonts);
	string templates_dir;

	if(import){
		cout << "Please enter the name of the new font:\n";
		string input;
		cin >> input;
		templates_dir = "templates/" + input;

		Font temp;
		temp.name = input;
		temp.ch_count = 0;

		for(int i=0; i<3; i++){
			string samples[3] = {"templates/ASCII_1.png", "templates/ASCII_2.png", "templates/ASCII_3.png"};
			imColor = cv::imread(samples[i], CV_LOAD_IMAGE_COLOR);
			
			if(! imColor.data )                              // Check for invalid input
			{
				cout <<  "Could not open or find the image" << std::endl ;
				system("pause");
				return -1;
			}
			letters = visionize(imColor, import, templates_dir, temp.ch_count);				//save image templates in their designated folder
			temp.ch_count += letters.size();
		}
		
		fonts.push_back(temp);
		write_to_file("templates/Fonts.txt", fonts);
	}
	else{
		if(fonts.size() == 0)
			cout << "no fonts are imported for carrying our OCR!\n";
		else{
			cout << "Please select one of the existing fonts templates for performing OCR\n";
			for(int i=0; i<fonts.size(); i++)
				cout << i+1 << ". " << fonts[i].name << endl;
			int sel;
			cin >> sel;
			int font_chars = fonts[sel-1].ch_count;

			//Initialize templates
			templates_dir = "templates/" + fonts[sel-1].name;
			init_templates(templates, templates_dir, font_chars);
		
			//Read letters
			letters = visionize(imColor, import, "", 0);							//load letters and save their images in their folder
		
			//Match letters to templates
			matchLetters(letters, templates, result, font_chars);
			cout << "\nThe program reads:\n" << result << endl;
		}
	}
	system("pause");
    return 0;
}

vector<cv::Mat> visionize(cv::Mat imColor, bool import, string outloc, int offset){

	cv::Mat imGrayscale, imBinary;
	vector<std::vector<int>> accum;
    vector<cv::Mat> letters;
    vector<Blob> vecBlobs;

	//converting to grayscale and smoothing
	cv::cvtColor( imColor, imGrayscale, CV_BGR2GRAY );
	cv::Mat imTester = imGrayscale.clone();
    smoothTran(imGrayscale, imBinary);
    doBinary(imBinary);

	//--------------------------------------------------------------------------------------------------------------------
	//CCL algorithm
    CCL(imBinary, accum);
	createBlobs(accum, vecBlobs);
	//sort(vecBlobs.begin(), vecBlobs.end());
	mergeBlobs(vecBlobs);
	sortLines(vecBlobs);
    drawRect(imColor, vecBlobs);
	createLetters(accum, vecBlobs, letters);
	string dir;
	if(import)
		dir = outloc;
	else
		dir = "letters";
	for(int i = 0; i < letters.size(); i++){				//write the found letters to files for testing and debugging
		cv::imwrite(dir+"/_" + std::to_string(offset+i) + ".png", letters[i]);
		//printImg(templates[30], "_" + std::to_string(1)); 
    }
	cv::imwrite("out.png", imBinary);
    cv::imwrite("outletters.png", imColor);
	return letters;
}

void matchLetters(vector<cv::Mat>& letters, vector<cv::Mat>& template_letters, string& output, int letter_count){
	cout << "=>matching characers...";
	for(int i=0; i<letters.size(); i++){
		cv::Mat curLetter = letters[i];
		int* coeffs;
		coeffs = new int[letter_count];
		for(int j=0; j<template_letters.size(); j++){
			cv::Mat tempLetter = template_letters[j].clone();
			coeffs[j] = matchWindows(curLetter, tempLetter);
			//cout << coeffs[j] << " ";
			/*if(coeffs[j]>9999999){
				printImg(curLetter, "letter");
				printImg(tempLetter, "");
				cv::waitKey(0);
			}*/
		}
		int ind = minIndex(coeffs, letter_count);
		output += char(ind + 33);
	}
	cout << "Done!\n";
}

int matchWindows(cv::Mat& src, cv::Mat& temp){
	int coeff = 0;
	float scalex = (float)temp.cols / src.cols;
	float scaley = (float)temp.rows / src.rows;
	if(abs(scalex - scaley)>0.5)
		return 9999999999999999;

	cv::resize(temp, temp, cv::Size(src.cols, src.rows));
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			int src_val = src.at<uchar>(i, j);
			int temp_val = temp.at<uchar>(i, j);
			coeff += std::pow(src_val - temp_val, 2.0);
			//diff_acc += std::pow(src_val - temp_val, 2.0);
			//src_acc += std::pow(src_val, 2.0);
			//temp_acc += std::pow(temp_val, 2.0);
		}
	}
	return coeff;
}

int minIndex(int ar[], int n){
	int min = ar[0], mini = 0;
	for(int i=1; i<n; i++){
		if(ar[i] < min){
			min = ar[i];
			mini = i;
		}
	}
	return mini;
}

void init_templates(vector<cv::Mat>& template_letters, string dir, int letter_count){
	cout << "=>Loading letter templates...";
	for(int i=0; i<letter_count; i++){
		cv::Mat temp;
		temp = cv::imread(dir+"/_"+std::to_string(i)+".png", CV_LOAD_IMAGE_GRAYSCALE);
		template_letters.push_back(temp);
	}
	cout << "Done!\n";
}

void read_from_file(string fname, vector<Font>& ar){
	ifstream infile;
	infile.open(fname);

	if(!infile.fail()){
		while(!infile.eof()){
			Font temp;
			infile >> temp.name;
			infile >> temp.ch_count;
			if(temp.name != "")
				ar.push_back(temp);
		}
	}
}

void write_to_file(string fname, vector<Font>& ar){
	ofstream outfile;
	outfile.open(fname);

	if(!outfile.fail()){
		for(int i=0; i<ar.size(); i++){
			if(i!=0)
				outfile << endl;
			outfile << ar[i].name << '\t' << ar[i].ch_count;
		}
	}
}
