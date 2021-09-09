#pragma once
#include <vector>

using namespace std;

class BBMMatrix
{
/*
Matrix structure - trasformation from rows and columns to vector elements:


column      1  2   3   4
		____________________
	row 1|  0  5   10  15
	row 2|  1  6   11  16
	row 3|  2  7   12  17
	row 4|  3  8   13  18
	row 5|  4  9   14  19

numbers inside show the number of element of the vector so vec[14]=matrix[5,3]
you usually don't need to bother about it...
*/


public:
	int maxrow, maxcol;
	BBMMatrix(int row, int column);
	~BBMMatrix(void);
	double GetElement (int row,int column);
	double GetElementZ (int row,int column);
	int PutElement (int row,int column, double value);
	int PutElementZ (int row,int column, double value);
	int PutColumn (int colno, double * Column);
	int PutRow (int rowno, double * Row);
	int PutMatrix (int row, int column, BBMMatrix MatrixA);
	int Resize (int newrow, int newcolumn);

	void Print ();
	void PrintPrecise ();
	void PrintColumn (int column);
	void PrintRow (int row);

	int Multiply (BBMMatrix* MatrixA, BBMMatrix* Result);
	int Multiply (double constant, BBMMatrix* Result);
    int Add (BBMMatrix* MatrixA, BBMMatrix* Result);
	int Substract (BBMMatrix* MatrixA, BBMMatrix* Result);
	int Transpose (BBMMatrix* Result);
	int Copy (BBMMatrix* Result);
	int Zero ();
	int Inverse3x3 (BBMMatrix* Result);
	int CopyPart (int RowBeg, int RowEnd, int ColBeg, int ColEnd, BBMMatrix*Result);
	int LUDecompose(BBMMatrix* Result, vector <int> *Index, int* d);
	int LUSolve (vector <int> *Index, double* RightHandSide);
	int Solve (double* RightHandSide);
	int FindLUInverse (BBMMatrix* Result,vector <int> *Index,int* d);
	int Inverse (BBMMatrix * Result);
	int Eigen (BBMMatrix *EigenVectors, BBMMatrix *EigenValues);
	int SwitchColumns (int Column1, int Column2);
	int VectorProduct (BBMMatrix *ToMultiply, BBMMatrix* Result);
	double RotateToEigen (double TVector[] ,BBMMatrix * EigenVectors);
	double RotateToOrigin (double TVector[] ,BBMMatrix * EigenVectors);
	double RotateToEigen (BBMMatrix* InOrigin ,BBMMatrix * EigenVectors);
	double RotateToOrigin (BBMMatrix* InEigen ,BBMMatrix * EigenVectors);

	double AbsMax();

private:

	double * matrix;


};
