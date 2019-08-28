#include <CCA/Components/MPM/Materials/ConstitutiveModel/QuocAnh/stdafx.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/QuocAnh/BBMMatrix.h>


using namespace std;
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
*/


BBMMatrix::BBMMatrix(int row, int column)
{
if ((row>0)&&(column>0))
{
	maxrow=row;
	maxcol=column;
	matrix=new double[row*column];
	for (int i=0; i<row*column; i++)
	{
		matrix[i]=0;

	}
}
else cout<<"ERROR - Given parameters while creating matrix must be positive, while row="<<row<<" and column="<<column<<endl;

}

BBMMatrix::~BBMMatrix(void)
{
	delete[] matrix;
}

double BBMMatrix::GetElement (int row,int column){

return matrix[(column-1)*maxrow+(row-1)];  //i want to start with giving 1,1 and returning 0 etc...

}

double BBMMatrix::GetElementZ (int row,int column) {
return matrix[column*maxrow+row];  //starting from 0,0
}

int BBMMatrix::PutElement (int row,int column, double value)
{
if ((row<1)||(column<1)) {cout<<"Matrix::PutElement Error- row or column<1"<<endl; return -1;}
if ((row>maxrow)||(column>maxcol))  {
	 cout<<"Matrix::PutElement Error- row or column>max"<<endl;
	return -1;}
else
	{
	matrix [(column-1)*maxrow+(row-1)]=value;
	return 0;
	}
}

int BBMMatrix::PutElementZ (int row,int column, double value)
{
if ((row<0)||(column<0)) {cout<<"Matrix::PutElement Error- row or column<1"<<endl; return -1;}
if ((row>maxrow-1)||(column>maxcol-1))  {cout<<"Matrix::PutElement Error- row or column>max"<<endl; return -1;}
else
	{
	matrix [column*maxrow+row]=value;
	return 0;
	}
}


int BBMMatrix::PutColumn (int colno, double * Column)
{
	if ((colno>maxcol)||(colno<1)) return -1;

	else
	{
		for (int i=0; i<maxrow; i++) matrix [(colno-1)*maxrow+i]=Column[i];
		return 0;
	}

}

int BBMMatrix::PutRow (int rowno, double * Row)
{
	if ((rowno>maxrow)||(rowno<1)) return -1;
	else
	{
		for (int i=0; i<maxcol; i++) matrix [i*maxcol+rowno]=Row[i];
		return 0;
	}
}

int BBMMatrix::PutMatrix (int row, int column, BBMMatrix MatrixA)
{
// row, column - entry point of the matrix
	if ((row<1)||(column<1)) return -1;
	if (((row+MatrixA.maxrow)>maxrow)||((column+MatrixA.maxcol)>maxcol)) return -1;
	else
	{
		for (int i=0; i<MatrixA.maxcol; i++)
		{
			for (int j=0; j<MatrixA.maxrow; j++)
				PutElement(row+j,column+i,MatrixA.GetElement(j+1,i+1));
		}
		return 0;
	}
}

int BBMMatrix::Resize (int newrow, int newcolumn)
{
	if ((newrow>0)&&(newcolumn>0))
	{
		maxrow=newrow;
		maxcol=newcolumn;
		delete[] matrix;
		matrix=new double[maxrow*maxcol];
		for (int i=0; i<maxrow*maxcol; i++) matrix[i]=0;
		return 0;
	}
	else return -1;
}

int BBMMatrix::Multiply (BBMMatrix* MatrixA, BBMMatrix* Result)
{
// the matrix used is multiplied by MatrixA. Results are saved in matrix Result
// Result is reinitiated at the beginning.

//1 check whether possible

if (maxcol!=MatrixA->maxrow)
{
	cout<<"Number of columns in first matrix is not equal to number of rows in second matrix"<<endl;
	cout<<"No multiplication done"<<endl;
	return -1;
}
// the procedure terminated if the multiplication is not possible, returns -1

//now we have to change the things in result martix...
BBMMatrix TT (maxrow,MatrixA->maxcol);
double value;

for (int i=1; i<=maxrow; i++)
{
	for (int j=1; j<=MatrixA->maxcol; j++)
	{
		value=0;
		for (int k=1; k<=maxcol; k++)
		{
			value=value+GetElement(i,k)*MatrixA->GetElement(k,j);
		}
		TT.PutElement(i,j,value);
	}
}
TT.Copy (Result);

return 0;
}


int BBMMatrix::Multiply (double constant, BBMMatrix* Result)
{
// the matrix used is multiplied constant. Results are saved in matrix Result
// Result is reinitiated at the beginning.

//now we have to change the things in result martix...

BBMMatrix TT (maxrow,maxcol);


for (int i=1; i<=maxrow; i++)
{
	for (int j=1; j<=maxcol; j++)
	{
		TT.PutElement(i,j,GetElement (i,j)*constant);
	}
}
TT.Copy (Result);

return 0;
}

int BBMMatrix::Add (BBMMatrix* MatrixA, BBMMatrix* Result)
{


//check whether adding possible
	if ((maxrow!=MatrixA->maxrow)||(maxcol!=MatrixA->maxcol))
	{
	cout<<"Dimensions of the matrices do not agree."<<endl;
	cout<<"No adding done"<<endl;
		return -1;
	}

//now we have to change the things in result martix...
BBMMatrix TT (maxrow,maxcol);

for (int i=1; i<=maxrow; i++)
{
	for (int j=1; j<=maxcol; j++)
	{
		TT.PutElement(i,j, GetElement(i,j)+ MatrixA->GetElement(i,j));
	}
}
TT.Copy (Result);
return 0;
}

int BBMMatrix::Substract (BBMMatrix* MatrixA, BBMMatrix* Result)
{
// the matrix used is multiplied constant. Results are saved in matrix Result
// Result is reinitiated at the beginning.

//check whether adding possible
	if ((maxrow!=MatrixA->maxrow)||(maxcol!=MatrixA->maxcol))
	{
	cout<<"Dimensions of the matrices do not agree."<<endl;
	cout<<"No substracting done"<<endl;
		return -1;
	}
//now we have to change the things in result martix...

BBMMatrix TT (maxrow,maxcol);

for (int i=1; i<=maxrow; i++)
{
	for (int j=1; j<=maxcol; j++)
	{
		TT.PutElement(i,j,GetElement (i,j)- MatrixA->GetElement(i,j));
	}
}
TT.Copy (Result);
return 0;
}

int BBMMatrix::Copy (BBMMatrix* Result)
{
	Result->Resize (maxrow,maxcol);
	for (int row=1; row<=maxrow; row++)
	{
		for (int column=1; column<=maxcol; column++)
		{
			Result->PutElement(row, column, GetElement (row,column));
		}
	}
	return 0;
}

int BBMMatrix::CopyPart (int RowBeg, int RowEnd, int ColBeg, int ColEnd, BBMMatrix*Result)
{
	if (RowBeg<1) {cout<<"Matrix::CopyPart Wrong Row range."<<endl; return -1;}
	if (RowEnd>maxrow)	{cout<<"Matrix::CopyPart Wrong Row range."<<endl; return -1;}
	if (ColBeg<1){cout<<"Matrix::CopyPart Wrong Col range."<<endl; return -1;}
	if (ColEnd>maxrow) {cout<<"Matrix::CopyPart Wrong Col range."<<endl; return -1;}
	if (ColBeg>ColEnd) {cout<<"Matrix::CopyPart Beginning and end rage mismatch."<<endl; return -1;}
	if (RowBeg>RowEnd) {cout<<"Matrix::CopyPart Beginning and end rage mismatch."<<endl; return -1;}
	if (this==Result) {cout<<"Cannot copy part of the matrix into the same matrix!"<<endl; return -1;}
	ColEnd++; RowEnd++;
	bool NeedToResize=false;
	if (Result->maxrow!=RowEnd-RowBeg) NeedToResize=true;
    if (Result->maxcol!=ColEnd-ColBeg) NeedToResize=true;
	if (NeedToResize) Result->Resize(RowEnd-RowBeg,ColEnd-ColBeg);

	for (int i=RowBeg; i<RowEnd; i++) for (int j=ColBeg; j<ColEnd; j++)
		Result->PutElement(i-RowBeg+1,j-ColBeg+1,GetElement(i,j));
	return 0;
}


int BBMMatrix::LUDecompose(BBMMatrix* Result, vector <int> *Index, int* d)
{
//This algorithm follows the Numerical Recipes in C++, Press et al., Cam. Univ. Press, 2002, pp 49-50. I believe it is changed enough to be not a copy that infringe copyrights

/* This routine writes a LU matrix into result matrix. The diagonal elements in the Result matrix come from the upper matrix
This routine is not thought to be used separately - you usually should consider using Solve or Inverse methods instead

Input: The matrix called, rest is uninportant and can have rubbish values
Output: In the Result - resulted matrix; Index vector contains row permutations effected by the partial pivotiong, d is +-1 */


if (maxrow!=maxcol) {cout<<"BBMMatrix::LUDecompose Matrix is not square!!!"<<endl; return -1;}

if (Result!=this) Copy (Result);	//if the same matrix, initial coping is not necessary
Index->clear();	//cleaning...


const double Tiny=1e-20;
int i, imax=0, j, k, n;
double Max, Dummy, Sum;
n=maxrow;
double * Scaling;
Scaling=new double [n]; //this table stores implicit scaling for each row
*d=1.0;

for (i=0; i<maxrow; i++) Index->push_back (i);

for (i=0; i<n; i++) {
	Max=0.0;
	for (j=0; j<n; j++)	if (fabs(Result->GetElementZ(i,j))>Max) Max=fabs(Result->GetElementZ(i,j));
	if (Max==0.0) {
		cout<<"BBMMatrix::LUDecompose Singular matrix...";}
	Scaling[i]=1.0/Max;	//The scaling is saved
	}

for (j=0; j<n; j++) {
	for (i=0; i<j; i++)	{
		Sum=Result->GetElementZ (i,j);
		for (k=0; k<i; k++) Sum-=Result->GetElementZ(i,k)*Result->GetElementZ(k,j);
		Result->PutElementZ(i,j,Sum);
	}
	Max=0.0;
	for (i=j; i<n; i++){
		Sum=Result->GetElementZ(i,j);
		for (k=0; k<j; k++) Sum-=Result->GetElementZ(i,k)*Result->GetElementZ(k,j);
		Result->PutElementZ(i,j,Sum);
		if ((Dummy=Scaling[i]*fabs(Sum))>=Max) {Max=Dummy; imax=i;}	//checking for best pivot
	}
	if (j!=imax) {	//and exchanging rows
		for (k=0; k<n; k++){
			Dummy=Result->GetElementZ(imax,k);
			Result->PutElementZ(imax,k,Result->GetElementZ(j,k));
			Result->PutElementZ(j,k,Dummy);
		}
		*d=-*d;	//parity of d
		Scaling[imax]=Scaling[j]; //and scale factor
	}
	Index->operator [](j)=imax;
	if (Result->GetElementZ(j,j)==0.0) Result->PutElementZ(j,j,Tiny);	//to avoid division by zero
	if (j!=n-1){
		Dummy=1.0/Result->GetElementZ(j,j);	//divide by the pivot element
		for (i=j+1;i<n;i++) Result->PutElementZ(i,j,Dummy*Result->GetElementZ(i,j));
	}
}

delete [] Scaling;
return 0;
}
int BBMMatrix::LUSolve (vector <int> *Index, double* RightHandSide)
{
	/* Solves the set of n linear equations AX=B. A- given, used, LUMatrix, Index - contains permutations, Right hadn size is B
	The procedure returns solution vector X in place of B. LUMatrix and Index are unaffected
	The procedure closely follows the procedure given in Numerical Recipes in C++, Press et al., Cam. Univ. Press, 2002, pp 50.
	However, I believe that is different enough to not have any copyright issues
	*/
	if (maxrow!=maxcol) {cout<<"BBMMAtrix::LUSolve. Only solve square matrices."<<endl; return -1;}

	int i, ii=0, ip, j,n=maxrow;
	double Sum;
	for (i=0; i<n; i++) {
		ip=Index->operator [](i);
		Sum=RightHandSide[ip];
		RightHandSide[ip]=RightHandSide[i];
		if (ii!=0.0)
			for (j=ii-1; j<i; j++) Sum-=GetElementZ(i,j)*RightHandSide[j];
		else if (Sum!=0) ii=i+1;
		RightHandSide[i]=Sum;
	}
	for (i=n-1; i>=0; i--) {
		Sum=RightHandSide[i];
		for (j=i+1; j<n; j++) Sum-=GetElementZ(i,j)*RightHandSide[j];
		RightHandSide[i]=Sum/GetElementZ(i,i);
	}
	return 0;
}

int BBMMatrix::Solve (double* RightHandSide)
{
	vector <int> Index;
	int d;
	LUDecompose (this, &Index, &d);
	LUSolve (&Index, RightHandSide);
	return 0;
}
int BBMMatrix::FindLUInverse (BBMMatrix* Result,vector <int> *Index,int* d)
{
	/* Inverses Matrix. A- given, used, LUMatrix, Index - contains permutations, d - sign
	The initial matrix is unaffected, unless it is the result matrix. Due to copying issues, it was decided to locally copy the
	matrix which is not good; an if clause would do the job, after initial checking whether the matrices are the same.
	The procedure closely follows the procedure given in Numerical Recipes in C++, Press et al., Cam. Univ. Press, 2002, pp 51.
	However, I believe that is different enough to not have any copyright issues
	*/


if (maxrow!=maxcol) {cout<<"BBMMAtrix::FindLUInverse. Only solve square matrices."<<endl; return -1;}
BBMMatrix Res(maxrow, maxcol);

int i,j;
double * Col;
Col=new double [maxrow];

for (j=0; j<maxrow; j++) {
	for (i=0; i<maxrow; i++) Col[i]=0.0;
	Col[j]=1.0;
	LUSolve (Index,Col);
	for (i=0; i<maxrow; i++) Res.PutElementZ(i,j,Col[i]);
}
Res.Copy(Result);

delete [] Col;
return 0;
}
int BBMMatrix::Inverse (BBMMatrix * Result)
{
	vector <int> Index;
	int d;
	LUDecompose (Result, &Index, &d);
	Result->FindLUInverse (Result, &Index, &d);
	return 0;
}


int BBMMatrix::Zero ()
{
	for (int row=1;row<=maxrow;row++)
	{
		for (int column=1;column<=maxcol;column++)
			PutElement (row,column,0);
	}
	return 0;
}

int BBMMatrix::Transpose (BBMMatrix* Result)
{

BBMMatrix TT(maxcol,maxrow);

for (int i=1; i<=maxrow; i++)
{
	for (int j=1; j<=maxcol; j++)
	{
		TT.PutElement(j,i,GetElement (i,j));
	}
}

TT.Copy (Result);

return 0;
}


void BBMMatrix::Print ()
{
	cout<<"Maxrow and MaxCol are:"<<maxrow<<"  "<<maxcol<<endl;
	for (int i=1; i<=maxrow; i++)
	{
		for (int j=1; j<=maxcol; j++)
		{
			cout<<"  "<<GetElement(i, j);
		}
		cout<<endl;

	}
}

void BBMMatrix::PrintPrecise ()
{
	int Temp;
	Temp=cout.precision ();
	cout<<"Maxrow and MaxCol are:"<<maxrow<<"  "<<maxcol<<endl;
	for (int i=1; i<=maxrow; i++)
	{
		for (int j=1; j<=maxcol; j++)
		{
			cout<< "  "  << setprecision (16)<< GetElement(i, j);
		}
		cout<<endl;
	}
	cout<<setprecision (Temp);
}



void BBMMatrix::PrintColumn (int column)
{
	for (int i=1; i<=maxrow; i++) cout<<"  "<<GetElement(i, column)<<" "<<endl;
}

void BBMMatrix::PrintRow (int row)
{
	for (int i=1; i<=maxcol; i++) cout<<"  "<<GetElement(row, i);
	cout<<endl;
}

int BBMMatrix::Inverse3x3 (BBMMatrix* Result)
{
//Procedure not fully tested on unsymmetric matrices
cout<<"Do not use Inverse3x3 unless bug removed!"<<endl;


if (maxrow!=3) {cout<<"This function only invert 3x3 matrices!"<<endl; return -1;}
if (maxcol!=3) {cout<<"This function only invert 3x3 matrices!"<<endl; return -1;}

double a11,a22,a33,a12,a13,a23,a21,a31,a32;

a11=GetElement(1,1);
a12=GetElement(1,2);
a13=GetElement(1,3);
a21=GetElement(2,1);
a22=GetElement(2,2);
a23=GetElement(2,3);
a31=GetElement(3,1);
a32=GetElement(3,2);
a33=GetElement(3,3);

//calculating determinant
double Det=a11*a22*a33+a12*a23*a31+a13*a21*a32-a31*a22*a13-a32*a23*a11-a33*a21*a12;
if (Det==0){cout<<"Singular matrix. Determinant equal to zero! Cannot invert!"<<endl; return -1;}

bool NeedToResize=false;
if (Result->maxrow!=3) NeedToResize=true;
if (Result->maxcol!=3) NeedToResize=true;
if (NeedToResize) Result->Resize(3,3);


//calculating inverted matrix; see e.g. http://mathworld.wolfram.com/MatrixInverse.html
Result->PutElement(1,1,a22*a33-a32*a23);
Result->PutElement(1,2,a13*a32-a33*a12);
Result->PutElement(1,3,a12*a23-a22*a13);

Result->PutElement(2,1,a33*a21-a23*a31);
Result->PutElement(2,2,a11*a33-a31*a13);
Result->PutElement(2,3,a13*a21-a23*a11);

Result->PutElement(3,1,a21*a32-a31*a22);
Result->PutElement(3,2,a12*a31-a32*a11);
Result->PutElement(2,3,a11*a22-a21*a12);

Det=1/Det;
Result->Multiply (Det, Result);
return 0;
}

int BBMMatrix::Eigen (BBMMatrix *EigenVectors, BBMMatrix *EigenValues)
{

/*


%Using the Jacobi Method as in Bathe,
%Finite Element Procedures, Chapter 11, pg 913-919, 2006

%Input: none (calculates eigenvalues and eigenvectors of the matrix)
%Output: Matrix 3x3 of vectors, vectors in columns, Matrix 3x3 of eigenvalues

*/

double Accuracy;
double pi4=3.1415926535897932384626433832795/4.0;
double VerySmall=1E-14;
BBMMatrix P(3,3);
BBMMatrix PT(3,3);
BBMMatrix P1(3,3);
BBMMatrix P2(3,3);
BBMMatrix P3(3,3);
BBMMatrix K1(3,3);
BBMMatrix Eye(3,3);
Eye.PutElement(1,1,1.0);
Eye.PutElement(2,2,1.0);
Eye.PutElement(3,3,1.0);
Eye.Copy(&P);
double Sweep=0, Delta, Theta;
double Temp;

//cout<<"BBMMatrix::Eigen: print input"<<endl;
//this->Print();

this->Copy(&K1);


Accuracy=K1.AbsMax()/double(1E13);
if (Accuracy<1E-100) return (0);
if (Accuracy==0) Delta=VerySmall;
else Delta=10*Accuracy;


//cout<<"Acc="<<Accuracy<<endl;

while (Delta>Accuracy)
{
Eye.Copy(&P1);
Eye.Copy(&P2);
Eye.Copy(&P3);

//first sweep
Delta=fabs(K1.GetElement(1,2));
   if (fabs(K1.GetElement(1,2))>0)
   {
    if (fabs(K1.GetElement(1,1)-K1.GetElement(2,2))>Accuracy )
        {
        Theta=0.5*atan (2*K1.GetElement(1,2)/(K1.GetElement(1,1)-K1.GetElement(2,2)));
        }

    else
        {
        Theta=pi4;
        }

	//cout<<"Theta="<<Theta<<" Cos Theta="<<cos(Theta)<<" Sin Theta="<<sin(Theta)<<endl;
	//P1.Print();
    P1.PutElement(1,1,cos(Theta));
    P1.PutElement(1,2,-sin(Theta));
    P1.PutElement(2,2,P1.GetElement(1,1));
    P1.PutElement(2,1,-P1.GetElement(1,2));
	//P1.Print();
    //%K1=P1'*K1*P1;
   }
//%second sweep
Delta=Delta+fabs(K1.GetElement(1,3));
if (fabs(K1.GetElement(1,3))>0)
	{
    if (fabs(K1.GetElement(1,1)-K1.GetElement(3,3))>Accuracy)
        Theta=0.5*atan (2*K1.GetElement(1,3)/(K1.GetElement(1,1)-K1.GetElement(3,3)));
    else
        Theta=pi4;

    P2.PutElement(1,1,cos(Theta));
    P2.PutElement(1,3,-sin(Theta));
    P2.PutElement(3,3,P2.GetElement(1,1));
    P2.PutElement(3,1,-P2.GetElement(1,3));
    //%K1=P2'*K1*P2;
	}
//%third sweep
Delta=Delta+fabs(K1.GetElement(2,3));
	if (fabs(K1.GetElement(2,3))>0)
	{
    if (fabs(K1.GetElement(2,2)-K1.GetElement(3,3))>Accuracy)
        Theta=0.5*atan (2*K1.GetElement(2,3)/(K1.GetElement(2,2)-K1.GetElement(3,3)));
    else
        Theta=pi4;


    P3.PutElement(2,2,cos(Theta));
    P3.PutElement(2,3,-sin(Theta));
    P3.PutElement(3,3,P3.GetElement(2,2));
    P3.PutElement(3,2,-P3.GetElement(2,3));
	}
    //%K1=P3'*K1*P3;
//%P1=P1
//%P2=P2
//%P3=P3

P.Multiply(&P1,&P);
P.Multiply(&P2,&P);
P.Multiply(&P3,&P);
//%UnityP1=P1'*P1
//%UnityP2=P2'*P2
//%UnityP3=P3'*P3
//Unity=P'*P;
P.Transpose(&PT);
PT.Multiply(this,&PT);
PT.Multiply(&P,&K1);
/*
cout<<"P1:"<<endl;
P1.Print();
cout<<"P2:"<<endl;
P2.Print();
cout<<"P3:"<<endl;
P3.Print();
cout<<"P:"<<endl;
P.Print();
cout<<"K1:"<<endl;
K1.Print();
*/
Sweep++;



}
//end %while

//sorting eigenvalues from largest to smallest
//cout<<"Eigen: unsorted eigenvalues and  vectors"<<endl;
//K1.Print();
//P.Print();

if(K1.GetElement(1,1)<K1.GetElement(2,2))
{
	/*
    if(K1.GetElement(1,1)<K1.GetElement(3,3))
    {
    K1.SwitchColumns(1,3);

	//column 1
	Temp=K1.GetElement(1,1);
	K1.PutElement(1,1,K1.GetElement(3,1));
	K1.PutElement(3,1,Temp);
	//column 3
	Temp=K1.GetElement(1,3);
	K1.PutElement(1,3,K1.GetElement(3,3));
	K1.PutElement(3,3,Temp);

    P.SwitchColumns(1,3);

	if(K1.GetElement(1,1)<K1.GetElement(2,2))
        {
        K1.SwitchColumns(1,2);
		//column 1
		Temp=K1.GetElement(1,1);
		K1.PutElement(1,1,K1.GetElement(2,1));
		K1.PutElement(2,1,Temp);
		//column 2
		Temp=K1.GetElement(1,2);
		K1.PutElement(1,2,K1.GetElement(2,2));
		K1.PutElement(2,2,Temp);

        P.SwitchColumns(1,2); //and we are done, as K(1,1)>K(3,3)
        }
    }
    else
    { */
    K1.SwitchColumns(1,2);
	//column 1
	Temp=K1.GetElement(1,1);
	K1.PutElement(1,1,K1.GetElement(2,1));
	K1.PutElement(2,1,Temp);
	//column 2
	Temp=K1.GetElement(1,2);
	K1.PutElement(1,2,K1.GetElement(2,2));
	K1.PutElement(2,2,Temp);

    P.SwitchColumns(1,2); //and we are done, as K(1,1)>K(3,3)
  //  }

}

if(K1.GetElement(1,1)<K1.GetElement(3,3))
//will not happen if the before if was called
//which means K(1,1)>K(2,2)
{

 /*   K1.SwitchColumns(1,2);
		//column 1
		Temp=K1.GetElement(1,1);
		K1.PutElement(1,1,K1.GetElement(2,1));
		K1.PutElement(2,1,Temp);
		//column 2
		Temp=K1.GetElement(1,2);
		K1.PutElement(1,2,K1.GetElement(2,2));
		K1.PutElement(2,2,Temp);


    P.SwitchColumns(1,2);*/
    K1.SwitchColumns(1,3);
	//column 1
	Temp=K1.GetElement(1,1);
	K1.PutElement(1,1,K1.GetElement(3,1));
	K1.PutElement(3,1,Temp);
	//column 3
	Temp=K1.GetElement(1,3);
	K1.PutElement(1,3,K1.GetElement(3,3));
	K1.PutElement(3,3,Temp);

    P.SwitchColumns(1,3); //and we are done
}

if(K1.GetElement(2,2)<K1.GetElement(3,3))
    {
    K1.SwitchColumns(2,3);

	//column 2
	Temp=K1.GetElement(2,2);
	K1.PutElement(2,2,K1.GetElement(3,2));
	K1.PutElement(3,2,Temp);
	//column 3
	Temp=K1.GetElement(2,3);
	K1.PutElement(2,3,K1.GetElement(3,3));
	K1.PutElement(3,3,Temp);

    P.SwitchColumns(2,3);
	}

//actually with the switch, I should put zeroes, as there should be zeroes, not the Temp value
//Things are commented as most likely the savings are negligible and complexity of algorithm is just not necessary

P.Copy(EigenVectors);
K1.Copy(EigenValues);
//K1.Print();
//P.Print();
//getchar();

return (0);

}

int BBMMatrix::SwitchColumns (int Column1, int Column2)
{
double Temp;

for (int i=1; i<=maxrow; i++)

    {
        Temp=GetElement(i,Column1);
        PutElement(i,Column1,GetElement(i,Column2));
        PutElement(i,Column2,Temp);
    }
return (0);

}

double BBMMatrix::AbsMax ()
{
	double Temp=0;
	for (int i=1;i<=maxrow;i++)
		for (int j=1; j<=maxcol;j++)
			if (Temp<fabs(GetElement(i,j))) Temp=fabs(GetElement(i,j));
	//if (Temp<1E-200) Temp=0;
	return (Temp);

}

double BBMMatrix::RotateToOrigin (double InEigen[] ,BBMMatrix * EigenVectors)
{
    BBMMatrix T(3,3),EigenVectT(3,3),R(3,3);

    T.PutElement(1,1,InEigen[0]);
    T.PutElement(2,2,InEigen[1]);
    T.PutElement(3,3,InEigen[2]);
    T.PutElement(1,2,InEigen[3]);
    T.PutElement(2,1,InEigen[3]);
    T.PutElement(1,3,InEigen[4]);
    T.PutElement(3,1,InEigen[4]);
    T.PutElement(2,3,InEigen[5]);
    T.PutElement(3,2,InEigen[5]);

    EigenVectors->Transpose(&EigenVectT);

    //Transition to origin:EigenVectors*Eigen*EigenVectorsT
    EigenVectors->Multiply(&T,&R);
    R.Multiply(&EigenVectT,&R);

    InEigen[0]=R.GetElement(1,1);
    InEigen[1]=R.GetElement(2,2);
    InEigen[2]=R.GetElement(3,3);
    InEigen[3]=R.GetElement(1,2);
    InEigen[4]=R.GetElement(1,3);
    InEigen[5]=R.GetElement(2,3);

    //InEigen changed to original coordinate system

    return 0;
}




double BBMMatrix::RotateToEigen (double InOrigin[] ,BBMMatrix * EigenVectors)
{
    BBMMatrix T(3,3),EigenVectT(3,3),R(3,3);

    T.PutElement(1,1,InOrigin[0]);
    T.PutElement(2,2,InOrigin[1]);
    T.PutElement(3,3,InOrigin[2]);
    T.PutElement(1,2,InOrigin[3]);
    T.PutElement(2,1,InOrigin[3]);
    T.PutElement(1,3,InOrigin[4]);
    T.PutElement(3,1,InOrigin[4]);
    T.PutElement(2,3,InOrigin[5]);
    T.PutElement(3,2,InOrigin[5]);

    EigenVectors->Transpose(&EigenVectT);

//Transition to eigen:EigenVectorsT*Origin*EigenVectors
    EigenVectT.Multiply(&T,&R);
    R.Multiply(EigenVectors,&R);

    InOrigin[0]=R.GetElement(1,1);
    InOrigin[1]=R.GetElement(2,2);
    InOrigin[2]=R.GetElement(3,3);
    InOrigin[3]=R.GetElement(1,2);
    InOrigin[4]=R.GetElement(1,3);
    InOrigin[5]=R.GetElement(2,3);

    //InOrigin changed to Eigenvalues coordinate system

    return 0;
}

double BBMMatrix::RotateToOrigin (BBMMatrix* InEigen ,BBMMatrix * EigenVectors)
{
    BBMMatrix T(3,3),EigenVectT(3,3),R(3,3);

	if (InEigen->maxrow==6 && InEigen->maxcol==1)
	{
    T.PutElement(1,1,InEigen->GetElement(1,1));
    T.PutElement(2,2,InEigen->GetElement(2,1));
    T.PutElement(3,3,InEigen->GetElement(3,1));
    T.PutElement(1,2,InEigen->GetElement(4,1));
    T.PutElement(2,1,InEigen->GetElement(4,1));
    T.PutElement(1,3,InEigen->GetElement(5,1));
    T.PutElement(3,1,InEigen->GetElement(5,1));
    T.PutElement(2,3,InEigen->GetElement(6,1));
    T.PutElement(3,2,InEigen->GetElement(6,1));

	EigenVectors->Transpose(&EigenVectT);
    //Transition to origin:EigenVectors*Eigen*EigenVectorsT
    EigenVectors->Multiply(&T,&R);
    R.Multiply(&EigenVectT,&R);

	InEigen->PutElement(1,1,R.GetElement(1,1));
    InEigen->PutElement(2,1,R.GetElement(2,2));
    InEigen->PutElement(3,1,R.GetElement(3,3));
    InEigen->PutElement(4,1,R.GetElement(1,2));
    InEigen->PutElement(5,1,R.GetElement(1,3));
    InEigen->PutElement(6,1,R.GetElement(2,3));

	}

	else if (InEigen->maxrow==3 && InEigen->maxcol==3)
		{
		InEigen->Copy(&T);
		EigenVectors->Transpose(&EigenVectT);
		//Transition to origin:EigenVectors*Eigen*EigenVectorsT
		EigenVectors->Multiply(&T,&R);
		R.Multiply(&EigenVectT,&R);
		R.Copy(InEigen);
		//InEigen changed to original coordinate system
		}
		else
		{
			cout<<"Rotate to Origin: the matrix given is not 6x1 or 3x3; Rotation NOT accomplished!!!"<<endl;
		}
    return 0;
}


double BBMMatrix::RotateToEigen (BBMMatrix* InOrigin ,BBMMatrix * EigenVectors)
{
    BBMMatrix T(3,3),EigenVectT(3,3),R(3,3);
	//Checked whether the InOrigin Matrix is 6x1 or 3x3
	if (InOrigin->maxrow==6 && InOrigin->maxcol==1)
	{
    T.PutElement(1,1,InOrigin->GetElement(1,1));
    T.PutElement(2,2,InOrigin->GetElement(2,1));
    T.PutElement(3,3,InOrigin->GetElement(3,1));
    T.PutElement(1,2,InOrigin->GetElement(4,1));
    T.PutElement(2,1,InOrigin->GetElement(4,1));
    T.PutElement(1,3,InOrigin->GetElement(5,1));
    T.PutElement(3,1,InOrigin->GetElement(5,1));
    T.PutElement(2,3,InOrigin->GetElement(6,1));
    T.PutElement(3,2,InOrigin->GetElement(6,1));

	EigenVectors->Transpose(&EigenVectT);
	//Transition to eigen:EigenVectorsT*Origin*EigenVectors
    EigenVectT.Multiply(&T,&R);
    R.Multiply(EigenVectors,&R);


	InOrigin->PutElement(1,1,R.GetElement(1,1));
    InOrigin->PutElement(2,1,R.GetElement(2,2));
    InOrigin->PutElement(3,1,R.GetElement(3,3));
    InOrigin->PutElement(4,1,R.GetElement(1,2));
    InOrigin->PutElement(5,1,R.GetElement(1,3));
    InOrigin->PutElement(6,1,R.GetElement(2,3));


	}
	else
		if (InOrigin->maxrow==3 && InOrigin->maxcol==3)
		{
	    InOrigin->Copy(&T);

		EigenVectors->Transpose(&EigenVectT);
		//Transition to eigen:EigenVectorsT*Origin*EigenVectors
	    EigenVectT.Multiply(&T,&R);
	    R.Multiply(EigenVectors,&R);
		R.Copy(InOrigin);
		}
		else
		{
		cout<<"Rotate to Eigen: the matrix given is not 6x1 or 3x3; Rotation NOT accomplished!!!"<<endl;
		}



    //InOrigin changed to Eigenvalues coordinate system

    return 0;
}

int BBMMatrix::VectorProduct (BBMMatrix *ToMultiply, BBMMatrix* Result)
{
	if ((maxrow!=3)||(maxcol!=1)||(ToMultiply->maxrow!=3)||(ToMultiply->maxcol!=1)) {
		cout<<"Procedure Matrix->VectorProduct Can only calculate cross product of 3x1 vectors. Matrices given are:"<<endl;
		Print();
		Result->Print();
		return -1;
	}
	double a1,a2,a3,b1,b2,b3;
	a1=GetElement(1,1);
	a2=GetElement(2,1);
	a3=GetElement(3,1);
	b1=ToMultiply->GetElement(1,1);
	b2=ToMultiply->GetElement(2,1);
	b3=ToMultiply->GetElement(3,1);

	if ((Result->maxcol!=1)||(Result->maxrow!=3)) {cout<<"Procedure Matrix->VectorProduct - Result matrix resized"<<endl; Result->Resize(3,1);}
	Result->PutElement(1,1,a2*b3-b2*a3);
	Result->PutElement(2,1,-a1*b3+b1*a3);
	Result->PutElement(3,1,a1*b2-b1*a2);
	return 0;
}
