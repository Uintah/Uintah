
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Common.h>
#include <stdlib.h>
#include <iostream>

using namespace std;
using namespace Uintah;
  
char**
Uintah::CharArray(int length, int n)
{
  int i; 
  char *temp, **matrix;
  matrix=new char*[n];
  temp=new char[length*n];
  for(i=0;i<n;i++) 
    { matrix[i]=&temp[i*length]; }
  if(matrix&&temp) {
    return matrix; /*success */
  }
  cout << "Error in CharArray\n";
  exit(0);
  return matrix;
}

void
Uintah::DeleteCharArray(char **matrix, int n)
{
  int i;
  for (i=0;i<n;i++)
    { delete [] matrix[i];}
}


double**
Uintah::AllocMatrix(int rows, int cols)
{
  int i;
  double *temp, **matrix;
  matrix=new double*[cols];
  temp=new double[rows*cols];
  for(i=0;i<cols;i++) 
    { matrix[i]=&temp[i*rows]; }
  if(matrix&&temp) {
    return matrix; /*success */
  }
  cout << "Error in AllocMatrix\n";
  exit(0);
  return matrix;
}

void
Uintah::DeallocMatrix(double **matrix, int cols)
{
  int i;
  for (i=0;i<cols;i++)
    { delete [] matrix[i];}
}

