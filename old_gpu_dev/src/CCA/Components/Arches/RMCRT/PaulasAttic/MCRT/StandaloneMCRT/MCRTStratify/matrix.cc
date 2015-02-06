#include <iomanip> 
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;
int main()
{

int **mat1;
int ***mat2;
int **result;
int row,col;
 int layer1, layer2, layer3;
 
cout<<"Please enter row/col"<<endl;
cin>>row>>col;
 
 cout <<"Please enter layer1, layer2, layer3 " << endl;
 cin>> layer1 >> layer2 >> layer3;
 
mat1 = new int *[row];
mat2 = new int **[layer1];
result = new int *[row];
int k,i,j;

 for (k=0; k<row; k++){
    mat1[k] = new int[col];
  }

 for ( k = 0; k < layer1; k++)
   mat2[k] = new int *[layer2];

 for ( k = 0; k < layer1; k++)
   for ( i = 0; i < layer2; i ++)
   mat2[k][i] = new int [layer3];

 for ( k = 0; k < layer1; k++)
   for ( j = 0; j < layer2; j ++)
     for ( i = 0; i < layer3; i++){
       mat2[k][j][i] = k * layer2 * layer3 + j * layer3 + i;
       //cout << mat2[k][j][i] << endl;
     }
 
 for ( k = 0; k < layer1; k++){
   cout << endl;
   cout << "layer1 k = " << k << endl;
   cout << "----------------------------------------" << endl;
   for ( j = 0; j < layer2; j ++)
     for ( i = 0; i < layer3; i++){
       cout << mat2[k][j][i] << " " ;
       if ( i == (layer3-1) ) cout << endl;
     }
 }
 
   	 
for ( int i = 0; i < row; i ++){
  for ( int j = 0; j < col; j ++){
    mat1[i][j] = j * row + i;
  }
 }
 
 for ( j = 0; j < row; j ++)
   for ( i = 0; i < col; i++){
     cout << endl;
     
     cout << mat1[j][i] << " " ;
     if ( i == (col-1) ) cout << endl;
   }
 
 
return 0;
}

