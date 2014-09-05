/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <Core/Math/FastMatrix.h>
#include <Core/Math/Rand48.h>
#include <Core/Math/MiscMath.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
using namespace std;
using namespace SCIRun;
using namespace Uintah;

static const double tolerance = 1.e-10;

static void checkIdentity(const FastMatrix& m)
{
  int size=m.numRows();
  bool err=false;
  for(int i=0;i<size;i++){
    for(int j=0;j<size;j++){
      double want=0;
      if(i==j)
	want=1;
      if(Abs(m(i,j)-want) > tolerance){
	cerr << "Error: product(" << i << ", " << j << ")=" << m(i,j) << '\n';
	err=true;
      }
    }
  }
  if(err)
    exit(1);
}

int main(int argc, char* argv[])
{
  int max=16;
  if(argc == 2)
    max=atoi(argv[1]);
  for(int size=1;size<=max;size++){
    cout << "\n\n___________________________" << endl;
    cout << "Testing inverse and destructive solves functions for " << size << " X " << size;
    
    // form a matrix with random numbers
    FastMatrix m(size, size);
    for(int i=0;i<size;i++){
      for(int j=0;j<size;j++){
	m(i,j)=drand48();
      }
    }
    
    // compute the condition number
    cout << "  Condition Number:" << m.conditionNumber() << endl;
    
    // print out matrix
    cout.setf(ios::scientific,ios::floatfield);
    cout.precision(16); 
    //m.print(cout);

    //__________________________________
    // compute the inverse
    FastMatrix m2(size, size);
    m2.copy(m);
    FastMatrix minv(size, size);
    minv.destructiveInvert(m2);

    //__________________________________
    // Test 1: inverse matrix * matrix
    //  check if result is identity matrix
    cout << "Test 1: inverse matrix * matrix" << endl;
    FastMatrix product(size, size);
    product.multiply(minv, m);
    checkIdentity(product);  
    //cout << "inverse matrix * matrix: should be the identity matrix" << endl;  
    //product.print(cout);
    
    //__________________________________
    // Test 2: matrix * inverse matrix
    //  check if result is identity matrix
    cout << "Test 2: matrix * inverse matrix" << endl;
    product.multiply(m, minv);
    checkIdentity(product);

    //__________________________________
    // Test 3: 
    // A_inverse * b = xx
    // destructiveSolve(A) = x
    // if (x - xx) > tolerance get mad
    cout << "Test 3" << endl;
    vector<double> v(size);
    vector<double> vcopy(size);
    for(int i=0;i<size;i++){
      v[i]=vcopy[i]=drand48();
    }
    FastMatrix m3(size, size);
    m3.copy(m);
    m3.destructiveSolve(&vcopy[0]);
    vector<double> xx(size);
    minv.multiply(v, xx);
    bool err=false;
    for(int i=0;i<size;i++){
      if(Abs(vcopy[i]-xx[i] > tolerance)){
	if(!err)
	  cerr << "size: " << size << '\n';
	cerr << "Error: rhs[" << i << "]=" << vcopy[i] << " vs. " << xx[i] << '\n';
	err=true;
      }
    }
    if(err){
      exit(1);
    }
    //__________________________________
    //  Hilibert test for poorly conditioned matrices
    FastMatrix A(size, size), A_inverse(size,size);
    vector<double> XX(size), B(size);
    bool runTest = false;
    switch(size){
      case 3:
      {
        runTest = true;
        A(0,0) =1.000000000000000e+00, A(0,1) =  5.000000000000000e-01, A(0,2) = 3.333333333333333e-01; 
        A(1,0) =5.000000000000000e-01, A(1,1) =  3.333333333333333e-01, A(1,2) = 2.500000000000000e-01; 
        A(2,0) =3.333333333333333e-01, A(2,1) =  2.500000000000000e-01, A(2,2) = 2.000000000000000e-01;
 
        B[0] =1.833333333333333e+00;     
        B[1] =1.083333333333333e+00;          
        B[2] =7.833333333333332e-01;         
      }
      break;
      case 4:
      {
        runTest = true;
        A(0,0) =1.000000000000000e+00, A(0,1) =  5.000000000000000e-01, A(0,2) = 3.333333333333333e-01, A(0,3) =  2.500000000000000e-01; 
        A(1,0) =5.000000000000000e-01, A(1,1) =  3.333333333333333e-01, A(1,2) = 2.500000000000000e-01, A(1,3) =  2.000000000000000e-01; 
        A(2,0) =3.333333333333333e-01, A(2,1) =  2.500000000000000e-01, A(2,2) = 2.000000000000000e-01, A(2,3) =  1.666666666666667e-01; 
        A(3,0) =2.500000000000000e-01, A(3,1) =  2.000000000000000e-01, A(3,2) = 1.666666666666667e-01, A(3,3) =  1.428571428571428e-01;
 
        B[0] =2.083333333333333e+00;     
        B[1] =1.283333333333333e+00;     
        B[2] =9.499999999999998e-01;     
        B[3] =7.595238095238095e-01;     
      }
      break;
      case 5:
      {
        runTest = true;
        A(0,0) =1.000000000000000e+00, A(0,1) =  5.000000000000000e-01, A(0,2) = 3.333333333333333e-01, A(0,3) =  2.500000000000000e-01, A(0,4) = 2.000000000000000e-01;
        A(1,0) =5.000000000000000e-01, A(1,1) =  3.333333333333333e-01, A(1,2) = 2.500000000000000e-01, A(1,3) =  2.000000000000000e-01, A(1,4) = 1.666666666666667e-01;
        A(2,0) =3.333333333333333e-01, A(2,1) =  2.500000000000000e-01, A(2,2) = 2.000000000000000e-01, A(2,3) =  1.666666666666667e-01, A(2,4) = 1.428571428571428e-01;
        A(3,0) =2.500000000000000e-01, A(3,1) =  2.000000000000000e-01, A(3,2) = 1.666666666666667e-01, A(3,3) =  1.428571428571428e-01, A(3,4) = 1.250000000000000e-01;
        A(4,0) =2.000000000000000e-01, A(4,1) =  1.666666666666667e-01, A(4,2) = 1.428571428571428e-01, A(4,3) =  1.250000000000000e-01, A(4,4) = 1.111111111111111e-01;

        B[0] =2.283333333333333e+00;
        B[1] =1.450000000000000e+00;
        B[2] =1.092857142857143e+00;
        B[3] =8.845238095238095e-01;
        B[4] =7.456349206349207e-01;
      }
      break;
      case 6:
      {
         runTest = true;
         A(0,0) =   1.000000000000000e+00, A(0,1) =   5.000000000000000e-01, A(0,2) =    3.333333333333333e-01, A(0,3) =    2.500000000000000e-01, A(0,4) =    2.000000000000000e-01, A(0,5) =      1.666666666666667e-01;
         A(1,0) =   5.000000000000000e-01, A(1,1) =   3.333333333333333e-01, A(1,2) =    2.500000000000000e-01, A(1,3) =    2.000000000000000e-01, A(1,4) =    1.666666666666667e-01, A(1,5) =      1.428571428571428e-01;
         A(2,0) =   3.333333333333333e-01, A(2,1) =   2.500000000000000e-01, A(2,2) =    2.000000000000000e-01, A(2,3) =    1.666666666666667e-01, A(2,4) =    1.428571428571428e-01, A(2,5) =      1.250000000000000e-01;
         A(3,0) =   2.500000000000000e-01, A(3,1) =   2.000000000000000e-01, A(3,2) =    1.666666666666667e-01, A(3,3) =    1.428571428571428e-01, A(3,4) =    1.250000000000000e-01, A(3,5) =      1.111111111111111e-01;
         A(4,0) =   2.000000000000000e-01, A(4,1) =   1.666666666666667e-01, A(4,2) =    1.428571428571428e-01, A(4,3) =    1.250000000000000e-01, A(4,4) =    1.111111111111111e-01, A(4,5) =      1.000000000000000e-01;
         A(5,0) =   1.666666666666667e-01, A(5,1) =   1.428571428571428e-01, A(5,2) =    1.250000000000000e-01, A(5,3) =    1.111111111111111e-01, A(5,4) =    1.000000000000000e-01, A(5,5) =      9.090909090909091e-02;

         B[0] =2.450000000000000e+00;
         B[1] =1.592857142857143e+00;
         B[2] =1.217857142857143e+00;
         B[3] =9.956349206349207e-01;
         B[4] =8.456349206349206e-01;
         B[5] =7.365440115440116e-01;
      }
      break;
      default:
      break;
    }
    if(runTest){
      cout << "\nHilbert matrix test " << endl;
      cout << "Condition Number:" << A.conditionNumber() << endl;
      FastMatrix A2(size, size);
      A2.copy(A);
      
      A_inverse.destructiveInvert(A);
      A_inverse.multiply(B,XX);
      cout << " A inverse " << endl;
      A_inverse.print(cout);

      
      cout << "X should be 1.0" << endl;
      for(int i = 0; i<size; i++){
        cout << " X["<<i<<"]= " << XX[i] << "  % error " << fabs(XX[i] - 1.0) * 100<< endl;
      }


      double XX2[16];
      for(int i=0;i<size;i++)
        XX2[i] = B[i];
      A2.destructiveSolve(XX2);
      
      cout << "X2 should be 1.0" << endl;
      for(int i = 0; i<size; i++){
        cout << " X2["<<i<<"]= " << XX2[i] << "  % error " << fabs(XX2[i] - 1.0) * 100<< endl;
      }
    }
  }
  exit(0);
}
