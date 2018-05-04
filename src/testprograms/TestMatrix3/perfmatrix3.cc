/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Math/Matrix3.h>
#include <Core/Math/Rand48.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Timers/Timers.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
using namespace std;

/*______________________________________________________________________
 Jim says 90% of matrix 3 is in
  - Determinant
  - Trace
  - * operator
*/

/*______________________________________________________________________
       Usage: perfmatrix3 <case> <reps>
       Case 1 runs Determinant(), 
       case 2 runs Trace(), and 
       case 3 multiplies two matrices.
*/

int main(int argc, char* argv[]) {

  // defaults
  int reps = 100;   // number of repetitions
  int testCase = 0; // What to test
  
  //__________________________________
  // parse user inputs
  if(argc == 1){
    cout << "Usage: perfmatrix3 <case> <reps>"
  	  <<"\nCase 1 runs Determinant(), case 2 runs Trace(), and case 3 multiplies two matrices.\n";
    exit(0);
  }

  if(argc >= 2){
    testCase = atoi(argv[1]);
  }
  if(argc >= 3){
    reps = atoi(argv[2]);
  } 
  
  //__________________________________
  //  populate matrix
  Uintah::Matrix3 perfMatrix;
  for (int i = 0; i<3; i++){
    for (int j =0; j<3; j++){
      perfMatrix.set(i,j,drand48());
    }
  }

  Uintah::Matrix3 perfMatrix2;
  for (int i = 0; i<3; i++){
    for (int j =0; j<3; j++){
      perfMatrix2.set(i,j,drand48());
    }
  }

  //__________________________________
  // start timer
  Timers::Simple timer;
  timer.start();

  if (testCase == 1){
    cout << "Running Determinant()\n";
    for(int i=0;i<reps;i++){
      perfMatrix.Determinant();   // double version
    }
  }
  else if (testCase == 2){
    cout << "Running Trace()\n";
    for(int i=0;i<reps;i++){
      perfMatrix.Trace();   // double version
    }
  } else {
    cout << "Running a dot operator test\n";
    for (int i = 0; i<reps; i++){
      Uintah::Matrix3 multTest = perfMatrix * perfMatrix2;
    }
  }

  std::cerr << reps << " in " << timer().seconds() << " seconds, "
    << timer().seconds()/reps*1000000 << " us/rep\n";
  exit(0);

}
