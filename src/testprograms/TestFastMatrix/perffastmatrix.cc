/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#include <Core/Math/FastMatrix.h>
#include <Core/Math/Rand48.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Timers/Timers.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

#define MAX_SIZE 16
int main(int argc, char* argv[])
{
  // defaults
  int size = 5;     // matrix size
  int reps = 100;   // number of repetitions
  
  //__________________________________
  // parse user inputs
  if(argc >= 2){
    size = atoi(argv[1]);
  }
  if(argc >= 3){
    reps = atoi(argv[2]);
  }  
   
  //__________________________________
  //  populate matrix
  Uintah::FastMatrix matrix(size, size);
  for(int i=0;i<size;i++){
    for(int j=0;j<size;j++){
      matrix(i,j) = drand48();
    }
  }
  
  Uintah::FastMatrix minv(size, size);
  
  //__________________________________
  // populate right hand sides
  std::vector<double> b(size);
  std::vector<double> b2(size);
  double b_D[MAX_SIZE];
  
  for(int i=0;i<size;i++){
    b[i]  = drand48();
    b2[i] = drand48();
    b_D[i] = drand48();
  }
  
  // solution vectors
  std::vector<double> x(size);
  std::vector<double> x2(size);

  //__________________________________
  // start timer
  Timers::Simple timer;
  timer.start();

#if 0
  for(int i=0;i<reps;i++){
    minv.destructiveInvert(matrix);
    minv.multiply(b, x);
    minv.multiply(b2, x2);
  }

  // 
  for(int i=0;i<reps;i++){
    minv.destructiveSolve(&b[0], &b[1]);
  }
  
  for(int i=0;i<reps;i++){
    matrix.destructiveSolve(b);    // vector version
  }  
#else
  for(int i=0;i<reps;i++){
    matrix.destructiveSolve(b_D);   // double version
  }
#endif

  std::cerr << reps << " in " << timer().seconds() << " seconds, "
	    << timer().seconds()/reps*1000000 << " us/rep\n";
  exit(0);
}

