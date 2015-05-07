/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef LeftRealSurface_H
#define LeftRealSurface_H

#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RealSurface.h>

class LeftRealSurface:public RealSurface{
  
public:

  LeftRealSurface(const int &iIndex,
      const int &jIndex,
      const int &kIndex,
      const int &Ncy);
  
  LeftRealSurface();
  ~LeftRealSurface();

// n top -- n = 1 i + 0 j + 0 k
  inline
  void get_n(){
    n[0] = 1;
    n[1] = 0;
    n[2] = 0;
  }


// t1 top -- t1 = 0 i + 1 j + 0 k
  inline
  void get_t1(){
    t1[0] = 0;
    t1[1] = 1;
    t1[2] = 0;
  }



// t2 top -- t2 = 0 i + 0 j + 1 k
  inline
  void get_t2(){
    t2[0] = 0;
    t2[1] = 0;
    t2[2] = 1;
  }


  inline
  void set_n(double *nn){
    for ( int i = 0; i < 3; i ++ )
      nn[i] = n[i];
  }


  inline 
  void get_limits(const double *X,
           const double *Y,
           const double *Z){
    
    // i, j, k is settled at the center of the VOLUME cell
    xlow = X[surfaceiIndex];
    xup = X[surfaceiIndex];
    
    ylow = Y[surfacejIndex];
    yup = Y[surfacejIndex+1];
    
    // note that for top surface, zlow = ztop
    zlow = Z[surfacekIndex];
    zup = Z[surfacekIndex+1];
    
  }

  
 //  virtual void set_n(double *nn);
//   virtual void get_n();
//   virtual void get_t1();
//   virtual void get_t2();
//   virtual void get_limits(const double *X,
//        const double *Y,
//        const double *Z);
  
  

};

#endif
  
