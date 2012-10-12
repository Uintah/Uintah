/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef VirtualSurface_H
#define VirtualSurface_H

#include "Surface.h"
#include "MersenneTwister.h"
#include <cmath>

class MTRand;

class VirtualSurface : public Surface{
public:
  VirtualSurface();
  ~VirtualSurface();

  inline
  void get_PhFunc(const int &PhFunc_Flag,
		  const double &linear_b,
		  const double &eddington_f,
		  const double &eddington_g){
    PhFunc = PhFunc_Flag;
    b = linear_b;
    f = eddington_f;
    g = eddington_g;
  }
    
  virtual void getTheta(const double &random);
  
  //get e1-- e1
  void get_e1(const double &random1,
	      const double &random2,
	      const double &random3,
	      const double *sIn);

  //get e2 -- e2
  void get_e2(const double *sIn);

  // get scatter_s
  void get_s(MTRand &MTrng, const double *sIn, double *s);
private:
  double e1[3], e2[3];
  int PhFunc;
  double b, f, g;
};

#endif
