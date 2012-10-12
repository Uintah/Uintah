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

#include <CCA/Components/Arches/MCRT/ArchesRMCRT/VirtualSurface.h>
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/Consts.h>

#include <iostream>
#include <cstdlib>

using namespace std;

VirtualSurface::VirtualSurface(){
}

VirtualSurface::~VirtualSurface(){
}



  
void VirtualSurface::set_theta(const double &random){
  if (PhFunc==ISOTROPIC){
    theta = acos( 1 - 2 * random);
  }
  else if (PhFunc==LINEAR_SCATTER) {// phasefunc = 1 + b* cos(theta)      
    // R_theta = (1 - cos(theta) + b/2 * sin(theta) * sin(theta))/2
    // cout << "linear scattering" << endl;
    theta = acos( ( -1 + sqrt(1 - 4 *b*(random - 0.5- 0.25*b)) )/b );
  }
  else if (PhFunc == EDDINGTON) { // delta-Eddington function
    // mu = cos(theta);
    // PhFunc = 2 * f * delta(1 - mu) + (1 - f)(1+3g*mu)
    // drop the forward direction term ( the 1st term)
    // calculate CDF from the 2nd term
    // reference 1998, Advanced heat transfer, Farmer
    
    theta = acos( ( -1+ sqrt( 1-6*g*(2*random-1-1.5*g) )) /3/g );
  }
  
}


// get_e1, sIn = sIn ( incoming direction vector)
void VirtualSurface::get_e1(const double &random1,
			    const double &random2,
			    const double &random3,
			    const double *sIn){

  // av is an arbitrary vector
  // e1 is then a unit vector
  // e1 =( av * sIn) / | av * sIn |

//   double av[3];
//   av[0] = random1;
//   av[1] = random2;
//   av[2] = random3;
  
  double e1i, e1j, e1k;
  e1i = random2 * sIn[2] - random3 * sIn[1];
  e1j = random3 * sIn[0] - random1 * sIn[2];
  e1k = random1 * sIn[1] - random2 * sIn[0];
  
//   e1i = av[1] * sIn[2] - av[2] * sIn[1];
//   e1j = av[2] * sIn[0] - av[0] * sIn[2];
//   e1k = av[0] * sIn[1] - av[1] * sIn[0];

  double as; // the |av*sIn|
  as = sqrt(e1i * e1i + e1j * e1j + e1k * e1k);

  e1[0] = e1i / as;
  e1[1] = e1j / as;
  e1[2] = e1k / as;
}

void VirtualSurface::get_e2(const double *sIn){
  
  // e2 = sIn * e1
  e2[0] = sIn[1] * e1[2] - sIn[2] * e1[1]; //i
  e2[1] = sIn[2] * e1[0] - sIn[0] * e1[2]; //j
  e2[2] = sIn[0] * e1[1] - sIn[1] * e1[0]; //k
}


void VirtualSurface::get_s(MTRand &MTrng, const double *sIn, double *s){

  get_e1(MTrng.randExc(),MTrng.randExc(), MTrng.randExc(), sIn);
  get_e2(sIn);

  this->set_theta(MTrng.randExc());   
  set_phi(MTrng.randExc());
    
  for ( int i = 0; i < 3; i ++ ) 
    s[i] = sin(theta) * ( cos(phi) * e1[i] + sin(phi) * e2[i] )
      + cos(theta) * sIn[i] ;
}  
  
  
