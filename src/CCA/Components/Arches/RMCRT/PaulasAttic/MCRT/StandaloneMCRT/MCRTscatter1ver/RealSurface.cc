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

#include "RealSurface.h"
#include "Consts.h"

#include <iostream>
#include <cstdlib>

using std::cout;
using std::endl;

RealSurface::RealSurface(){
}


void RealSurface::get_s(MTRand &MTrng, double *s){
   
  // double random1, random2;
  // use this pointer, so to get to 6 different surfaces
  this->get_n();
  this->get_t1();
  this->get_t2();
  
  this->getTheta(MTrng.randExc()); 

  // getPhi is inherited from Class Surface
  this->getPhi(MTrng.randExc());

//   cout << "random1 = " << random1 << "; random2 = " << random2 << endl;
//   cout << " theta = " << theta << "; phi = " << phi << endl;
  
  for ( int i = 0; i < 3; i ++ ) 
    s[i] = sin(theta) * ( cos(phi) * this->t1[i] + sin(phi) * this->t2[i] )
      + cos(theta) * this->n[i] ;
    
}


// to get q on surface elements ( make it efficient to calculate intensity
// on surface element 
double RealSurface::SurfaceEmissFlux(const int &i,
				     const double *emiss_surface,
				     const double *T_surface,
				     const double *a_surface){
	    
 
  double emiss, Ts, SurEmissFlux;

  emiss = emiss_surface[i];
  Ts = T_surface[i] * T_surface[i];
  
  SurEmissFlux = emiss * SB * Ts * Ts * a_surface[i];
  
  return SurEmissFlux;
  
}



double RealSurface::SurfaceEmissFluxBlack(const int &i,
					  const double *T_surface,
					  const double *a_surface){
  
  
  double Ts, SurEmissFlux;


  Ts = T_surface[i] * T_surface[i];
  
  // should use the black emissive intensity?
  // guess not the black one for surface? right?
  // cuz the attenuation caused by the medium didnot include
  // the surface absorption
  
  SurEmissFlux =  SB * Ts * Ts * a_surface[i]; 
  
  return SurEmissFlux;
  
}




// whether put the SurfaceEmissFlux inside the intensity function is a question
// 1. if just calculate q once, then to get different intensity , separate
// 2. acutally both q and I are just calculated once. put inside.

// to get Intensity on surface elements
double RealSurface::SurfaceIntensity(const int &i,
				     const double *emiss_surface,
				     const double *T_surface,
				     const double *a_surface){
  
  double Ts, SurInten;

  Ts = T_surface[i] * T_surface[i];
  
  SurInten = emiss_surface[i] * SB * Ts * Ts * a_surface[i] / pi;
  
  return SurInten;
  
}


double RealSurface::SurfaceIntensityBlack(const int &i,
					  const double *T_surface,
					  const double *a_surface){

  double Ts, SurInten;

  Ts = T_surface[i] * T_surface[i];
  
  SurInten = SB * Ts * Ts * a_surface[i] / pi;
  
  return SurInten;
  
}



RealSurface::~RealSurface(){
}


