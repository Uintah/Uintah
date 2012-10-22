/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <CCA/Components/Arches/MCRT/ArchesRMCRT/VolElement.h>
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/Consts.h>

VolElement::VolElement(){ 
}


VolElement::VolElement(const int &iIndex,
		       const int &jIndex,
		       const int &kIndex,
		       const int &Ncx_,
		       const int &Ncy_){
  
  VoliIndex = iIndex;
  VoljIndex = jIndex;
  VolkIndex = kIndex;

  Ncx = Ncx_;
  Ncy = Ncy_;
  
  VolIndex = VoliIndex + VoljIndex * Ncx + VolkIndex * Ncx * Ncy;
  
}



VolElement::~VolElement(){
}



  
double VolElement::VolumeEmissFluxBlack(const int &vIndex,
					const double *T_Vol,
					const double *a_Vol){
    
  
  // we need either the emission position or hit position
  
  double Ts;
  double VolEmissFlux;
  
  Ts = T_Vol[vIndex] * T_Vol[vIndex];
  
  VolEmissFlux = 4 * SB * Ts * Ts * a_Vol[vIndex];
  
  return VolEmissFlux;
  
}


  
double VolElement::VolumeEmissFlux(const int &vIndex,
				   const double *kl_Vol,
				   const double *T_Vol,
				   const double *a_Vol){
  
  
  // we need either the emission position or hit position
    
  double kl, T, Ts;
  double VolEmissFlux;
  
  kl = kl_Vol[vIndex];
    
  T = T_Vol[vIndex];
  
  Ts = T * T;
  
  VolEmissFlux = 4 * kl * SB * Ts * Ts * a_Vol[vIndex];
  
  return VolEmissFlux;
  
}


 
double VolElement::VolumeIntensityBlack(const int &vIndex,
					const double *T_Vol,
					const double *a_Vol){
  
  double Ts, VolInten;
  
  Ts = T_Vol[vIndex] * T_Vol[vIndex];
  
  VolInten = SB * Ts * Ts * a_Vol[vIndex] / pi;  
    
  return VolInten;
  
}
  

  
double VolElement::VolumeIntensityBlack(const double &TVol,
					const double &aVol){
  
  double Ts, VolInten;
  
  Ts = TVol * TVol;
  VolInten = SB * Ts * Ts * aVol / pi;
  
  return VolInten;
  
}

  
  
double VolElement::VolumeIntensity(const int &vIndex,
				   const double *kl_Vol,
				   const double *T_Vol,
				   const double *a_Vol){
  
    double Ts, VolInten;
    
    Ts = T_Vol[vIndex] * T_Vol[vIndex];
    
    VolInten = kl_Vol[vIndex] * SB * Ts * Ts * a_Vol[vIndex] / pi;  
    
    return VolInten;
    
}
  




