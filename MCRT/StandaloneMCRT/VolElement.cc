#include <iostream>
#include "VolElement.h"
#include "RNG.h"
#include "Consts.h"

using std::cout;
using std::endl;

class RNG;

VolElement::VolElement(){ 
}

// inline
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


int VolElement::get_VolIndex(){

  return VolIndex;
  
}



VolElement::~VolElement(){
}

void VolElement::get_limits(const double *X,
			    const double *Y,
			    const double *Z) {

  xlow = X[VoliIndex];
  xup = X[VoliIndex+1];

  ylow = Y[VoljIndex];
  yup = Y[VoljIndex+1];
  
  zlow = Z[VolkIndex];
  zup = Z[VolkIndex+1];


}



double VolElement::get_xlow(){
  return xlow;
}


double VolElement::get_xup(){
  return xup;
}


double VolElement::get_ylow(){
  return ylow;
}


double VolElement::get_yup(){
  return yup;
}


double VolElement::get_zlow(){
  return zlow;
}


double VolElement::get_zup(){
  return zup;
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




double VolElement::VolumeIntensity(const int &vIndex,
				   const double *kl_Vol,
				   const double *T_Vol,
				   const double *a_Vol){

  double Ts, VolInten;

  Ts = T_Vol[vIndex] * T_Vol[vIndex];

  VolInten = kl_Vol[vIndex] * SB * Ts * Ts * a_Vol[vIndex] / pi;  
  
  return VolInten;
  
}


