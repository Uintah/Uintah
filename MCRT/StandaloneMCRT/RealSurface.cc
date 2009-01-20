#include "RealSurface.h"
#include "Consts.h"

#include <cmath>
#include <iostream>
#include <cstdlib>

using std::cout;
using std::endl;

RealSurface::RealSurface(){
}


void RealSurface::getTheta(const double &random){
  theta = asin(sqrt(random));
}


int RealSurface::get_surfaceIndex(){
  return this->surfaceIndex;
}


double RealSurface::get_xlow(){
  return this->xlow;
}

double RealSurface::get_xup(){
  return this->xup;
}


double RealSurface::get_ylow(){
  return this->ylow;
}


double RealSurface::get_yup(){
  return this->yup;
}


double RealSurface::get_zlow(){
  return this->zlow;
}


double RealSurface::get_zup(){
  return this->zup;
}


int RealSurface::get_surfaceiIndex(){
  return this->surfaceiIndex;
}


int RealSurface::get_surfacejIndex(){
  return this->surfacejIndex;
}


int RealSurface::get_surfacekIndex(){
  return this->surfacekIndex;
}



void RealSurface::get_s(RNG &rng, double *s){
  
  double random1, random2;
  // use this pointer, so to get to 6 different surfaces
  this->get_n();
  this->get_t1();
  this->get_t2();
  
  rng.RandomNumberGen(random1);
  this->getTheta(random1);

  rng.RandomNumberGen(random2);
  // getPhi is inherited from Class Surface
  this->getPhi(random2);

//   cout << "random1 = " << random1 << "; random2 = " << random2 << endl;
//   cout << " theta = " << theta << "; phi = " << phi << endl;
  
  for ( int i = 0; i < 3; i ++ ) 
    s[i] = sin(theta) * ( cos(phi) * this->t1[i] + sin(phi) * this->t2[i] )
      + cos(theta) * this->n[i] ;
  
//   cout << "n = " << n[0] << "i + " << n[1] << "j+" << n[2] << "k" << endl;
//   cout << "t1 = " << t1[0] << "i + " << t1[1] << "j+" << t1[2] << "k" << endl;
//   cout << "t2 = " << t2[0] << "i + " << t2[1] << "j+" << t2[2] << "k" << endl;
//   cout << "s = " << s[0] << "i + " << s[1] << "j + " << s[2] << "k" << endl;
  
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


