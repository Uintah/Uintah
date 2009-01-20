#include "RealSurface.h"

#include <cmath>
#include <iostream>
#include <cstdlib>

using std::cout;
using std::endl;

//const double R = 8;
//const double H = 20;

RealSurface::RealSurface(){
}

// RealSurface::RealSurface(int _surfaceIndex){
//   surfaceIndex =  _surfaceIndex;
// }
  
void RealSurface::get_public_limits(double &_alow, double &_aup,
				    double &_blow, double &_bup,
				    double &_constv){
  _alow = alow;
  _aup = aup;
  _blow = blow;
  _bup = bup;
  _constv = constv;
}


void RealSurface::getTheta(double &theta, double &random){
  theta = asin(sqrt(random));
}

void RealSurface::get_s(RNG &rng,
			double &theta, double &random1,
			double &phi, double &random2,
			double *s){

  // use this pointer, so to get to 6 different surfaces
  //cout << " line 41 " << endl;
  this->get_n();
  //cout << " line 43 " << endl;
  this->get_t1();
  this->get_t2();
  //cout << " line 44 " << endl;
  
  rng.RandomNumberGen(random1);

  this->getTheta(theta,random1);
  rng.RandomNumberGen(random2);

  // getPhi is inherited from Class Surface
  this->getPhi(phi,random2);

  //cout << " line 54 " << endl;
  
  for ( int i = 0; i < 3; i ++ ) 
    s[i] = sin(theta) * ( cos(phi) * this->t1[i] + sin(phi) * this->t2[i] )
      + cos(theta) * this->n[i] ;
  
}

RealSurface::~RealSurface(){
}


