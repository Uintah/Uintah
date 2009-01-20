#include "VirtualSurface.h"
#include "Consts.h"

#include <cmath>
#include <iostream>
#include <cstdlib>

VirtualSurface::VirtualSurface(){
}

VirtualSurface::~VirtualSurface(){
}

void VirtualSurface::getTheta(double &theta, double &random){
  if (isotropic)
    theta = acos( 1 - 2 * random);
}

// get sIn
void VirtualSurface::get_sIn(double *sIncoming){
  for ( int i = 0; i < 3; i ++ )
    sIn[i] = sIncoming[i];
}

// get_e1, sIn = sIn ( incoming direction vector)
void VirtualSurface::get_e1(double &random1, double &random2, double &random3){
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

void VirtualSurface::get_e2(){
  
  // e2 = sIn * e1
  e2[0] = sIn[1] * e1[2] - sIn[2] * e1[1]; //i
  e2[1] = sIn[2] * e1[0] - sIn[0] * e1[2]; //j
  e2[2] = sIn[0] * e1[1] - sIn[1] * e1[0]; //k
}


void VirtualSurface::get_s(RNG &rng,double *sIncoming,
			   double &theta, double &phi,
			   double &random1, double &random2, double &random3,
			   double *s){

  get_sIn(sIncoming);
  
  // how to generate random numbers @ [-1, 1]
  rng.RandomNumberGen(random1);
  rng.RandomNumberGen(random2);
  rng.RandomNumberGen(random3);
  
  get_e1(random1, random2, random3);
  get_e2();

  rng.RandomNumberGen(random1);
  getTheta(theta, random1);

  rng.RandomNumberGen(random2);
  getPhi(phi, random2);

  for ( int i = 0; i < 3; i ++ ) 
    s[i] = sin(theta) * ( cos(phi) * e1[i] + sin(phi) * e2[i] )
      + cos(theta) * sIn[i] ;
}  
  
  
