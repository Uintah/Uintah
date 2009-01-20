//pseudo-random number generator

#include "RNG.h"

#include <cmath>
#include <ctime>
#include <cstdlib> // for drand48 function


RNG::RNG(){
}

RNG::~RNG(){
}

void RNG::RandomNumberGen(double &random){
  random = drand48();
}

 

// double  *RNG::RandomNumberGen(){
 
//   for (int i = 0; i < 2 ; i ++ )

    
//     //rng[i] = (double) rand()/ RAND_MAX ;// [0 1] all inclusive
    
//     rng[i] = drand48(); // return non-negative double-precision floating-point values uniformly distributed between [0.0, 1.0).
  
//   // if put (double) (rand() / RAND_MAX), will get all zero,
//   // this is [0,1], 0 and 1 inclusive
//   // if (float) rand() / (RAND_MAX + 1), this is [0,1), 1 is exclusive
//   // RAND_MAX = 2147483647
  
//   // rng[i] = ( rand() % 1000000 ) / 1000000. ;
//   // more accurate, more digits it has, less to get slotnumber???/ right,
//   // rand() % 1000 / 1000. constrains it from 0 to 1, and put the accuracy into 3 digital numbers,
    
//   return rng;
// }


