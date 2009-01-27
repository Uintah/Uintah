/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//pseudo-random number generator

#include "RNG.h"

#include <cmath>
#include <ctime>


RNG::RNG(){
}

RNG::~RNG(){
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


