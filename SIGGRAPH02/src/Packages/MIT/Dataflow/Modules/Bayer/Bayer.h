

#ifndef Bayer_h
#define Bayer_h

#include "cvode/llnltyps.h"
#include "cvode/nvector.h"

typedef void(F)( int, double, N_Vector, N_Vector, void * );


class Bayer {
public:
  static void bayer_function_(int, double, N_Vector, N_Vector, void * );

 private:
  double k1, k21, k22, k31, k32, k2;

 public:
  Bayer() {}

  void set_p( double * );
  void get_y0( N_Vector y );
  
  F *f() { return bayer_function_; }
};

#endif Bayer_h
