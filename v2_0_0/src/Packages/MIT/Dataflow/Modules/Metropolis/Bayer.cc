#include <math.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/Bayer.h>


void
Bayer::set_p ( double *p )
{
  k1  = exp( p[0] );
  k21 = exp( p[1] );
  k22 = exp( p[2] );
  k31 = exp( p[3] );
  k32 = exp( p[4] );
  k2  = exp( p[5] );
};

  
void
Bayer::get_y0( N_Vector y )
{
  N_VIth(y,0) = 0.0;
  N_VIth(y,1) = 2.8111;
  N_VIth(y,2) = 2.5148;
  N_VIth(y,3) = 0.0;
  N_VIth(y,4) = 0.0;
  N_VIth(y,5) = 0.0;
  N_VIth(y,6) = 0.0048;
}



void  
Bayer::bayer_function_ ( int n, double t , 
			 N_Vector y, N_Vector y_prime, 
			 void *data )
{
  Bayer &b = *(Bayer *) data;
  N_VIth(y_prime,0) =  b.k1*N_VIth(y,1)*N_VIth(y,2) - b.k21*N_VIth(y,0) 
                     + b.k22*N_VIth(y,3)*N_VIth(y,6) 
                     - b.k2*N_VIth(y,1)*N_VIth(y,0);

  N_VIth(y_prime,1) = - b.k1*N_VIth(y,1)*N_VIth(y,2) 
                      - 2.0*b.k31*N_VIth(y,1)*N_VIth(y,1)
                      + 2.0*b.k32*N_VIth(y,4)*N_VIth(y,6)
                      - b.k2*N_VIth(y,1)*N_VIth(y,0);

  N_VIth(y_prime,2) = - b.k1*N_VIth(y,1)*N_VIth(y,2);

  N_VIth(y_prime,3) =   b.k21*N_VIth(y,0) - b.k22*N_VIth(y,3)*N_VIth(y,6);

  N_VIth(y_prime,4) =   b.k31*N_VIth(y,1)*N_VIth(y,1)
                      - b.k32*N_VIth(y,4)*N_VIth(y,6);

  N_VIth(y_prime,5) =   b.k2*N_VIth(y,1)*N_VIth(y,0);

  N_VIth(y_prime,6) =   b.k21*N_VIth(y,0)
                      - b.k22*N_VIth(y,3)*N_VIth(y,6)
                      + b.k31*N_VIth(y,1)*N_VIth(y,1)
                      - b.k32*N_VIth(y,4)*N_VIth(y,6);
}
