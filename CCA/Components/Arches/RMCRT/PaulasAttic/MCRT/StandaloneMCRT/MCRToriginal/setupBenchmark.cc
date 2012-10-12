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

   
#include <math.h>
#include <stdlib.h>

//  ************************ benchmark case **********************

void
setupBenchmark( int Ncx, int Ncy, int Ncz, // number of cells in x, y, z directions
                double * T_Vol, 
                double * X, double * Y, double * Z,
                double * kl_Vol, double * a_Vol, double * scatter_Vol,
                int TopBottomNo, int FrontBackNo, int LeftRightNo,
                double * a_top_surface,     double * a_bottom_surface, double * a_front_surface, double * a_back_surface, double * a_left_surface, double * a_right_surface,
                double * rs_top_surface,    double * rs_bottom_surface, double * rs_front_surface, double * rs_back_surface, double * rs_left_surface, double * rs_right_surface,
                double * rd_top_surface,    double * rd_bottom_surface, double * rd_front_surface, double * rd_back_surface, double * rd_left_surface, double * rd_right_surface,
                double * alpha_top_surface, double * alpha_bottom_surface, double * alpha_front_surface, double * alpha_back_surface, double * alpha_left_surface, double * alpha_right_surface,
                double * emiss_top_surface, double * emiss_bottom_surface, double * emiss_front_surface, double * emiss_back_surface, double * emiss_left_surface, double * emiss_right_surface,
                double * T_top_surface,     double * T_bottom_surface, double * T_front_surface, double * T_back_surface, double * T_left_surface, double * T_right_surface)
{
  // benchmark case
  int fakeIndex = 0;
  double xx, yy, zz;
  for ( int k = 0; k < Ncz; k ++ ){
    for ( int j = 0; j < Ncy; j ++) {
      for ( int i = 0; i < Ncx; i ++ ) {
	
	T_Vol[fakeIndex] = 64.80721904; // k
	xx = (X[i] + X[i+1])/2;
	yy = (Y[j] + Y[j+1])/2;
	zz = (Z[k] + Z[k+1])/2;

	kl_Vol[fakeIndex] = 0.9 * ( 1 - 2 * fabs ( xx ) )
	  * ( 1 - 2 * fabs ( yy ) )
	  * ( 1 - 2 * fabs ( zz ) ) + 0.1;
	
	a_Vol[fakeIndex] = 1;
	scatter_Vol[fakeIndex] = 0;
	fakeIndex++;
	
      }
    }
  }
  
  
  // with participating media, and all cold black surfaces around
  
  // top bottom surfaces
  for ( int i = 0; i < TopBottomNo; i ++ ) {
    rs_top_surface[i] = 0;
    rs_bottom_surface[i] = 0;

    rd_top_surface[i] = 0;
    rd_bottom_surface[i] = 0;
    
    alpha_top_surface[i] = 1 - rs_top_surface[i] - rd_top_surface[i];
    alpha_bottom_surface[i] = 1 - rs_bottom_surface[i] - rd_bottom_surface[i];
        
    emiss_top_surface[i] = alpha_top_surface[i];
    emiss_bottom_surface[i] = alpha_bottom_surface[i];

    T_top_surface[i] = 0;
    T_bottom_surface[i] = 0;

    a_top_surface[i] = 1;
    a_bottom_surface[i] = 1;
    
  }


  // front back surfaces
  for ( int i = 0; i < FrontBackNo; i ++ ) {
    rs_front_surface[i] = 0;
    rs_back_surface[i] = 0;

    rd_front_surface[i] = 0;
    rd_back_surface[i] = 0;
    
    alpha_front_surface[i] = 1 - rs_front_surface[i] - rd_front_surface[i];
    alpha_back_surface[i] = 1 - rs_back_surface[i] - rd_back_surface[i];
        
    emiss_front_surface[i] = alpha_front_surface[i];
    emiss_back_surface[i] = alpha_back_surface[i];

    T_front_surface[i] = 0;
    T_back_surface[i] = 0;

    a_front_surface[i] = 1;
    a_back_surface[i] = 1;

  }

  
  // from left right surfaces
  for ( int i = 0; i < LeftRightNo; i ++ ) {
    rs_left_surface[i] = 0;
    rs_right_surface[i] = 0;

    rd_left_surface[i] = 0;
    rd_right_surface[i] = 0;
    
    alpha_left_surface[i] = 1 - rs_left_surface[i] - rd_left_surface[i];
    alpha_right_surface[i] = 1 - rs_right_surface[i] - rd_right_surface[i];
        
    emiss_left_surface[i] = alpha_left_surface[i];
    emiss_right_surface[i] = alpha_right_surface[i];

    T_left_surface[i] = 0;
    T_right_surface[i] = 0;

    a_left_surface[i] = 1;
    a_right_surface[1] = 1;
        
    
  }
} // end setupBenchmark()


  
