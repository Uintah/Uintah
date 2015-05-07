/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

  
   //   *********  the benchmark coupled with surfaces case ***************

int fakeIndex = 0;
double xx, yy, zz;
for ( int k = 0; k < Ncz; k ++ ){
  for ( int j = 0; j < Ncy; j ++) {
    for ( int i = 0; i < Ncx; i ++ ) {
      
  T_Vol[fakeIndex] = 64.80721904; // k
  xx = (X[i] + X[i+1])/2;
  yy = (Y[j] + Y[j+1])/2;
  zz = (Z[k] + Z[k+1])/2;
  
  kl_Vol[fakeIndex] = 0.9 * ( 1 - 2 * abs ( xx ) )
    * ( 1 - 2 * abs ( yy ) )
    * ( 1 - 2 * abs ( zz ) ) + 0.1;
  
  a_Vol[fakeIndex] = 1;
  scatter_Vol[fakeIndex] = 0;
  fakeIndex++;
  
    }
  }
 }

// top bottom surface
for ( int i = 0; i < TopBottomNo; i ++ ) { 
  rs_surface[TOP][i] = 0.02;
  emiss_surface[TOP][i] = 0.9;
  alpha_surface[TOP][i] = emiss_surface[TOP][i]; // for gray diffuse surface??
  rd_surface[TOP][i] =  1 - rs_surface[TOP][i] - alpha_surface[TOP][i];
  T_surface[TOP][i] = 1200;
  a_surface[TOP][i] = 1;
  
  rs_surface[BOTTOM][i] = 0.04;
  emiss_surface[BOTTOM][i] = 0.8;
  alpha_surface[BOTTOM][i] = emiss_surface[BOTTOM][i]; // for gray diffuse surface??
  rd_surface[BOTTOM][i] =  1 - rs_surface[BOTTOM][i] - alpha_surface[BOTTOM][i];
  T_surface[BOTTOM][i] = 900;
  a_surface[BOTTOM][i] = 1;
 }
   

// front back surface
for ( int i = 0; i < FrontBackNo; i ++ ) { 
  rs_surface[FRONT][i] = 0.475;
  emiss_surface[FRONT][i] = 0.05;
  alpha_surface[FRONT][i] = emiss_surface[FRONT][i]; // for gray diffuse surface??
  rd_surface[FRONT][i] =  1 - rs_surface[FRONT][i] - alpha_surface[FRONT][i];
  T_surface[FRONT][i] = 1400;
  a_surface[FRONT][i] = 1;
  
  rs_surface[BACK][i] = 0.19;
  emiss_surface[BACK][i] = 0.05;
  alpha_surface[BACK][i] = emiss_surface[BACK][i]; // for gray diffuse surface??
  rd_surface[BACK][i] =  1 - rs_surface[BACK][i] - alpha_surface[BACK][i];
  T_surface[BACK][i] = 2000;
  a_surface[BACK][i] = 1;
 }



// left right surface
for ( int i = 0; i < LeftRightNo; i ++ ) { 
  rs_surface[LEFT][i] = 0.76;
  emiss_surface[LEFT][i] = 0.2;
  alpha_surface[LEFT][i] = emiss_surface[LEFT][i]; // for gray diffuse surface??
  rd_surface[LEFT][i] =  1 - rs_surface[LEFT][i] - alpha_surface[LEFT][i];
  T_surface[LEFT][i] = 600;
  a_surface[LEFT][i] = 1;

  rs_surface[RIGHT][i] = 0.76;
  emiss_surface[RIGHT][i] = 0.2;
  alpha_surface[RIGHT][i] = emiss_surface[RIGHT][i]; // for gray diffuse surface??
  rd_surface[RIGHT][i] =  1 - rs_surface[RIGHT][i] - alpha_surface[RIGHT][i];
  T_surface[RIGHT][i] = 600;
  a_surface[RIGHT][i] = 1;
 }

// ************************* end of this case ********************
   
