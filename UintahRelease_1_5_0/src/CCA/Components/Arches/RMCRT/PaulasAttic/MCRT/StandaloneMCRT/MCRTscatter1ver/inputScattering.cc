/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

   // with scattering media 1D slab with cold black walls
   
   double omega, ext, scat;
   double kl;
   double T;
   
   // more scattering will reduce the SD for Volume
   // when increase the extinction coeff, SD VOL increases
   
   // SD vol = center 0.049 continuously to-> boundary 0.078
   if (casePlates == 1 ) { // SD top bottom = 0.055; 
     omega = 0.3;
     ext = 1; // [m^-1]
   }
   // with more scattering SD Vol is much smaller: from center 0.0034 to boundary 0.0039
   else if (casePlates == 2 ) { // SD top bottom = 0.035;
     omega = 0.9;
     ext = 1;
   }
   // SD Vol = 0.7, 0.2, 0.09, 0.03, 0.01, 0.03, 0.09, 0.2, 0.7
   else if (casePlates == 3){
     // SD top bottom = 0.057 with rayNo = 1000, stop = 1e-5;
     // SD top bottom = 0.059 with rayNo = 5000, stop = 1e-5;
     // SD top bottom = 0.059 with rayNo = 5000, stop = 1e-10;
     
     omega = 0.3;
     ext = 10;
   }
   // SD Vol = 0.117, 0.097, 0.074, 0.054, 0.043, 0.054, 0.074. 0.097, 0.117
   else if ( casePlates == 4 ) {
     // SD top bottom = 0.113 with rayNo = 1000, stop = 1e-5
     // SD top bottom = 0.114 with rayNo = 5000, stop = 1e-10;
     
     omega = 0.9;
     ext = 10;
   }
   
   scat = omega * ext;
  
   kl = ext - scat;
   
   T = sqrt(sqrt(1/SB));
   
   for ( int i = 0; i < VolElementNo; i ++ ) { 
     
     T_Vol[i] = T;
     kl_Vol[i] = kl;
     scatter_Vol[i] = scat;
     a_Vol[i] = 1;
   }

   
   // top bottom black surfaces     
   for (int i = 0; i < TopBottomNo; i ++ ) {
     
     rs_top_surface[i] = 0;
     emiss_top_surface[i] = 1;
     alpha_top_surface[i] = emiss_top_surface[i]; // for gray diffuse surface
     rd_top_surface[i] =  1 - rs_top_surface[i] - alpha_top_surface[i];
     T_top_surface[i] = 0;
     a_top_surface[i] = 1;

     rs_bottom_surface[i] = 0;
     emiss_bottom_surface[i] = 1;
     alpha_bottom_surface[i] = emiss_bottom_surface[i]; // for gray diffuse surface
     rd_bottom_surface[i] =  1 - rs_bottom_surface[i] - alpha_bottom_surface[i];
     T_bottom_surface[i] = 0;
     a_bottom_surface[i] = 1;
     
   }

   
   // front back surfaces mirrors
   for ( int i = 0; i < FrontBackNo; i ++ ) {     
     rs_front_surface[i] = 1;
     emiss_front_surface[i] = 0;
     alpha_front_surface[i] = emiss_front_surface[i]; // for gray diffuse surface
     rd_front_surface[i] =  1 - rs_front_surface[i] - alpha_front_surface[i];
     T_front_surface[i] = 0;
     a_front_surface[i] = 1;
     
     rs_back_surface[i] = 1;
     emiss_back_surface[i] = 0;
     alpha_back_surface[i] = emiss_back_surface[i]; // for gray diffuse surface
     rd_back_surface[i] =  1 - rs_back_surface[i] - alpha_back_surface[i];
     T_back_surface[i] = 0;
     a_back_surface[i] = 1;
   }
     

   // left right surfaces mirrors
   for ( int i = 0; i < LeftRightNo; i ++ ) {     
     rs_left_surface[i] = 1;
     emiss_left_surface[i] = 0;
     alpha_left_surface[i] = emiss_left_surface[i]; // for gray diffuse surface
     rd_left_surface[i] =  1 - rs_left_surface[i] - alpha_left_surface[i];
     T_left_surface[i] = 0;
     a_left_surface[i] = 1;
     
     rs_right_surface[i] = 1;
     emiss_right_surface[i] = 0;
     alpha_right_surface[i] = emiss_right_surface[i]; // for gray diffuse surface
     rd_right_surface[i] =  1 - rs_right_surface[i] - alpha_right_surface[i];
     T_right_surface[i] = 0;
     a_right_surface[i] = 1;
   }

// ++++ end of scattering media 1D slab  ++++
