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
   
   double kl;
   double T;
   double emiss_top, emiss_bottom;

   // more scattering will reduce the SD for Volume
   // when increase the extinction coeff, SD VOL increases
   
   if (casePlates == 11 ) { 
     scat = 0.1;
     emiss_top = 0.8;
     emiss_bottom = 1.0;
     
   }
   // with more scattering SD Vol is much smaller: from center 0.0034 to boundary 0.0039
   else if (casePlates == 12 ) {
      scat = 1.0;
     emiss_top = 0.8;
     emiss_bottom = 1.0;    
   }
   else if (casePlates == 13){
     scat = 3.0;
     emiss_top = 0.8;
     emiss_bottom = 1.0;
   }
   else if ( casePlates == 21 ) {
     scat = 0.1;
     emiss_top = 0.8;
     emiss_bottom = 0.5;     
   }
   else if ( casePlates == 22 ) {
     scat = 1.0;
     emiss_top = 0.8;
     emiss_bottom = 0.5;          
     }
   else if ( casePlates == 23 ) {
     scat = 3.0;
     emiss_top = 0.8;
     emiss_bottom = 0.5;    
   }
   else if ( casePlates == 31 ) {
     scat = 0.1;
     emiss_top = 0.8;
     emiss_bottom = 0.1;     
   }
   else if ( casePlates == 32 ) {
     scat = 1.0;
     emiss_top = 0.8;
     emiss_bottom = 0.1;          
     }
   else if ( casePlates == 33 ) {
     scat = 3.0;
     emiss_top = 0.8;
     emiss_bottom = 0.1;    
   }



  
   kl = 0;
   
   T = sqrt(sqrt(1/SB));
   
   for ( int i = 0; i < VolElementNo; i ++ ) { 
     
     T_Vol[i] = 0;
     kl_Vol[i] = kl;
     scatter_Vol[i] = scat;
     a_Vol[i] = 1;
   }

   
   // top bottom black surfaces     
   for (int i = 0; i < TopBottomNo; i ++ ) {
     
     rs_top_surface[i] = 0;
     emiss_top_surface[i] = emiss_top;
     alpha_top_surface[i] = emiss_top_surface[i]; // for gray diffuse surface
     rd_top_surface[i] =  1 - rs_top_surface[i] - alpha_top_surface[i];
     T_top_surface[i] = T;
     a_top_surface[i] = 1;

     rs_bottom_surface[i] = 0;
     emiss_bottom_surface[i] = emiss_bottom;
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
