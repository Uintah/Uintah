/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


 
   // ************* parallel plates for testing on surface case (no media)******************
   
   // Modest Book, page 225
   double T, emissSur, rhos;
   
   for ( int i = 0; i < VolElementNo ; i ++ ) {
     
     T_Vol[i] = 0; // k      
     kl_Vol[i] = 0; // set as "1" to let absorb happens all the time
     a_Vol[i] = 1;
     scatter_Vol[i] = 0;
   }


   // emiss*SB*T^4 =1
   // on top & bottom and SD doesnot change much with rayNo or StopLowerBound
   if (casePlates == 11){ // SD top bottom = 0.147
     emissSur = 0.9;
     rhos = 0;
   }
   else if(casePlates == 12) { // SD top bottom = 0.1505
     emissSur = 0.9;
     rhos = 0.1;
   }
   else if ( casePlates == 21 ) { // SD top bottom = 0.105
     emissSur = 0.5;
     rhos = 0;
   }
   else if ( casePlates == 22 ) { // SD top bottom = 0.111
     emissSur = 0.5;
     rhos = 0.25;
   }
   else if ( casePlates == 23 ) { // SD top bottom = 0.121 
     emissSur = 0.5;
     rhos = 0.5;
   }
   else if ( casePlates == 31 ) { // SD top bottom = 0.031
     emissSur = 0.1;
     rhos = 0;
   }
   else if ( casePlates == 32 ) { // SD top bottom = 0.042
     emissSur = 0.1;
     rhos = 0.6;
   }
   else if ( casePlates == 33 ) { // SD top bottom = 0.068
     emissSur = 0.1;
     rhos = 0.9;
   }
   
  
   T = sqrt(sqrt(1/emissSur/SB));

   // top bottom surfaces has the property
   // front and back surfaces are pure specular cold surfaces
   // left and right surfaces are cold black surfaces

   // top bottom surface
   for ( int i = 0; i < TopBottomNo; i ++ ) { 
     rs_top_surface[i] = rhos;
     emiss_top_surface[i] = emissSur;
     alpha_top_surface[i] = emiss_top_surface[i]; // for gray diffuse surface
     rd_top_surface[i] =  1 - rs_top_surface[i] - alpha_top_surface[i];
     T_top_surface[i] = T;
     a_top_surface[i] = 1;     

     rs_bottom_surface[i] = rhos;
     emiss_bottom_surface[i] = emissSur;
     alpha_bottom_surface[i] = emiss_bottom_surface[i]; // for gray diffuse surface
     rd_bottom_surface[i] =  1 - rs_bottom_surface[i] - alpha_bottom_surface[i];
     T_bottom_surface[i] = T;
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
     
 
   // left right black surfaces     
   for (int i = 0; i < LeftRightNo; i ++ ) {
     
     rs_left_surface[i] = 0;
     emiss_left_surface[i] = 1;
     alpha_left_surface[i] = emiss_left_surface[i]; // for gray diffuse surface
     rd_left_surface[i] =  1 - rs_left_surface[i] - alpha_left_surface[i];
     T_left_surface[i] = 0;
     a_left_surface[i] = 1;

     rs_right_surface[i] = 0;
     emiss_right_surface[i] = 1;
     alpha_right_surface[i] = emiss_right_surface[i]; // for gray diffuse surface
     rd_right_surface[i] =  1 - rs_right_surface[i] - alpha_right_surface[i];
     T_right_surface[i] = 0;
     a_right_surface[i] = 1;
     
   }
       
   // ********* end of parallel plates case ******************************
   
 
