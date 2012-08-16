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
     rs_surface[TOP][i] = rhos;
     emiss_surface[TOP][i] = emissSur;
     alpha_surface[TOP][i] = emiss_surface[TOP][i]; // for gray diffuse surface
     rd_surface[TOP][i] =  1 - rs_surface[TOP][i] - alpha_surface[TOP][i];
     T_surface[TOP][i] = T;
     a_surface[TOP][i] = 1;     

     rs_surface[BOTTOM][i] = rhos;
     emiss_surface[BOTTOM][i] = emissSur;
     alpha_surface[BOTTOM][i] = emiss_surface[BOTTOM][i]; // for gray diffuse surface
     rd_surface[BOTTOM][i] =  1 - rs_surface[BOTTOM][i] - alpha_surface[BOTTOM][i];
     T_surface[BOTTOM][i] = T;
     a_surface[BOTTOM][i] = 1;     
   }
     

   // front back surfaces mirrors
   for ( int i = 0; i < FrontBackNo; i ++ ) {     
     rs_surface[FRONT][i] = 1;
     emiss_surface[FRONT][i] = 0;
     alpha_surface[FRONT][i] = emiss_surface[FRONT][i]; // for gray diffuse surface
     rd_surface[FRONT][i] =  1 - rs_surface[FRONT][i] - alpha_surface[FRONT][i];
     T_surface[FRONT][i] = 0;
     a_surface[FRONT][i] = 1;
     
     rs_surface[BACK][i] = 1;
     emiss_surface[BACK][i] = 0;
     alpha_surface[BACK][i] = emiss_surface[BACK][i]; // for gray diffuse surface
     rd_surface[BACK][i] =  1 - rs_surface[BACK][i] - alpha_surface[BACK][i];
     T_surface[BACK][i] = 0;
     a_surface[BACK][i] = 1;
   }
     
 
   // left right black surfaces     
   for (int i = 0; i < LeftRightNo; i ++ ) {
     
     rs_surface[LEFT][i] = 0;
     emiss_surface[LEFT][i] = 1;
     alpha_surface[LEFT][i] = emiss_surface[LEFT][i]; // for gray diffuse surface
     rd_surface[LEFT][i] =  1 - rs_surface[LEFT][i] - alpha_surface[LEFT][i];
     T_surface[LEFT][i] = 0;
     a_surface[LEFT][i] = 1;

     rs_surface[RIGHT][i] = 0;
     emiss_surface[RIGHT][i] = 1;
     alpha_surface[RIGHT][i] = emiss_surface[RIGHT][i]; // for gray diffuse surface
     rd_surface[RIGHT][i] =  1 - rs_surface[RIGHT][i] - alpha_surface[RIGHT][i];
     T_surface[RIGHT][i] = 0;
     a_surface[RIGHT][i] = 1;
     
   }
       
   // ********* end of parallel plates case ******************************
   
 
