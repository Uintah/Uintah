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


   // with scattering media 1D slab with cold black walls
   
   double omega, ext;
   double kl;
   double T;

// Reference:Original case description:
// the astrophysical journal vol 149 1967. page 655- 664
// "Reflection and transmission of light by a thick atmosphere according to a phase function
// 1 + x costheta" -- Busbridge and Orchard.

// Cited by:
// Journal of Heat Transfer 1987 Vol 109 page 809-812
// "Discrete Ordinate Methods for Radiative Heat Transfer in Isotropically and Anisotropically
// Scattering Media" by W. A. Fiveland

// to compare hemispherical reflectance,
// simply set medium T = 0, kl = 0, only scattering coeff.
// and bottom surface T = 0, but to set Top T so that deltaT^4 = 1.
// set both top and bottom surface to be black surfaces to mimic transparent surfaces
// then the Incident flux on Top surface = the outgoing blackbody intensity.
// the reflected flux on top surface = incoming intensity
// reflectance = incoming/outgoing
// q = out - incoming
// thus reflectance = 1 - q

//   scat = 10;
   ext = scat;
   
   kl = ext - scat;
   
   T = sqrt(sqrt(1/SB));
   
   for ( int i = 0; i < VolElementNo; i ++ ) { 
     
     T_Vol[i] = 0;
     kl_Vol[i] = kl;
     scatter_Vol[i] = scat;
     a_Vol[i] = 1;
   }

   
   // top bottom black surfaces     
   for (int i = 0; i < TopBottomNo; i ++ ) {
     
     rs_surface[TOP][i] = 0;
     emiss_surface[TOP][i] = 1;
     alpha_surface[TOP][i] = emiss_surface[TOP][i]; // for gray diffuse surface
     rd_surface[TOP][i] =  1 - rs_surface[TOP][i] - alpha_surface[TOP][i];
     T_surface[TOP][i] = T;
     a_surface[TOP][i] = 1;

     rs_surface[BOTTOM][i] = 0;
     emiss_surface[BOTTOM][i] = 1;
     alpha_surface[BOTTOM][i] = emiss_surface[BOTTOM][i]; // for gray diffuse surface
     rd_surface[BOTTOM][i] =  1 - rs_surface[BOTTOM][i] - alpha_surface[BOTTOM][i];
     T_surface[BOTTOM][i] = 0;
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
     

   // left right surfaces mirrors
   for ( int i = 0; i < LeftRightNo; i ++ ) {     
     rs_surface[LEFT][i] = 1;
     emiss_surface[LEFT][i] = 0;
     alpha_surface[LEFT][i] = emiss_surface[LEFT][i]; // for gray diffuse surface
     rd_surface[LEFT][i] =  1 - rs_surface[LEFT][i] - alpha_surface[LEFT][i];
     T_surface[LEFT][i] = 0;
     a_surface[LEFT][i] = 1;
     
     rs_surface[RIGHT][i] = 1;
     emiss_surface[RIGHT][i] = 0;
     alpha_surface[RIGHT][i] = emiss_surface[RIGHT][i]; // for gray diffuse surface
     rd_surface[RIGHT][i] =  1 - rs_surface[RIGHT][i] - alpha_surface[RIGHT][i];
     T_surface[RIGHT][i] = 0;
     a_surface[RIGHT][i] = 1;
   }

// ++++ end of scattering media 1D slab  ++++
