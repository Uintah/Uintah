/****************************************************************************
//
// Filename: prefilter.c
//
// Prefilter environment maps, computing spherical harmonic
// lighting coefficients. 
//
//
// Output      : RGB values for lighting coefficients L_{lm} with 
//              0 <= l <= 2 and -l <= m <= l.  There are 9 coefficients
//      	in all.
//
//Reference   : This is an implementation of the method described by
//              Ravi Ramamoorthi and Pat Hanrahan in their SIGGRAPH 2001 
//	      paper, "An Efficient Representation for Irradiance
//	      Environment Maps".
//
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include <Packages/rtrt/Core/Color.h>

#ifndef PREFILTER_H
#include <Packages/rtrt/Core/prefilter.h>
#endif

using namespace rtrt;

inline float sinc( float x ) {               /* Supporting sinc function */
  if (fabs(x) < 1.0e-4) return 1.0 ;
  else return(sin(x)/x) ;
}
     
void
updatecoeffs( const Color& color,
	      float domega, float x, float y, float z,
	      SHCoeffs& coeffs )
{ 
    
  /****************************************************************** 
   Update the coefficients (i.e. compute the next term in the
   integral) based on the lighting value hdr[3], the differential
   solid angle domega and cartesian components of surface normal x,y,z

   Inputs:  hdr = L(x,y,z) [note that x^2+y^2+z^2 = 1]
            i.e. the illumination at position (x,y,z)

            domega = The solid angle at the pixel corresponding to 
	    (x,y,z).  For these light probes, this is given by 

	    x,y,z  = Cartesian components of surface normal

   Notes:   Of course, there are better numerical methods to do
            integration, but this naive approach is sufficient for our
	    purpose.

  *********************************************************************/

    for( int col = 0; col < 3; col++ ) {
    float c ; /* A different constant for each coefficient */

    /* L_{00}.  Note that Y_{00} = 0.282095 */
    c = 0.282095 ;
    coeffs.coeffs[0][col] += color[col]*c*domega ;

    /* L_{1m}. -1 <= m <= 1.  The linear terms */
    c = 0.488603 ;
    coeffs.coeffs[1][col] += color[col]*(c*y)*domega ;   /* Y_{1-1} = 0.488603 y  */
    coeffs.coeffs[2][col] += color[col]*(c*z)*domega ;   /* Y_{10}  = 0.488603 z  */
    coeffs.coeffs[3][col] += color[col]*(c*x)*domega ;   /* Y_{11}  = 0.488603 x  */

    /* The Quadratic terms, L_{2m} -2 <= m <= 2 */

    /* First, L_{2-2}, L_{2-1}, L_{21} corresponding to xy,yz,xz */
    c = 1.092548 ;
    coeffs.coeffs[4][col] += color[col]*(c*x*y)*domega ; /* Y_{2-2} = 1.092548 xy */ 
    coeffs.coeffs[5][col] += color[col]*(c*y*z)*domega ; /* Y_{2-1} = 1.092548 yz */ 
    coeffs.coeffs[7][col] += color[col]*(c*x*z)*domega ; /* Y_{21}  = 1.092548 xz */ 

    /* L_{20}.  Note that Y_{20} = 0.315392 (3z^2 - 1) */
    c = 0.315392 ;
    coeffs.coeffs[6][col] += color[col]*(c*(3*z*z-1))*domega ; 

    /* L_{22}.  Note that Y_{22} = 0.546274 (x^2 - y^2) */
    c = 0.546274 ;
    coeffs.coeffs[8][col] += color[col]*(c*(x*x-y*y))*domega ;

  }
}

void
prefilter( const texture* text,
	   SHCoeffs& coeffs )
{
    
    /* The main integration routine.  Of course, there are better ways
       to do quadrature but this suffices.  Calls updatecoeffs to
       actually increment the integral. Width is the size of the
       environment map */
    
    int width = text->img_width;
    int height = text->img_height;
    int halfWidth = int( 0.5 * width );
    int halfHeight = int( 0.5 * height );
    float C1 = 2.0*M_PI / width;
    float C2 = 2.0*M_PI / height;
    Color color;
    char* ppmImage = text->texImage;
    float* floatImage = text->texImagef;
    
    for( int i = 0; i < width; i++ )
	for( int j = 0 ; j < height; j++ ) {

	    /* We now find the cartesian components for the point (i,j) */
	    
	    float v = ( halfHeight - j) / halfHeight;  /* v ranges from -1 to 1 */
	    float u = ( i - halfWidth ) / halfWidth;    /* u ranges from -1 to 1 */
	    float r = sqrt( u*u + v*v );               /* The "radius" */
	    if( r > 1.0 )
		continue;           /* Consider only circle with r<1 */
	    
	    float theta = M_PI*r;                    /* theta parameter of (i,j) */
	    float phi = atan2( v, u );               /* phi parameter */
	    
	    float x = sin( theta ) * cos( phi );         /* Cartesian components */
	    float y = sin( theta ) * sin( phi );
	    float z = cos( theta );

      /* Computation of the solid angle.  This follows from some
         elementary calculus converting sin(theta) d theta d phi into
         coordinates in terms of r.  This calculation should be redone 
         if the form of the input changes */

	    float domega = C1*C2*sinc( theta );

	    if( text->type == GFXIO_FLOAT ) {
		double r = *(floatImage++);
		double g = *(floatImage++);
		double b = *(floatImage++);
		color = Color( r, g, b );
	    } else if( text->type == GFXIO_UBYTE ) {
		double r = ppmImage[0] / 255.;
		double g = ppmImage[1] / 255.;
		double b = ppmImage[2] / 255.;
		ppmImage += 4;
		color = Color( r, g, b );
	    }
		
	    updatecoeffs( color,
			  domega, x, y, z,
			  coeffs ); /* Update Integration */
	}
}

//
// Convert spherical harmonics coefficients into matrix quadratic form
//
void
tomatrix( const SHCoeffs& coeffs,
	  SHCoeffsMatrix& matrix )
{
    
  /* Form the quadratic form matrix (see equations 11 and 12 in paper) */

  int col ;
  float c1,c2,c3,c4,c5 ;
  c1 = 0.429043 ; c2 = 0.511664 ;
  c3 = 0.743125 ; c4 = 0.886227 ; c5 = 0.247708 ;

  for (col = 0 ; col < 3 ; col++) { /* Equation 12 */

    matrix.matrix[0][0][col] = c1*coeffs.coeffs[8][col] ; /* c1 L_{22}  */
    matrix.matrix[0][1][col] = c1*coeffs.coeffs[4][col] ; /* c1 L_{2-2} */
    matrix.matrix[0][2][col] = c1*coeffs.coeffs[7][col] ; /* c1 L_{21}  */
    matrix.matrix[0][3][col] = c2*coeffs.coeffs[3][col] ; /* c2 L_{11}  */

    matrix.matrix[1][0][col] = c1*coeffs.coeffs[4][col] ; /* c1 L_{2-2} */
    matrix.matrix[1][1][col] = -c1*coeffs.coeffs[8][col]; /*-c1 L_{22}  */
    matrix.matrix[1][2][col] = c1*coeffs.coeffs[5][col] ; /* c1 L_{2-1} */
    matrix.matrix[1][3][col] = c2*coeffs.coeffs[1][col] ; /* c2 L_{1-1} */

    matrix.matrix[2][0][col] = c1*coeffs.coeffs[7][col] ; /* c1 L_{21}  */
    matrix.matrix[2][1][col] = c1*coeffs.coeffs[5][col] ; /* c1 L_{2-1} */
    matrix.matrix[2][2][col] = c3*coeffs.coeffs[6][col] ; /* c3 L_{20}  */
    matrix.matrix[2][3][col] = c2*coeffs.coeffs[2][col] ; /* c2 L_{10}  */

    matrix.matrix[3][0][col] = c2*coeffs.coeffs[3][col] ; /* c2 L_{11}  */
    matrix.matrix[3][1][col] = c2*coeffs.coeffs[1][col] ; /* c2 L_{1-1} */
    matrix.matrix[3][2][col] = c2*coeffs.coeffs[2][col] ; /* c2 L_{10}  */
    matrix.matrix[3][3][col] = c4*coeffs.coeffs[0][col] - c5*coeffs.coeffs[6][col] ; 
                                            /* c4 L_{00} - c5 L_{20} */
  }
}







