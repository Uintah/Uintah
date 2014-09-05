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
// Reference   : This is an implementation of the method described by
//              Ravi Ramamoorthi and Pat Hanrahan in their SIGGRAPH 2001 
//	      paper, "An Efficient Representation for Irradiance
//	      Environment Maps".
//
****************************************************************************/

#ifndef PREFILTER_H
#define PREFILTER_H 1

#if defined __cplusplus

#include <Packages/rtrt/Core/ppm.h>

class SHCoeffs
{

public:

    float coeffs[9][3];

};

class SHCoeffsMatrix
{

public:

    float matrix[4][4][3];

};

void prefilter( const texture* text, SHCoeffs& coeffs );
void tomatrix( const SHCoeffs& coeffs, SHCoeffsMatrix& matrix );

#endif
#endif
