/*
 *  Pixel.h: Pixel storage for a Ray Matrix image
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_Pixel_h
#define SCI_project_Pixel_h 1

#include <Classlib/Array1.h>
#include <Classlib/Persistent.h>
#include <Datatypes/DenseMatrix.h>
#include <Datatypes/ColumnMatrix.h>
#include <Datatypes/Spectrum.h>
#include <Datatypes/TriDiagonalMatrix.h>
#include <Geom/Color.h>
#include <Geometry/Point.h>

class Pixel {
public:
    int nSamples;
    LiteSpectrum spec;	
    Point xyz;
    CharColor c;		// color of this pixel
    Array1<DenseMatrix> S;	// specular reflected
    Array1<DenseMatrix> D;	// direct diffuse
    DenseMatrix R;		// all the S and D matrices composited
    Array1<double> S0;		// light from the first materials to the eye
    Array1<double> E;		// emitted light
public:
    Pixel();
    Pixel(const Pixel &copy);
    Pixel(const Array1<DenseMatrix>& S, const Array1<DenseMatrix>& D, 
	  const DenseMatrix& R, const Array1<double>& S0, 
	  const Array1<double>& E, const LiteSpectrum& spec, 
	  const Point &xyz, const CharColor& c);
    ~Pixel();
};
void Pio(Piostream&, Pixel&);

#endif
