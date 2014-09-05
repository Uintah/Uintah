
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

#ifndef SCI_Packages_DaveW_Datatypes_Pixel_h
#define SCI_Packages_DaveW_Datatypes_Pixel_h 1

#include <Packages/DaveW/Core/Datatypes/CS684/Spectrum.h>
#include <Core/Containers/Array1.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/TriDiagonalMatrix.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geometry/Point.h>
#include <Core/Persistent/Persistent.h>

namespace DaveW {
using namespace SCIRun;

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



} // End namespace DaveW

namespace SCIRun {
void Pio( Piostream &, DaveW::Pixel & );
}
#endif
