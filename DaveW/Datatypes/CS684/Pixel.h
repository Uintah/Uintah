
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

#ifndef SCI_DaveW_Datatypes_Pixel_h
#define SCI_DaveW_Datatypes_Pixel_h 1

#include <DaveW/Datatypes/CS684/Spectrum.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/TriDiagonalMatrix.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Persistent/Persistent.h>

namespace DaveW {
namespace Datatypes {

using SCICore::GeomSpace::CharColor;
using SCICore::Datatypes::DenseMatrix;

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

void Pio( Piostream &, Pixel & );
} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/08/25 03:35:47  sparker
// *** empty log message ***
//
// Revision 1.1  1999/08/23 02:52:56  dmw
// Dave's Datatypes
//
// Revision 1.2  1999/05/03 04:52:01  dmw
// Added and updated DaveW Datatypes/Modules
//
//

#endif
