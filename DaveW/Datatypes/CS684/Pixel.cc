//static char *id="@(#) $Id$";

/*
 *  Pixel.cc: Generate Pixel points in a domain
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <iostream.h>

#include <DaveW/Datatypes/CS684/Pixel.h>

namespace DaveW {
namespace Datatypes {

Pixel::Pixel()
{
}

Pixel::Pixel(const Pixel &copy)
: S(copy.S), D(copy.D), R(copy.R), S0(copy.S0), E(copy.E), spec(copy.spec),
  xyz(copy.xyz), c(copy.c)
{
}

Pixel::Pixel(const Array1<DenseMatrix>& S, const Array1<DenseMatrix>& D, 
	     const DenseMatrix& R, const Array1<double>& S0, 
	     const Array1<double>& E, const LiteSpectrum& spec, 
	     const Point &xyz, const CharColor& c)
: S(S), D(D), R(R), S0(S0), E(E), spec(spec), xyz(xyz), c(c)
{
}

Pixel::~Pixel() {
}

void Pio(Piostream& stream, Pixel& p)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
    using SCICore::GeomSpace::Pio;
    using SCICore::Geometry::Pio;
    using DaveW::Datatypes::Pio;

    stream.begin_cheap_delim();
    Pio(stream, p.S);
    Pio(stream, p.D);
    Pio(stream, p.R);
    Pio(stream, p.S0);
    Pio(stream, p.E);
    Pio(stream, p.spec);
    Pio(stream, p.xyz);
    Pio(stream, p.c);
    stream.end_cheap_delim();
}

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/08/23 02:52:56  dmw
// Dave's Datatypes
//
// Revision 1.2  1999/05/03 04:52:01  dmw
// Added and updated DaveW Datatypes/Modules
//
//
