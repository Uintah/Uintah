
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

#include <Packages/DaveW/Core/Datatypes/CS684/Pixel.h>

namespace DaveW {
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
} // End namespace DaveW

namespace SCIRun {
using namespace DaveW;

void Pio(Piostream& stream, Pixel& p)
{
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

} // End namespace SCIRun
