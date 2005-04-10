/* REFERENCED */
static char *id="@(#) $Id$";

#include "Stream.h"

using namespace Uintah::ArchesSpace;

Stream::Stream()
{
}

Stream::Stream(const Stream& strm) // copy constructor
{
  d_speciesConcn = strm.d_speciesConcn;
  d_pressure = strm.d_pressure;
  d_density = strm.d_density;
  d_temperature = strm.d_temperature;
  d_enthalpy = strm.d_enthalpy;
  d_mole = strm.d_mole;
}

Stream::~Stream()
{
}

Stream& Stream::linInterpolate(double upfactor, double lowfactor,
			       Stream& rightvalue) {
  d_pressure = upfactor*d_pressure+lowfactor*rightvalue.d_pressure;
  d_density = upfactor*d_density+lowfactor*rightvalue.d_density;
  d_temperature = upfactor*d_temperature+lowfactor*rightvalue.d_temperature;
  d_enthalpy = upfactor*d_enthalpy+lowfactor*rightvalue.d_enthalpy;
  for (int i = 0; i < d_speciesConcn.size(); i++)
    d_speciesConcn[i] = upfactor*d_speciesConcn[i] +
                   lowFactor*rightvalue.d_speciesConcn[i];
  return *this;
}
void
Stream::print(std::ostream& out) const {
  out << "Integrated values"<< '\n';
  out << "Density: "<< d_density << '/n';
  out << "Pressure: "<< d_pressure << '/n';
  out << "Temperature: "<< d_temperature << '/n';
  out << "Enthalpy: "<< d_enthalpy << '/n';
  out << "Species concentration in moles: " << '/n'
    for (int ii = 0; ii < d_speciesConcn.size(); ii++) {
      out.width(10);
      out << d_speciesConcn[ii] << " " ; 
      if (!(ii % 10)) out << endl; 
    }
}
//
// $Log$
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
//
