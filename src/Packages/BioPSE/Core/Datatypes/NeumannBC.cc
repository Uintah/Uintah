// NeumannBC.cc - Neuamann boundary condition
//
//  Written by:
//   Alexei Samsonov
//   Department of Computer Science
//   University of Utah
//   April 2000, December 2000
//
//  Copyright (C) 2000 SCI Institute

#include <Packages/BioPSE/Core/Datatypes/NeumannBC.h>

namespace SCIRun {

using std::cout;
using std::endl;

//////////
// PIO for NeumannBC objects
void Pio(Piostream& stream, BioPSE::NeumannBC& nmn){
  stream.begin_cheap_delim();
  Pio(stream, nmn.dir);
  Pio(stream, nmn.val);
  stream.end_cheap_delim();
}
//////////
//
ostream& operator<<(ostream& ostr, BioPSE::NeumannBC& nmn){
  ostr << "["<< nmn.dir << ", " << nmn.val << "]";
  return ostr;
}
}  // end namespace SCIRun
