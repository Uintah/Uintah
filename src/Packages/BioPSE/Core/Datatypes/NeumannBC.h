// Attrib.h - the base attribute class.
//
//  Written by:
//   Eric Kuehne, Alexei Samsonov
//   Department of Computer Science
//   University of Utah
//   April 2000, December 2000
//
//  Copyright (C) 2000 SCI Institute
//
//  General storage class for Fields.
//

#ifndef SCI_project_NeumannBC_h
#define SCI_project_NeumannBC_h 1

#include <vector>
#include <string>
#include <iostream>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Vector.h>

namespace BioPSE {

using namespace SCIRun;

/////////
// Structure to hold Neumann BC related values
class NeumannBC {
public:  
  // GROUP: public data
  //////////
  // 
  NeumannBC(){};
  NeumannBC(Vector v, double d): dir(v), val(d){};
  //////////
  // Direction to take derivative in
  Vector dir;
  //////////
  // Value of the derivative
  double val;
};
} // end namespace BioPSE


namespace SCIRun {
using namespace std;
//////////
// PIO for NeumannBC objects
void  Pio(Piostream&, BioPSE::NeumannBC&);
ostream& operator<<(ostream&, BioPSE::NeumannBC&);
} // end namespace SCIRun

#endif
