//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   May 2000
//
//  Copyright (C) 2000 SCI Institute
//
//
//
// To add an Interface, a new class should be created in this file,
// and the appropriate query pure virtual should be added in the Field class.
//


#ifndef Datatypes_FieldInterface_h
#define Datatypes_FieldInterface_h

#include <Core/Geometry/Point.h>

namespace SCIRun {

class InterpolateToScalar {
public:
  virtual bool interpolate(const Point& p, double& value) = 0;
  
  // TODO: Memory management of queried interfaces
};

} // end namespace SCIRun


#endif // Datatypes_FieldInterface_h


