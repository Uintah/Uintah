// FieldInterface.h -  This file holds all of the interfaces available for fields
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
// and the appropriate lines of code should be added to the querry_interface
// function in the Field class (or a class derived from Field).
//
// Per convention, each interface should contain at least one function
// that bears the same name as the class, but in all lower case.
//
// Eric Kuehne

#ifndef SCI_project_FieldInterface_h
#define SCI_project_FieldInterface_h 1

#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

namespace SCICore{
namespace Datatypes{

using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
//////////
// The base classes for all field interfaces
class SCICORESHARE FieldInterface{
public:
};

class SLInterpolate:public FieldInterface{
public:
  virtual int slinterpolate(const Point&, double&, double eps=1.e-6) = 0;
};

class VLInterpolate:public FieldInterface{
public:
  virtual int vlinterpolate(const Point&, Vector&) = 0;
};

class Gradient:public FieldInterface{
public:
  virtual Vector gradient(const Point&) = 0;
};

} // end namespace Datatypes
} // end namespace SCICore


#endif


