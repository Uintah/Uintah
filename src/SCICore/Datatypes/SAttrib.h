//  SAttrib.h - Scalar Attribute
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_SAttrib_h
#define SCI_project_SAttrib_h 1

#include <SCICore/Datatypes/Attrib.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore{
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class SAttrib:public Attrib //abstract class
{
public:
  SAttrib();
  SAttrib(const SAttrib& copy);
  SAttrib(int, int, int);
  SAttrib(int, int);
  SAttrib(int);
  
  virtual ~SAttrib() { };

  /////////
  // Return the min and max data values;
  virtual bool get_minmax(double&, double&);
  virtual bool compute_minmax() = 0;
  
protected:

  double min;
  double max;
  bool has_minmax;
};



}  // end Datatypes
}  // end SCICore

#endif



