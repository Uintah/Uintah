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

#include <vector>
#include <string>

namespace SCICore{
namespace Datatypes{

using std::vector;
using std::string;
using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class SAttrib:public Attrib //abstract class
{
public:
  virtual ~SAttrib() { };
protected:
  
};



}  // end Datatypes
}  // end SCICore

#endif



