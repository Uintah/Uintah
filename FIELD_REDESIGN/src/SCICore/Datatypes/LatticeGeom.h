//  LatticeGeom.h - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_LatticeGeom_h
#define SCI_project_LatticeGeom_h 1

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Datatypes/StructuredGeom.h>
#include <SCICore/Containers/LockingHandle.h>

#include <vector>
#include <string>

namespace SCICore{
namespace Datatypes{

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

  
class LatticeGeom:public StructuredGeom
{
public:

  virtual ~LatticeGeom() { };

  virtual string get_info() = 0;

protected:

};


} // end Datatypes
} // end SCICore


#endif
