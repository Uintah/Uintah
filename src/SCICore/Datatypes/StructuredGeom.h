// StructuredGeom.h - Geometries that live in a structured space
// (lattice, curvelinear, etc.)
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_StructuredGeom_h
#define SCI_project_StructuredGeom_h 1

#include <SCICore/Datatypes/Geom.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

#include <vector>
#include <string>


namespace SCICore {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;


class SCICORESHARE StructuredGeom : public Geom {
public:
  
  virtual ~StructuredGeom() {}

protected:
};


} // end namespace Datatypes
} // end namespace SCICore
  

#endif
