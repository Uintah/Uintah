// StructuredGeom.h - Geometries that live in a structured space
// (lattice, curvelinear, etc.)
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_StructuredGeom_h
#define SCI_project_StructuredGeom_h 1

#include <Core/Datatypes/Geom.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <vector>
#include <string>


namespace SCIRun {

using std::vector;
using std::string;


class SCICORESHARE StructuredGeom : public Geom {
public:
  
  virtual ~StructuredGeom() {}

protected:
};


} // End namespace SCIRun
  

#endif
