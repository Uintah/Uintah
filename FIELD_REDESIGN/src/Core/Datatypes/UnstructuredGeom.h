// UnstructuredGeom.h - Geometries that live in an unstructured
// space. (Mesh, surface, etc.)
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_UnstructuredGeom_h
#define SCI_project_UnstructuredGeom_h 1

#include <SCICore/Datatypes/Geom.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

#include <vector>
#include <string>


namespace SCICore {
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class Tetrahedral{
public:
  Point points[4];
private:

};

class Node{
public:
  Point p;
private:
};

class SCICORESHARE UnstructuredGeom:public Geom{  
public:
  
  virtual ~UnstructuredGeom(){ };
  

protected:
  
};


} // end namespace Datatypes
} // end namespace SCICore
  

#endif
