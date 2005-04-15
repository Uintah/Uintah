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
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>

#include <vector>
#include <string>

//#define BOUNDARY -2

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
//using SCICore::GeomSpace::GeomGroup;
//using SCICore::GeomSpace::GeomMaterial;
//using SCICore::GeomSpace::MaterialHandle;
//using SCICore::GeomSpace::GeomTrianglesP;
using namespace SCICore::GeomSpace;
using namespace SCICore::Malloc;


class NodeSimp {
public:
  NodeSimp();
  NodeSimp(const Point&);
  ~NodeSimp();

  void draw(double radius, const MaterialHandle& matl, GeomGroup
	    *group);
  
  
  Point p;
private:
};


class EdgeSimp {
public:
  EdgeSimp();
  //////////
  // Construct an edge consisting of the two points
  EdgeSimp(int, int);
  ~EdgeSimp();

  //////////
  // Compare two edges.
  // Warning: Assumes that the two edges live on the same mesh (use
  // the same node list)
  bool operator==(const EdgeSimp&) const;
  bool operator<(const EdgeSimp&) const;
  
  int nodes[2];
private:
};


class FaceSimp {
public:
  FaceSimp();
  FaceSimp(int, int, int);
  ~FaceSimp();
  
  int nodes[3];
  int neighbors[3];
private:
};

  
class TetSimp {
public:
  TetSimp();
  TetSimp(int, int, int, int);
  ~TetSimp();

  bool draw(const vector<NodeSimp>&, GeomTrianglesP* group);
  int nodes[4];
  int neighbors[4];
private:

};



class SCICORESHARE UnstructuredGeom : public Geom
{
public:
  
  virtual ~UnstructuredGeom() {};

protected:
};


} // end namespace Datatypes
} // end namespace SCICore
  

#endif
