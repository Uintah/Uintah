// UnstructuredGeom.h - Geometries that live in an unstructured
// space. (Mesh, surface, etc.)
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_UnstructuredGeom_h
#define SCI_project_UnstructuredGeom_h 1

#include <Core/Datatypes/Geom.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/Material.h>
#include <Core/Malloc/Allocator.h>

#include <vector>
#include <string>

//#define BOUNDARY -2

namespace SCIRun {

using std::vector;
using std::string;
//using GeomGroup;
//using GeomMaterial;
//using MaterialHandle;
//using GeomTrianglesP;


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


} // End namespace SCIRun
  

#endif
