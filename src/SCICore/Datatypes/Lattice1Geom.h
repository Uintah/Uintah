//  LatticeGeom.h - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 1000
//
//  Copyright (C) 1000 SCI Institute


#ifndef SCI_project_Lattice1Geom_h
#define SCI_project_Lattice1Geom_h 1

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Datatypes/StructuredGeom.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Math/MiscMath.h>

#include <sstream>
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
using SCICore::Math::Interpolate;
  
class Lattice1Geom:public StructuredGeom
{
public:

  Lattice1Geom(int, const Point&, const Point&);
  Lattice1Geom();
  ~Lattice1Geom();

  virtual string get_info();

  //////////
  // Compute the bounding box and diagnal, set has_bbox to true
  virtual bool compute_bbox();


  //////////
  // Set the bounding box
  virtual void set_bbox(const BBox&);
  virtual void set_bbox(const Point&, const Point&);
  
  //////////
  // Interpolate
  template <class A>
  int slinterpolate(A* att, elem_t, const Point& p, double& outval,
		    double eps=1.0e-6);
  
  //////////
  // Return the point relative to the min in the bounding box
  virtual Point get_point(int);
  
  inline virtual int get_nx() {return nx;};
  virtual void resize(int ix);

  /////////
  // Return the indexes of the node defining the cell containing p which
  // is closest to the orgin
  virtual bool locate(const Point& p, int&);

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  
protected:
  // number of grid lines in each axis
  int nx;
  // Distance between grid lines
  double dx;
  ////////////
  // Compute the distance between grid lines in each dimension
  void compute_deltas();
};

template <class A>
int Lattice1Geom::slinterpolate(A* att, elem_t elem_type, const Point& p, double& outval,
				    double eps){
  return 1;
}

} // end Datatypes
} // end SCICore


#endif
