//  LatticeGeom.h - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_Lattice3Geom_h
#define SCI_project_Lattice3Geom_h 1

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
  
class Lattice3Geom:public StructuredGeom
{
public:

  Lattice3Geom(int, int, int, const Point&, const Point&);
  Lattice3Geom();
  ~Lattice3Geom();

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
  virtual Point get_point(int, int, int);
  
  inline virtual int get_nx() {return nx;};
  inline virtual int get_ny() {return ny;};
  inline virtual int get_nz() {return nz;};
  virtual void resize(int ix, int iy, int iz);

  /////////
  // Return the indexes of the node defining the cell containing p which
  // is closest to the orgin
  virtual bool locate(const Point& p, int&, int&, int&);

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  
protected:
  // number of grid lines in each axis
  int nx, ny, nz;
  // Distance between grid lines
  double dx, dy, dz;
  void compute_deltas();
};

template <class A>
int Lattice3Geom::slinterpolate(A* att, elem_t elem_type, const Point& p, double& outval,
				    double eps){
  if(elem_type == NODE){
    Vector pn=p-bbox.min();
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();
    double x=pn.x()*(nx-1)/dx;
    double y=pn.y()*(ny-1)/dy;
    double z=pn.z()*(nz-1)/dz;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    int ix1=ix+1;
    int iy1=iy+1;
    int iz1=iz+1;
    if (ix<0) { if (x < -eps) return 0; else ix=0; }
    if (iy<0) { if (y < -eps) return 0; else iy=0; }
    if (iz<0) { if (z < -eps) return 0; else iz=0; }
    if (ix1>=nx) { if (x>nx-1+eps) return 0; else ix1=ix; }
    if (iy1>=ny) { if (y>ny-1+eps) return 0; else iy1=iy; }
    if (iz1>=nz) { if (z>nz-1+eps) return 0; else iz1=iz; }
    double fx =x-ix;
    double fy =y-iy;
    double fz =z-iz;
    double x00=Interpolate(att->get3(ix, iy, iz),
			   att->get3(ix1, iy, iz), fx);
    double x01=Interpolate(att->get3(ix, iy, iz1),
			   att->get3(ix1, iy, iz1), fx);
    double x10=Interpolate(att->get3(ix, iy1, iz),
			   att->get3(ix1, iy1, iz), fx);
    double x11=Interpolate(att->get3(ix, iy1, iz1),
			   att->get3(ix1, iy1, iz1), fx);
    double y0 =Interpolate(x00, x10, fy);
    double y1 =Interpolate(x01, x11, fy);
    outval=Interpolate(y0, y1, fz);
  }
  return 1;
}

} // end Datatypes
} // end SCICore


#endif
