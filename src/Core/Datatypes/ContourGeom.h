//  ContourGeom.h - A base class for regular geometries with alligned axes
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_ContourGeom_h
#define SCI_project_ContourGeom_h 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomPolyline.h>
#include <Core/Datatypes/PointCloudGeom.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <sstream>  // for std::ostringstream on linux
#include <vector>
#include <string>
//#include <set>


namespace SCIRun {

using std::vector;
using std::string;
using std::ostringstream;

class ContourGeom;
typedef LockingHandle<ContourGeom> ContourGeomHandle;

class ContourGeom : public PointCloudGeom
{
public:

  ContourGeom();
  ContourGeom(const vector<NodeSimp>&, const vector<EdgeSimp>&);
  ~ContourGeom();

  virtual string getInfo();
  virtual string getTypeName(int=0);
  
  //////////
  // Interpolate
  template <class A>
  int slinterpolate(A* att, elem_t, const Point& p, double& outval,
		    double eps=1.0e-6);

  //////////
  // Deletes these pointers if they are already set.
  void setEdges(const vector<EdgeSimp>&);

  ///////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static string typeName(int);
  vector<EdgeSimp> edge_;

protected:

private:
  static DebugStream dbg;
};

template <class A>
int ContourGeom::slinterpolate(A* att, elem_t elem_type, const Point& p,
			       double& outval, double eps)
{
}

} // End namespace SCIRun


#endif
