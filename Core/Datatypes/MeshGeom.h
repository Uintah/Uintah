//  MeshGeom.h - A base class for regular geometries with alligned axes
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_MeshGeom_h
#define SCI_project_MeshGeom_h 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomPolyline.h>
#include <Core/Datatypes/SurfaceGeom.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
//#include <sstream>
#include <list>
#include <vector>
#include <string>
//#include <set>


namespace SCIRun {

using std::vector;
using std::string;

class MeshGeom:public SurfaceGeom
{
public:

  MeshGeom();
  virtual ~MeshGeom() {}
  
  virtual string getInfo();
  
  ///////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
 
protected:

  vector<list<int> > d_cell;

private:
  static DebugStream dbg;
};

} // End namespace SCIRun


#endif
