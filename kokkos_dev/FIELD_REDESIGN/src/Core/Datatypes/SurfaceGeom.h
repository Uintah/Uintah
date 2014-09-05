//  SurfaceGeom.h - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_SurfaceGeom_h
#define SCI_project_SurfaceGeom_h 1

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Geom/GeomPolyline.h>
#include <SCICore/Datatypes/ContourGeom.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Util/DebugStream.h>
//#include <sstream>
#include <list>
#include <vector>
#include <string>
//#include <set>


namespace SCICore {
namespace Datatypes {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;
using SCICore::GeomSpace::GeomTrianglesP;
using SCICore::GeomSpace::GeomPolyline;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Math::Interpolate;
using SCICore::Util::DebugStream;

class SurfaceGeom : public ContourGeom
{
public:

  SurfaceGeom();
  virtual ~SurfaceGeom() {}
  
  virtual string getInfo();
  
  ///////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  
  vector<list<int> > d_face;

private:
  static DebugStream dbg;
};

} // end Datatypes
} // end SCICore


#endif
