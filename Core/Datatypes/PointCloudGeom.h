//  PointCloudGeom.h - A group of points in 3 space
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_PointCloudGeom_h
#define SCI_project_PointCloudGeom_h 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomPolyline.h>
#include <Core/Datatypes/UnstructuredGeom.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
//#include <sstream>
#include <vector>
#include <string>
//#include <set>


namespace SCIRun {

using std::vector;
using std::string;

class PointCloudGeom : public UnstructuredGeom
{
public:

  PointCloudGeom();
  PointCloudGeom(const vector<NodeSimp>&);
  ~PointCloudGeom();

  //////////
  // Set nodes and tets vectors.
  // Deletes these pointers if they are already set.
  void setNodes(const vector<NodeSimp>&);

  ///////////
  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  virtual string getInfo();


protected:

  virtual bool computeBoundingBox();

  vector<NodeSimp> d_node;

private:
  static DebugStream dbg;
};


} // End namespace SCIRun


#endif
