/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  PointCloudMesh.cc: PointCloud mesh
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Vector.h>
#include <float.h>  // for DBL_MAX
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID PointCloudMesh::type_id("PointCloudMesh", "MeshBase", maker);

BBox 
PointCloudMesh::get_bounding_box() const
{
  BBox result;

  for (node_iterator i = node_begin();
       i!=node_end();
       ++i) 
    result.extend(points_[*i]);

  return result;
}

bool
PointCloudMesh::locate(node_index &idx, const Point &p) const
{
  node_iterator ni = node_begin();
  idx = *node_begin();

  if (ni==node_end())
    return false;

  double closest = (p-points_[*ni]).length2();

  ++ni;
  for (; ni != node_end(); ++ni) {
    if ( (p-points_[*ni]).length2() < closest ) {
      closest = (p-points_[*ni]).length2();
      idx = *ni;
    }
  }

  return true;
}

PointCloudMesh::node_index
PointCloudMesh::add_point(const Point &p)
{
  points_.push_back(p);
  return points_.size() - 1;
}

#define PointCloudMESH_VERSION 1

void
PointCloudMesh::io(Piostream& stream)
{
  stream.begin_class(type_name().c_str(), PointCloudMESH_VERSION);

  MeshBase::io(stream);

  // IO data members, in order
  Pio(stream,points_);

  stream.end_class();
}

const string 
PointCloudMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "PointCloudMesh";
  return name;
}


} // namespace SCIRun
