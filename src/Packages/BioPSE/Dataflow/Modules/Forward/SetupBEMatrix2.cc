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
 *  SetupBEMatrix2.cc: TODO: Describe this module.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   May, 2003
 *   
 *   Copyright (C) 2003 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TriSurfField.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <math.h>

#include <map>
#include <iostream>
#include <string>
#include <fstream>

namespace BioPSE {

using namespace SCIRun;


class SetupBEMatrix2 : public Module
{
private:
  
  bool ray_triangle_intersect(double &t,
			      const Point &p,
			      const Vector &v,
			      const Point &p0,
			      const Point &p1,
			      const Point &p2) const;
  void compute_intersections(vector<pair<double, int> > &results,
			     const TriSurfMeshHandle &mesh,
			     const Point &p, const Vector &v,
			     int marker) const;


  int compute_parent(const vector<TriSurfMeshHandle> &meshes, int index);

  bool compute_nesting(vector<int> &nesting,
		       const vector<TriSurfMeshHandle> &meshes);
  
public:
  
  //! Constructor
  SetupBEMatrix2(GuiContext *context);
  
  //! Destructor  
  virtual ~SetupBEMatrix2();

  virtual void execute();
};


DECLARE_MAKER(SetupBEMatrix2)


SetupBEMatrix2::SetupBEMatrix2(GuiContext *context): 
  Module("SetupBEMatrix2", context, Source, "Forward", "BioPSE")
{
}


SetupBEMatrix2::~SetupBEMatrix2()
{
}


// C++ized MollerTrumbore97 Ray Triangle intersection test.
#define EPSILON 1.0e-6
bool
SetupBEMatrix2::ray_triangle_intersect(double &t,
				       const Point &point,
				       const Vector &dir,
				       const Point &p0,
				       const Point &p1,
				       const Point &p2) const
{
  // Find vectors for two edges sharing p0.
  const Vector edge1 = p1 - p0;
  const Vector edge2 = p2 - p0;
  
  // begin calculating determinant - also used to calculate U parameter.
  const Vector pvec = Cross(dir, edge2);

  // if determinant is near zero, ray lies in plane of triangle.
  const double det = Dot(edge1, pvec);
  if (det > -EPSILON && det < EPSILON)
  {
    return false;
  }
  const double inv_det = 1.0 / det;

  // Calculate distance from vert0 to ray origin.
  const Vector tvec = point - p0;

  // Calculate U parameter and test bounds.
  const double u = Dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0)
  {
    return false;
  }

  // Prepare to test V parameter.
  const Vector qvec = Cross(tvec, edge1);

  // Calculate V parameter and test bounds.
  const double v = Dot(dir, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0)
  {
    return false;
  }

  // Calculate t, ray intersects triangle.
  t = Dot(edge2, qvec) * inv_det;
  
  return true;
}


void
SetupBEMatrix2::compute_intersections(vector<pair<double, int> > &results,
				      const TriSurfMeshHandle &mesh,
				      const Point &p, const Vector &v,
				      int marker) const
{
  TriSurfMesh::Face::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  double t;
  while (itr != eitr)
  {
    TriSurfMesh::Node::array_type nodes;
    mesh->get_nodes(nodes, *itr);
    Point p0, p1, p2;
    mesh->get_center(p0, nodes[0]);
    mesh->get_center(p1, nodes[1]);
    mesh->get_center(p2, nodes[2]);
    if (ray_triangle_intersect(t, p, v, p0, p1, p2))
    {
      results.push_back(pair<double, int>(t, marker));
    }
    ++itr;
  }
}

static bool
pair_less(const pair<double, int> &a,
	  const pair<double, int> &b)
{
  return a.first < b.first;
}


int
SetupBEMatrix2::compute_parent(const vector<TriSurfMeshHandle> &meshes,
			       int index)
{
  Point point;
  meshes[index]->get_center(point, TriSurfMesh::Node::index_type(0));
  Vector dir(1.0, 1.0, 1.0);
  vector<pair<double, int> > intersections;

  unsigned int i;
  for (i = 0; i < meshes.size(); i++)
  {
    compute_intersections(intersections, meshes[i], point, dir, i);
  }

  std::sort(intersections.begin(), intersections.end(), pair_less);

  vector<int> counts(meshes.size(), 0);
  for (i = 0; i < intersections.size(); i++)
  {
    if (intersections[i].second == index)
    {
      // First odd count is parent.
      for (int j = i-1; j >= 0; j--)
      {
	if (counts[intersections[j].second] & 1)
	{
	  return intersections[j].second;
	}
      }
      // No odd parent, is outside.
      return meshes.size();
    }
    counts[intersections[i].second]++;
  }

  // Indeterminant, we should intersect with ourselves.
  return meshes.size();
}



bool
SetupBEMatrix2::compute_nesting(vector<int> &nesting,
				const vector<TriSurfMeshHandle> &meshes)
{
  nesting.resize(meshes.size());

  int i;
  for (i = 0; i < meshes.size(); i++)
  {
    nesting[i] = compute_parent(meshes, i);
  }

  return true;
}



void
SetupBEMatrix2::execute()
{
  port_range_type range = get_iports("Surface");
  if (range.first == range.second)
  {
    remark("No surfaces connected.");
    return;
  }

  // Gather up the surfaces from the input ports.
  vector<FieldHandle> fields;
  vector<TriSurfMeshHandle> meshes;
  port_map_type::iterator pi = range.first;
  while (pi != range.second)
  {
    FieldIPort *fip = (FieldIPort *)get_iport(pi->second);
    if (!fip)
    {
      error("Unable to initialize iport '" + to_string(pi->second) + "'.");
      return;
    }
    FieldHandle field;
    if (fip->get(field))
    {
      if (field.get_rep() == 0)
      {
	error("Surface port '" + to_string(pi->second) +
	      "' contained no data.");
	return;
      }
      TriSurfMesh *mesh = 0;
      if (!(mesh = dynamic_cast<TriSurfMesh *>(field->mesh().get_rep())))
      {
	error("Surface port '" + to_string(pi->second) +
	      "' does not contain a TriSurfField");
	return;
      }
      fields.push_back(field);
      meshes.push_back(mesh);
    }
    ++pi;
  }

  // Compute the nesting tree for the input meshes.
  vector<int> nesting;
  if (!compute_nesting(nesting, meshes))
  {
    error("Unable to compute a valid nesting for this set of surfaces.");
  }

  // Debugging code, print out the tree.
  int i;
  for (i=0; i < nesting.size(); i++)
  {
    cout << "parent of " << i << " is " << nesting[i] << "\n";
  }
}

} // end namespace BioPSE
