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
 *  GenDistanceField.cc:  Unfinished modules
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/Butson/Dataflow/Modules/Modeling/GenDistanceField.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Algorithms/Visualization/mcube2.h>
#include <iostream>


namespace SCIRun {

class GenDistanceField : public Module
{
private:

public:
  GenDistanceField(GuiContext *context);
  virtual ~GenDistanceField();

  virtual void execute();

private:
  TriSurfMeshHandle extract(LatVolMeshHandle mesh,
			    Handle<GenDistanceFieldAlgo> algo,
			    FieldHandle skel);
};


DECLARE_MAKER(GenDistanceField)


GenDistanceField::GenDistanceField(GuiContext *context)
  : Module("GenDistanceField", context, Filter, "Modeling", "Butson")
{
}



GenDistanceField::~GenDistanceField()
{
}


static
TriSurfMesh::Node::index_type
find_or_add_edgepoint(LatVolMesh::Node::index_type n0,
		      LatVolMesh::Node::index_type n1,
		      const Point &p,
		      TriSurfMeshHandle trisurf,
		      map<long int, TriSurfMesh::Node::index_type> &vertex_map,
		      int nx, int ny)
{
  map<long int, TriSurfMesh::Node::index_type>::iterator node_iter;
  long int key0 = 
    (long int) n0.i_ + nx * ((long int) n0.j_ + (ny * (long int) n0.k_));
  long int key1 = 
    (long int) n1.i_ + nx * ((long int) n1.j_ + (ny * (long int) n1.k_));
  long int small_key;
  int dir;
  if (n0.k_ != n1.k_) dir=2;
  else if (n0.j_ != n1.j_) dir=1;
  else dir=0;
  if (n0.k_ < n1.k_) small_key = key0;
  else if (n0.k_ > n1.k_) small_key = key1;
  else if (n0.j_ < n1.j_) small_key = key0;
  else if (n0.j_ > n1.j_) small_key = key1;
  else if (n0.i_ < n1.i_) small_key = key0;
  else small_key = key1;
  long int key = (small_key * 4) + dir;
  TriSurfMesh::Node::index_type node_idx;
  node_iter = vertex_map.find(key);
  if (node_iter == vertex_map.end()) { // first time to see this node
    node_idx = trisurf->add_point(p);
    vertex_map[key] = node_idx;
  } else {
    node_idx = (*node_iter).second;
  }
  return node_idx;
}


TriSurfMeshHandle
GenDistanceField::extract(LatVolMeshHandle mesh,
			  Handle<GenDistanceFieldAlgo> algo,
			  FieldHandle skel)
{
  const double iso = 1.0;

  TriSurfMeshHandle trisurf = scinew TriSurfMesh();
  map<long int, TriSurfMesh::Node::index_type> vertex_map;
  const int nx = mesh->get_ni();
  const int ny = mesh->get_ni();

  LatVolMesh::Node::array_type node(8);
  Point p[8];
  double value[8];

  LatVolMesh::Cell::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  for (; itr != eitr; ++itr)
  {
    mesh->get_nodes(node, *itr);

    int code = 0;
    for (int i=7; i>=0; i--) {
      mesh->get_point( p[i], node[i] );
      value[i] = algo->get_dist(p[i], skel);
      code = code*2+(value[i] < iso );
    }

    if ( code == 0 || code == 255 )
      continue;

    //  TriangleCase *tcase=&tri_case[code];
    TRIANGLE_CASES *tcase=&triCases[code];
    int *vertex = tcase->edges;
  
    Point q[12];
    TriSurfMesh::Node::index_type surf_node[12];

    // interpolate and project vertices
    int v = 0;
    vector<bool> visited(12, false);
    while (vertex[v] != -1)
    {
      const int i = vertex[v++];
      if (visited[i]) continue;
      visited[i]=true;
      const int v1 = edge_tab[i][0];
      const int v2 = edge_tab[i][1];
      q[i] = Interpolate(p[v1], p[v2], 
			 (value[v1]-iso)/double(value[v1]-value[v2]));
      surf_node[i] = find_or_add_edgepoint(node[v1], node[v2], q[i],
					   trisurf, vertex_map, nx, ny);
    }    
  
    v = 0;
    while (vertex[v] != -1)
    {
      const int v0 = vertex[v++];
      const int v1 = vertex[v++];
      const int v2 = vertex[v++];
      trisurf->add_triangle(surf_node[v0], surf_node[v1], surf_node[v2]);
    }
  }

  return trisurf;
}
  



void
GenDistanceField::execute()
{
  // Read in the LatVolField<double>, clone it.
  FieldIPort *dst_port = (FieldIPort *)get_iport("Destination");
  if (!dst_port) {
    error("Unable to initialize iport 'Destination'.");
    return;
  }
  FieldHandle dfieldhandle;
  if (!(dst_port->get(dfieldhandle) && dfieldhandle.get_rep()))
  {
    return;
  }
#if 0
  if (!dfieldhandle->is_scalar())
  {
    error("Destination field must be of scalar type, preferably doubles.");
    return;
  }
#endif

  //bool did_once_p = false;
  port_range_type range = get_iports("Skeleton");
  port_map_type::iterator pi = range.first;
  while (pi != range.second)
  {
    FieldIPort *port = (FieldIPort *)get_iport(pi->second);
    if (!port) {
      error("Unable to initialize iport '" + to_string(pi->second) + "'.");
      return;
    }
    ++pi;

    // Do something with port.
    FieldHandle skel_handle;
    if (port->get(skel_handle) && skel_handle.get_rep())
    {
      if (!skel_handle->is_scalar())
      {
	warning("Skeleton models must be scalar fields, skipping.");
	continue;
      }
      CompileInfoHandle ci = GenDistanceFieldAlgo::
	get_compile_info(skel_handle->get_type_description());
      Handle<GenDistanceFieldAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) return;

#if 0
      if (!did_once_p) { dfieldhandle.detach(); }
      if (skel_handle->data_at() == Field::NODE)
      {
	algo->execute_node(dfieldhandle, skel_handle, did_once_p);
      }
      else if (skel_handle->data_at() == Field::EDGE)
      {
	algo->execute_edge(dfieldhandle, skel_handle, did_once_p);
      }
      else
      {
	// TODO:  Could make execute_face and execute_cell functions.
	error("Skeleton data location must be at nodes or edges!");
	return;
      }
      
      did_once_p = true;
#endif

      LatVolMeshHandle latmesh =
	dynamic_cast<LatVolMesh *>(dfieldhandle->mesh().get_rep());
      TriSurfMeshHandle trisurf = extract(latmesh, algo, skel_handle);

      FieldHandle ofield = scinew TriSurfField<double>(trisurf, Field::NODE);

      // Forward the lattice.
      FieldOPort *ofp = (FieldOPort *)get_oport("Distances");
      if (!ofp) {
	error("Unable to initialize " + name + "'s output port.");
	return;
      }
      ofp->send(ofield);
    }
  }
#if 0  
  if (did_once_p)
  {
    // Forward the lattice.
    FieldOPort *ofp = (FieldOPort *)get_oport("Distances");
    if (!ofp) {
      error("Unable to initialize " + name + "'s output port.");
      return;
    }
    ofp->send(dfieldhandle);
  }  
#endif  
}


double
GenDistanceFieldAlgo::distance_to_line2(const Point &p,
					const Point &a, const Point &b) const
{
  Vector m = b - a;
  Vector n = p - a;
  if (m.length2() < 1e-6)
  {
    return n.length2();
  }
  else
  {
    const double t0 = Dot(m, n) / Dot(m, m);
    if (t0 <= 0) return (n).length2();
    else if (t0 >= 1.0) return (p - b).length2();
    else return (n - m * t0).length2();
  }
}


CompileInfoHandle
GenDistanceFieldAlgo::get_compile_info(const TypeDescription *fsrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GenDistanceFieldAlgoT");
  static const string base_class_name("GenDistanceFieldAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name());
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun

