/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


//    File   : StreamLines.h
//    Author : Michael Callahan
//    Date   : July 2001

#if !defined(_STREAMLINES_H_)
#define _STREAMLINES_H_

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Geometry/CompGeom.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/FieldInterface.h>
#include <algorithm>
#include <sstream>

namespace SCIRun {
typedef CurveMesh<CrvLinearLgn<Point> > CMesh;
typedef CrvLinearLgn<double>            DatBasis;
typedef GenericField<CMesh, DatBasis, vector<double> > CField;
 
using namespace std;

typedef struct _SLData {
  FieldHandle fh;
  Mutex lock;
  FieldHandle seed_field_h;
  VectorFieldInterfaceHandle vfi;
  double tolerance;
  double stepsize;
  int maxsteps;
  int direction;
  int value;
  bool rcp;
  int met;
  int np;

  _SLData() : lock("StreamLines Lock") {}
} SLData;


vector<Point>::iterator
StreamLinesCleanupPoints(vector<Point> &input, double e2);

class StreamLinesAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle seed_field_h,
			      VectorFieldInterfaceHandle vfi,
			      double tolerance,
			      double stepsize,
			      int maxsteps,
			      int direction,
			      int value,
			      bool remove_colinear_p,
			      int method, 
			      int np) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *dsrc,
					    const TypeDescription *sloc,
					    int value);
protected:

  //! This particular implementation uses Runge-Kutta-Fehlberg.
  void FindNodes(vector<Point>&, Point, double, double, int, 
		 const VectorFieldInterfaceHandle &,
		 bool remove_colinear_p, int method);
};


template <class SFLD, class STYPE, class SLOC>
class StreamLinesAlgoT : public StreamLinesAlgo
{
public:
  //! virtual interface. 
  void parallel_generate(int proc, SLData *d);

  virtual FieldHandle execute(FieldHandle seed_field_h,
			      VectorFieldInterfaceHandle vfi,
			      double tolerance,
			      double stepsize,
			      int maxsteps,
			      int direction,
			      int value,
			      bool remove_colinear_p,
			      int method,
			      int np);
};


template <class SFIELD, class STYPE, class SLOC>
void
StreamLinesAlgoT<SFIELD, STYPE, SLOC>::
parallel_generate( int proc, SLData *d)
{
  SFIELD *sfield = (SFIELD *) d->seed_field_h.get_rep();
  typename SFIELD::mesh_handle_type smesh = sfield->get_typed_mesh();


  typedef CrvLinearLgn<STYPE> DatBasisL;
  typedef GenericField<CMesh, DatBasisL, vector<STYPE> > CFieldL;

  CFieldL *cfield = (CFieldL *) d->fh.get_rep();

  const double tolerance2 = d->tolerance * d->tolerance;

  Point seed;
  Vector test;
  vector<Point> nodes;
  nodes.reserve(d->maxsteps);

  vector<Point>::iterator node_iter;
  CMesh::Node::index_type n1, n2;

  // Try to find the streamline for each seed point.
  typename SLOC::iterator seed_iter, seed_iter_end;
  smesh->begin(seed_iter);
  smesh->end(seed_iter_end);

  typename SFIELD::fdata_type::iterator data_iter = sfield->fdata().begin();

  int count = 0;

  while (seed_iter != seed_iter_end)
  {
    // If this seed doesn't "belong" to this parallel thread,
    // ignore it and continue on the next seed.
    if (count%d->np != proc) {
      ++seed_iter;
      ++data_iter;
      ++count;
      continue;
    }

    smesh->get_point(seed, *seed_iter);

    // Is the seed point inside the field?
    if (!d->vfi->interpolate(test, seed))
    {
      ++seed_iter;
      ++data_iter;
      ++count;
      continue;
    }

    nodes.clear();
    nodes.push_back(seed);

    int cc = 0;

    // Find the negative streamlines.
    if( d->direction <= 1 )
    {
      FindNodes(nodes, seed, tolerance2, -d->stepsize, d->maxsteps,
		d->vfi, d->rcp, d->met);
      if ( d->direction == 1 )
      {
	reverse(nodes.begin(), nodes.end());
	cc = nodes.size();
	cc = -(cc - 1);
      }
    }
    // Append the positive streamlines.
    if( d->direction >= 1 ){
      FindNodes(nodes, seed, tolerance2, d->stepsize, d->maxsteps,
		d->vfi, d->rcp, d->met);
    }

    double length = 0;

    Point p1;

    if( d->value == 4 ) {
      node_iter = nodes.begin();
      if (node_iter != nodes.end()) {
	p1 = *node_iter;	
	++node_iter;

	while (node_iter != nodes.end()) {
	  length += Vector( *node_iter-p1 ).length();
	  p1 = *node_iter;
	  ++node_iter;
	}
      }
    }

    node_iter = nodes.begin();

    if (node_iter != nodes.end()) {
      d->lock.lock();
      n1 = cfield->get_typed_mesh()->add_node(*node_iter);
      p1 = *node_iter;

      std::ostringstream str;
      str << "Streamline " << count << " Node Index";      
      d->fh->set_property( str.str(), n1, false );

      cfield->resize_fdata();

      if( d->value == 0 )
	cfield->set_value((*data_iter), n1);
      else if( d->value == 1 )
	cfield->set_value((double)(*seed_iter), n1);
      else if( d->value == 2)
	cfield->set_value((double)abs(cc), n1);
      else if( d->value == 3)
	cfield->set_value( length, n1);
      else if( d->value == 4)
	cfield->set_value( length, n1);

      ++node_iter;

      cc++;

      while (node_iter != nodes.end()) {
	n2 = cfield->get_typed_mesh()->add_node(*node_iter);
	cfield->resize_fdata();

	if( d->value == 0 )
	  cfield->set_value((*data_iter), n2);
	else if( d->value == 1 )
	  cfield->set_value((double)(*seed_iter), n2);
	else if( d->value == 2)
	  cfield->set_value((double)abs(cc), n2);
	else if( d->value == 3) {
	  length += Vector( *node_iter-p1 ).length();
	  cfield->set_value( length, n2);
	  p1 = *node_iter;
	} else if( d->value == 4)
	  cfield->set_value( length, n2);

	cfield->get_typed_mesh()->add_edge(n1, n2);

	n1 = n2;
	++node_iter;

	cc++;
      }
      d->lock.unlock();
    }

    ++seed_iter;
    ++data_iter;
    ++count;
  }

  d->fh->set_property( "Streamline Count", count, false );
}


template <class SFIELD, class STYPE, class SLOC>
FieldHandle
StreamLinesAlgoT<SFIELD, STYPE, SLOC>::
execute(FieldHandle seed_field_h,
	VectorFieldInterfaceHandle vfi,
	double tolerance,
	double stepsize,
	int maxsteps,
	int direction,
	int value,
	bool rcp,
	int met,
	int np)
{
  SLData d;
  d.seed_field_h=seed_field_h;
  d.vfi=vfi;
  d.tolerance=tolerance;
  d.stepsize=stepsize;
  d.maxsteps=maxsteps;
  d.direction=direction;
  d.value=value;
  d.rcp=rcp;
  d.met=met;
  d.np=np;

  CMesh::handle_type cmesh = scinew CMesh();
  CField *cf = scinew CField(cmesh);
  
  d.fh = FieldHandle(cf);

  Thread::parallel(this,
                   &StreamLinesAlgoT<SFIELD, STYPE, SLOC>::parallel_generate,
                   np, &d);

  cf->freeze();

  return cf;

#if 0
  CMesh::Node::size_type count;
  cf->get_typed_mesh()->size(count);
  if (((unsigned int)count) == 0)
  {
    delete cf;
    return 0;
  }
  else
  {
    return cf;
  }
#endif
}


class StreamLinesAccAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle seed_field_h,
			      FieldHandle vfield_h,
			      int maxsteps,
			      int direction,
			      int value,
			      bool remove_colinear_p) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *dsrc,
					    const TypeDescription *sloc,
					    const TypeDescription *vfld,
					    int value);

};


template <class SFIELD, class STYPE, class SLOC, class VFLD>
class StreamLinesAccAlgoT : public StreamLinesAccAlgo
{
public:

  virtual FieldHandle execute(FieldHandle seed_field_h,
			      FieldHandle vfield_h,
			      int maxsteps,
			      int direction,
			      int value,
			      bool remove_colinear_p);

  void FindNodes(vector<Point>& nodes, Point seed, int maxsteps, 
		 VFLD *vfield, bool remove_colinear_p, bool back);
};


template <class SFIELD, class STYPE, class SLOC, class VFLD>
void
StreamLinesAccAlgoT<SFIELD, STYPE, SLOC, VFLD>::
FindNodes(vector<Point> &v,
	  Point seed,
	  int maxsteps,
	  VFLD *vfield,
	  bool remove_colinear_p,
	  bool back)
{
  typename VFLD::mesh_handle_type vmesh = vfield->get_typed_mesh();

  typename VFLD::mesh_type::Elem::index_type elem, neighbor;
  typename VFLD::mesh_type::Face::array_type faces;
  typename VFLD::mesh_type::Node::array_type nodes;
  typename VFLD::mesh_type::Face::index_type minface;
  Vector lastnormal, minnormal;
  Vector dir;

  if (!vmesh->locate(elem, seed)) { return; }
  for (int i=0; i < maxsteps; i++)
  {
    vfield->value(dir, elem);
    dir.safe_normalize();
    if (back) { dir *= -1.0; }
    
    double ddl;
    if (i && (ddl = Dot(dir, lastnormal)) < 1.0e-3)
    {
      dir = dir - lastnormal * (ddl / Dot (lastnormal, lastnormal));
      if (dir.safe_normalize() < 1.0e-3) { break; }
    }

    vmesh->get_faces(faces, elem);
    double mindist = 1.0e24;
    bool found = false;
    Point ecenter;
    vmesh->get_center(ecenter, elem);
    for (unsigned int j=0; j < faces.size(); j++)
    {
      Point p0, p1, p2;
      vmesh->get_nodes(nodes, faces[j]);
      vmesh->get_center(p0, nodes[0]);
      vmesh->get_center(p1, nodes[1]);
      vmesh->get_center(p2, nodes[2]);
      Vector normal = Cross(p1-p0, p2-p0);
      if (Dot(normal, ecenter-p0) > 0.0) { normal *= -1.0; }
      const double dist = RayPlaneIntersection(seed, dir, p0, normal);
      if (dist > -1.0e-6 && dist < mindist)
      {
	mindist = dist;
	minface = faces[j];
	minnormal = normal;
	found = true;
      }
    }
    if (!found) { break; }

    seed = seed + dir * mindist;

    v.push_back(seed);
    if (!vmesh->get_neighbor(neighbor, elem, minface)) { break; }
    elem = neighbor;
    lastnormal = minnormal;
    if (Dot(lastnormal, dir) < 0.0) { lastnormal *= -1; }
  }

  if (remove_colinear_p)
  {
    v.erase(StreamLinesCleanupPoints(v, 1.0e-6), v.end());
  }
}
						  


template <class SFIELD, class STYPE, class SLOC, class VFLD>
FieldHandle
StreamLinesAccAlgoT<SFIELD, STYPE, SLOC, VFLD>::
execute(FieldHandle seed_field_h,
	FieldHandle vfield_h,
	int maxsteps,
	int direction,
	int value,
	bool remove_colinear_p)
{
  SFIELD *sfield = (SFIELD *) seed_field_h.get_rep();
  typename SFIELD::mesh_handle_type smesh = sfield->get_typed_mesh();

  VFLD *vfield = (VFLD *) vfield_h.get_rep();

  vfield->mesh()->synchronize(Mesh::FACE_NEIGHBORS_E);

  CMesh::handle_type cmesh = scinew CMesh();
  CField *cf = scinew CField(cmesh);

  Point seed;
  typename VFLD::mesh_type::Elem::index_type elem;
  vector<Point> nodes;
  nodes.reserve(maxsteps);

  vector<Point>::iterator node_iter;
  CMesh::Node::index_type n1, n2;

  // Try to find the streamline for each seed point.
  typename SLOC::iterator seed_iter, seed_iter_end;
  smesh->begin(seed_iter);
  smesh->end(seed_iter_end);

  typename SFIELD::fdata_type::iterator data_iter = sfield->fdata().begin();

  int count = 0;

  while (seed_iter != seed_iter_end)
  {
    smesh->get_point(seed, *seed_iter);

    // Is the seed point inside the field?
    if (!vfield->get_typed_mesh()->locate(elem, seed))
    {
      ++seed_iter;
      ++data_iter;
      ++count;
      continue;
    }

    nodes.clear();
    nodes.push_back(seed);

    int cc = 0;

    // Find the negative streamlines.
    if( direction <= 1 )
    {
      FindNodes(nodes, seed, maxsteps, vfield, remove_colinear_p, true);
      if ( direction == 1 )
      {
	std::reverse(nodes.begin(), nodes.end());
	cc = nodes.size();
	cc = -(cc - 1);
      }
    }
    // Append the positive streamlines.
    if( direction >= 1 )
    {
      FindNodes(nodes, seed, maxsteps, vfield, remove_colinear_p, false);
    }

    node_iter = nodes.begin();

    if (node_iter != nodes.end())
    {
      lock.lock();
      n1 = cf->get_typed_mesh()->add_node(*node_iter);

      ostringstream str;
      str << "Streamline " << count << " Node Index";      
      cf->set_property( str.str(), n1, false );

      cf->resize_fdata();

      if( value == 0 )
	cf->set_value((*data_iter), n1);
      else if( value == 1)
	cf->set_value((double)abs(cc), n1);
      else
	cf->set_value((double)(*seed_iter), n1);

      ++node_iter;

      cc++;

      while (node_iter != nodes.end())
      {
	n2 = cf->get_typed_mesh()->add_node(*node_iter);
	cf->resize_fdata();

	if( value == 0 )
	  cf->set_value((*data_iter), n2);
	else if( value == 1)
	  cf->set_value((double)abs(cc), n2);
	else
	  cf->set_value((double)(*seed_iter), n2);

	cf->get_typed_mesh()->add_edge(n1, n2);

	n1 = n2;
	++node_iter;

	cc++;
      }
      lock.unlock();
    }

    ++seed_iter;
    ++data_iter;
    ++count;
  }

  cf->set_property( "Streamline Count", count, false );

  cf->freeze();

  return FieldHandle(cf);
}

} // end namespace SCIRun

#endif // _STREAMLINES_H_
