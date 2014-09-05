/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


//    File   : GenerateStreamLines.h
//    Author : Michael Callahan
//    Date   : July 2001

#if !defined(_STREAMLINES_H_)
#define _STREAMLINES_H_

#include <Dataflow/Network/Module.h>

#include <Core/Geometry/CompGeom.h>
#include <Core/Algorithms/Math/BasicIntegrators.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Thread/Thread.h>
#include <algorithm>
#include <sstream>

#include <Dataflow/Modules/Visualization/share.h>

namespace SCIRun {

typedef CurveMesh<CrvLinearLgn<Point> > CMesh;
 
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
  bool remove_colinear_pts;
  int method;
  int nthreads;
  ProgressReporter *reporter;

  _SLData() : lock("GenerateStreamLines Lock") {}
} SLData;


class SCISHARE GenerateStreamLinesAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *reporter,
                              FieldHandle seed_field_h,
			      VectorFieldInterfaceHandle vfi,
			      double tolerance,
			      double stepsize,
			      int maxsteps,
			      int direction,
			      int value,
			      bool remove_colinear_pts,
			      int method, 
			      int nthreads) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const string &dsrc,
					    const TypeDescription *sloc,
					    int value);

  static vector<Point>::iterator CleanupPoints(vector<Point> &in, double e2);
};


template <class SFLD, class STYPE, class SLOC>
class GenerateStreamLinesAlgoT : public GenerateStreamLinesAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter,
                              FieldHandle seed_field_h,
			      VectorFieldInterfaceHandle vfi,
			      double tolerance,
			      double stepsize,
			      int maxsteps,
			      int direction,
			      int value,
			      bool remove_colinear_pts,
			      int method,
			      int nthreads);

  void parallel_generate(int proc, SLData *d);

  virtual void set_result_value(Field *cf,
                                CMesh::Node::index_type di,
                                SFLD *sfield,
                                typename SLOC::index_type si,
                                double data) = 0;
};


template <class FSRC, class STYPE, class SLOC>
FieldHandle
GenerateStreamLinesAlgoT<FSRC, STYPE, SLOC>::
execute(ProgressReporter *reporter,
        FieldHandle seed_field_h,
	VectorFieldInterfaceHandle vfi,
	double tolerance,
	double stepsize,
	int maxsteps,
	int direction,
	int value,
	bool remove_colinear_pts,
	int method,
	int nthreads)
{
  SLData d;
  d.seed_field_h=seed_field_h;
  d.vfi=vfi;
  d.tolerance=tolerance;
  d.stepsize=stepsize;
  d.maxsteps=maxsteps;
  d.direction=direction;
  d.value=value;
  d.remove_colinear_pts=remove_colinear_pts;
  d.method=method;
  d.nthreads=nthreads;
  d.reporter=reporter;

  typedef CrvLinearLgn<STYPE> DatBasisL;
  typedef GenericField<CMesh, DatBasisL, vector<STYPE> > CFieldL;

  CMesh::handle_type cmesh = scinew CMesh();
  CFieldL *cf = scinew CFieldL(cmesh);
  
  d.fh = FieldHandle(cf);

  typename SLOC::size_type prsize_tmp;
  FSRC *sfield = (FSRC *) seed_field_h.get_rep();
  typename FSRC::mesh_handle_type smesh = sfield->get_typed_mesh();
  smesh->size(prsize_tmp);
  const unsigned int prsize = (unsigned int)prsize_tmp;
  reporter->update_progress(0, prsize);

  Thread::parallel(this,
                   &GenerateStreamLinesAlgoT<FSRC, STYPE, SLOC>::parallel_generate,
                   nthreads, &d);

  cf->freeze();

  return cf;
}


template <class FSRC, class STYPE, class SLOC>
void GenerateStreamLinesAlgoT<FSRC, STYPE, SLOC>::
parallel_generate( int proc, SLData *d)
{
  FSRC *sfield = (FSRC *) d->seed_field_h.get_rep();
  typename FSRC::mesh_handle_type smesh = sfield->get_typed_mesh();

  typedef CrvLinearLgn<STYPE> DatBasisL;
  typedef GenericField<CMesh, DatBasisL, vector<STYPE> > CFieldL;

  CFieldL *cfield = (CFieldL *) d->fh.get_rep();

  CMesh::Node::index_type n1, n2;

  Vector test;

  BasicIntegrators BI;
  BI.nodes_.reserve(d->maxsteps);                   // storage for points
  BI.tolerance2_  = d->tolerance * d->tolerance;    // square error tolerance
  BI.maxsteps_    = d->maxsteps;                    // max number of steps
  BI.vfi_         = d->vfi;                         // the vector field

  vector<Point>::iterator node_iter;

  // Try to find the streamline for each seed point.
  typename SLOC::iterator siter, siter_end;
  smesh->begin(siter);
  smesh->end(siter_end);

  int count = 0;
  
  while (siter != siter_end)
  {
    // If this seed doesn't "belong" to this parallel thread,
    // ignore it and continue on the next seed.
    if (count%d->nthreads != proc) {
      ++siter;
      ++count;
      continue;
    }

    d->reporter->increment_progress();

    smesh->get_point(BI.seed_, *siter);

    // Is the seed point inside the field?
    if (!d->vfi->interpolate(test, BI.seed_))
    {
      ++siter;
      ++count;
      continue;
    }

    BI.nodes_.clear();
    BI.nodes_.push_back(BI.seed_);

    int cc = 0;

    // Find the negative streamlines.
    if( d->direction <= 1 ) {
      BI.stepsize_ = -d->stepsize;   // initial step size
      BI.integrate( d->method );

      if ( d->direction == 1 ) {

	BI.seed_ = BI.nodes_[0];  // Reset the seed

	reverse(BI.nodes_.begin(), BI.nodes_.end());
	cc = BI.nodes_.size() - 1;
	cc = -(cc - 1);
      }
    }

    // Append the positive streamlines.
    if( d->direction >= 1 ) {
      BI.stepsize_ = d->stepsize;   // initial step size
      BI.integrate( d->method );
    }

    if (d->remove_colinear_pts)
      BI.nodes_.erase(CleanupPoints(BI.nodes_, BI.tolerance2_),
		      BI.nodes_.end());

    double length = 0;

    Point p1;

    if( d->value == 4 ) {
      node_iter = BI.nodes_.begin();
      if (node_iter != BI.nodes_.end()) {
	p1 = *node_iter;	
	++node_iter;

	while (node_iter != BI.nodes_.end()) {
	  length += Vector( *node_iter-p1 ).length();
	  p1 = *node_iter;
	  ++node_iter;
	}
      }
    }

    node_iter = BI.nodes_.begin();

    if (node_iter != BI.nodes_.end()) {
      d->lock.lock();
      n1 = cfield->get_typed_mesh()->add_node(*node_iter);
      p1 = *node_iter;

      std::ostringstream str;
      str << "Streamline " << count << " Node Index";      
      d->fh->set_property( str.str(), n1, false );

      cfield->resize_fdata();

      if( d->value == 0 )
        set_result_value(cfield, n1, sfield, *siter, 0);
      else if( d->value == 1 )
        set_result_value(cfield, n1, sfield, *siter, (double)(*siter));
      else if( d->value == 2)
        set_result_value(cfield, n1, sfield, *siter, (double)abs(cc));
      else if( d->value == 3)
        set_result_value(cfield, n1, sfield, *siter, length);
      else if( d->value == 4)
        set_result_value(cfield, n1, sfield, *siter, length);

      ++node_iter;

      cc++;

      while (node_iter != BI.nodes_.end()) {
	n2 = cfield->get_typed_mesh()->add_node(*node_iter);
	cfield->resize_fdata();

	if( d->value == 0 )
          set_result_value(cfield, n2, sfield, *siter, 0);
	else if( d->value == 1 )
          set_result_value(cfield, n2, sfield, *siter, (double)(*siter));
	else if( d->value == 2)
          set_result_value(cfield, n2, sfield, *siter, (double)abs(cc));
	else if( d->value == 3)
        {
	  length += Vector( *node_iter-p1 ).length();
          set_result_value(cfield, n2, sfield, *siter, length);
	  p1 = *node_iter;
	}
        else if( d->value == 4)
          set_result_value(cfield, n2, sfield, *siter, length);


	cfield->get_typed_mesh()->add_edge(n1, n2);

	n1 = n2;
	++node_iter;

	cc++;
      }
      d->lock.unlock();
    }

    ++siter;
    ++count;
  }

  d->fh->set_property( "Streamline Count", count, false );
}


template <class SFLD, class STYPE, class SLOC>
class GenerateStreamLinesAlgoTM : public GenerateStreamLinesAlgoT<SFLD, STYPE, SLOC>
{
public:
  virtual void set_result_value(Field *f,
                                CMesh::Node::index_type di,
                                SFLD *sfield,
                                typename SLOC::index_type si,
                                double data)
  {
    typedef CrvLinearLgn<STYPE> DatBasisL;
    typedef GenericField<CMesh, DatBasisL, vector<STYPE> > CFieldL;
    CFieldL *cf = (CFieldL *) f;
    cf->set_value(data, di);
  }
};


template <class SFLD, class STYPE, class SLOC>
class GenerateStreamLinesAlgoTF : public GenerateStreamLinesAlgoT<SFLD, STYPE, SLOC>
{
public:
  virtual void set_result_value(Field *f,
                                CMesh::Node::index_type di,
                                SFLD *sfield,
                                typename SLOC::index_type si,
                                double data)
  {
    typedef CrvLinearLgn<STYPE> DatBasisL;
    typedef GenericField<CMesh, DatBasisL, vector<STYPE> > CFieldL;
    CFieldL *cf = (CFieldL *) f;
    typename CFieldL::value_type val;
    sfield->value(val, si);
    cf->set_value(val, di);
  }
};





class GenerateStreamLinesAccAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *reporter,
                              FieldHandle seed_field_h,
			      FieldHandle vfield_h,
			      int maxsteps,
			      int direction,
			      int value,
			      bool remove_colinear_p) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *sloc,
					    const TypeDescription *vfld,
                                            const string &fdst,
					    int value);

};



template <class FSRC, class SLOC, class VFLD, class FDST>
class GenerateStreamLinesAccAlgoT : public GenerateStreamLinesAccAlgo
{
public:
  virtual FieldHandle execute(ProgressReporter *reporter,
                              FieldHandle seed_field_h,
			      FieldHandle vfield_h,
			      int maxsteps,
			      int direction,
			      int value,
			      bool remove_colinear_p);

  void FindNodes(vector<Point>& nodes, Point seed, int maxsteps, 
		 VFLD *vfield, bool remove_colinear_p, bool back);

  virtual void set_result_value(FDST *cf,
                                typename FDST::mesh_type::Node::index_type di,
                                FSRC *sfield,
                                typename SLOC::index_type si,
                                double data) = 0;

};


template <class FSRC, class SLOC, class VFLD, class FDST>
FieldHandle
GenerateStreamLinesAccAlgoT<FSRC, SLOC, VFLD, FDST>::execute(ProgressReporter *reporter,
                                                     FieldHandle seed_field_h,
                                                     FieldHandle vfield_h,
                                                     int maxsteps,
                                                     int direction,
                                                     int value,
                                                     bool remove_colinear_p)
{
  FSRC *sfield = (FSRC *) seed_field_h.get_rep();
  typename FSRC::mesh_handle_type smesh = sfield->get_typed_mesh();

  VFLD *vfield = (VFLD *) vfield_h.get_rep();

  vfield->mesh()->synchronize(Mesh::FACE_NEIGHBORS_E);

  typename FDST::mesh_handle_type cmesh = scinew typename FDST::mesh_type();
  FDST *cf = scinew FDST(cmesh);

  Point seed;
  typename VFLD::mesh_type::Elem::index_type elem;
  vector<Point> nodes;
  nodes.reserve(maxsteps);

  vector<Point>::iterator node_iter;
  typename FDST::mesh_type::Node::index_type n1, n2;

  // Try to find the streamline for each seed point.
  typename SLOC::iterator siter, siter_end;
  smesh->begin(siter);
  smesh->end(siter_end);

  typename SLOC::size_type prsize_tmp;
  smesh->size(prsize_tmp);
  const unsigned int prsize = (unsigned int)prsize_tmp;

  int count = 0;

  while (siter != siter_end)
  {
    smesh->get_point(seed, *siter);

    // Is the seed point inside the field?
    if (!vfield->get_typed_mesh()->locate(elem, seed))
    {
      ++siter;
      ++count;
      continue;
    }

    reporter->update_progress(count, prsize);

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
      n1 = cf->get_typed_mesh()->add_node(*node_iter);

      ostringstream str;
      str << "Streamline " << count << " Node Index";      
      cf->set_property( str.str(), n1, false );

      cf->resize_fdata();

      if (value == 0)
        set_result_value(cf, n1, sfield, *siter, 0);
      else if( value == 1)
        set_result_value(cf, n1, sfield, *siter, (double)*siter);
      else if (value == 2)
        set_result_value(cf, n1, sfield, *siter, (double)abs(cc));

      ++node_iter;

      cc++;

      while (node_iter != nodes.end())
      {
	n2 = cf->get_typed_mesh()->add_node(*node_iter);
	cf->resize_fdata();
        
        if (value == 0)
          set_result_value(cf, n2, sfield, *siter, 0);
        else if( value == 1)
          set_result_value(cf, n2, sfield, *siter, (double)*siter);
        else if (value == 2)
          set_result_value(cf, n2, sfield, *siter, (double)abs(cc));

	cf->get_typed_mesh()->add_edge(n1, n2);

	n1 = n2;
	++node_iter;

	cc++;
      }
    }

    ++siter;
    ++count;
  }

  cf->set_property( "Streamline Count", count, false );

  cf->freeze();

  return FieldHandle(cf);
}


template <class FSRC, class SLOC, class VFLD, class FDST>
void
GenerateStreamLinesAccAlgoT<FSRC, SLOC, VFLD, FDST>::FindNodes(vector<Point> &v,
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
    v.erase(GenerateStreamLinesAlgo::CleanupPoints(v, 1.0e-6), v.end());
  }
}



template <class FSRC, class SLOC, class VFLD, class FDST>
class GenerateStreamLinesAccAlgoTM : public GenerateStreamLinesAccAlgoT<FSRC, SLOC, VFLD, FDST>
{
public:

  virtual void set_result_value(FDST *cf,
                                typename FDST::mesh_type::Node::index_type di,
                                FSRC *sfield,
                                typename SLOC::index_type si,
                                double data)
  {
    cf->set_value(data, di);
  }
};


template <class FSRC, class SLOC, class VFLD, class FDST>
class GenerateStreamLinesAccAlgoTF : public GenerateStreamLinesAccAlgoT<FSRC, SLOC, VFLD, FDST>
{
public:

  virtual void set_result_value(FDST *cf,
                                typename FDST::mesh_type::Node::index_type di,
                                FSRC *sfield,
                                typename SLOC::index_type si,
                                double data)
  {
    typename FDST::value_type val;
    sfield->value(val, si);
    cf->set_value(val, di);
  }
};


} // end namespace SCIRun

#endif // _STREAMLINES_H_
