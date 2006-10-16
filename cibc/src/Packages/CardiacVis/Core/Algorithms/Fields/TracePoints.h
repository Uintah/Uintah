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

#ifndef CARDIACVIS_CORE_ALGORITHMS_FIELDS_TRACEPOINTS_H
#define CARDIACVIS_CORE_ALGORITHMS_FIELDS_TRACEPOINTS_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <float.h>

// Additionally we include sci_hash_map here as it is needed by the algorithm

namespace CardiacVis {

using namespace SCIRun;

class TracePointsAlgo : public DynamicAlgoBase
{
public:
  virtual bool TracePoints(ProgressReporter *pr, FieldHandle pointcloud, FieldHandle old_curvefield, FieldHandle& new_curvefield, double val, double tol);
};

template <class FSRC, class FDST>
class TracePointsAlgoT : public TracePointsAlgo
{
public:
  virtual bool TracePoints(ProgressReporter *pr, FieldHandle pointcloud, FieldHandle old_curvefield, FieldHandle& new_curvefield, double val, double tol);
};


template <class FSRC, class FDST>
bool TracePointsAlgoT<FSRC, FDST>::TracePoints(ProgressReporter *pr, FieldHandle pointcloud, FieldHandle old_curvefield, FieldHandle& curvefield, double val, double tol)
{
  FSRC *pfield = dynamic_cast<FSRC *>(pointcloud.get_rep());
  if (pfield == 0)
  { 
    pr->error("TracePoints: Could not obtain PointCloud field");
    return (false);
  }

  typename FSRC::mesh_type* pmesh = dynamic_cast<typename FSRC::mesh_type*>(pfield->get_mesh());
  if (pmesh == 0)
  {
    pr->error("TracePoints: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_type *cmesh;
  FDST *cfield;
  
  if (old_curvefield.get_rep())
  {
    old_curvefield.detach();
    curvefield = old_curvefield;

    cfield = dynamic_cast<FDST *>(curvefield.get_rep());  
    cmesh = dynamic_cast<typename FDST::mesh_type*>(cfield->get_mesh());
  }
  else
  {
    cmesh = scinew typename FDST::mesh_type();
    if (cmesh == 0)
    {
      pr->error("TracePoints: Could not create output mesh");
      return (false);
    }

    curvefield = dynamic_cast<Field*>(scinew FDST(cmesh));
    if (curvefield.get_rep() == 0)
    {
      pr->error("TracePoints: Could not create output field");
      return (false);
    }
    cfield = dynamic_cast<FDST *>(curvefield.get_rep());  
  }

  if (cfield == 0)
  { 
    pr->error("TracePoints: Could not obtain input field");
    return (false);
  }  

  FieldHandle pointcloud_old;
  double val_old;

  if (cfield->is_property("end_points"))
  {
    cfield->get_property("end_points",pointcloud_old);
    cfield->set_property("end_points",pointcloud,false);
    cfield->get_property("value",val_old);
    cfield->set_property("value",val,false);    
  }
  else
  {
    cfield->set_property("end_points",pointcloud,false);
    cfield->set_property("value",val,false);    
    return (true);
  }

  FSRC *pfield2 = dynamic_cast<FSRC *>(pointcloud_old.get_rep());
  if (pfield2 == 0)
  { 
    pr->error("TracePoints: Could not obtain end_points field");
    return (false);
  }

  typename FSRC::mesh_handle_type pmesh2 = pfield->get_typed_mesh();
  if (pmesh2 == 0)
  {
    pr->error("TracePoints: No mesh associated with end_points input field");
    return (false);
  }
  

  typename FSRC::mesh_type::Node::iterator it, it_end;
  typename FSRC::mesh_type::Node::iterator it2, it_end2;
  typename FSRC::value_type pval, pval2;
  typename FDST::mesh_type::Node::array_type na(2);
  Point p, p2,p3;
  double dist = DBL_MAX;
  double tol2 = tol*tol;
  
  pmesh->begin(it);
  pmesh->end(it_end);
  
  while (it != it_end)
  {
    pmesh->get_center(p,*it);
    pfield->value(pval,*it);
    pmesh2->begin(it2);
    pmesh2->end(it_end2);
    while (it2 != it_end2)
    {
      pmesh2->get_center(p2,*it2);
      pfield2->value(pval2,*it);
      Vector v(p2-p);
      if (pval2 == pval) if (v.length2() < dist) p3 = p2;
    }

    if (dist < tol2) 
    {
      na[0] = cmesh->add_point(p);
      cfield->fdata().push_back(val);
      na[1] = cmesh->add_point(p3);
      cfield->fdata().push_back(val_old);
      pmesh2->add_elem(na);
    }
  }

  // Success:
  return (true);
}


} // end namespace CardiacVis

#endif 

