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

//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   May 2002
//
//  Copyright (C) 2002 SCI Institute
//
//
//
// This is the templated implementations of the FieldInterface classes.
// It should not need to be widely included.
//


#ifndef Datatypes_FieldInterfaceAux_h
#define Datatypes_FieldInterfaceAux_h

#include <Core/Datatypes/Field.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Util/DynamicLoader.h>
#include <float.h> // for DBL_MAX


namespace SCIRun {

//! Should only be instantiated for fields with scalar data.
template <class F, class L>
class SFInterface : public ScalarFieldInterface {
public:
  SFInterface(LockingHandle<F> fld) :
    field_(fld)
  {}
  
  virtual bool compute_min_max(double &minout, double &maxout,
			       bool cache = true);
  virtual bool interpolate(double &result, const Point &p) const;
  virtual bool interpolate_many(vector<double> &results,
				const vector<Point> &points) const;
  virtual double find_closest(double &result, const Point &p) const;

  virtual void io(Piostream&){};
private:

  bool compute_min_max_aux(double &minout, double &maxout) const;
  bool finterpolate(double &result, const Point &p) const;

  LockingHandle<F> field_;
};


class ScalarFieldInterfaceMaker : public DynamicAlgoBase
{
public:
  virtual ScalarFieldInterfaceHandle make(FieldHandle field) = 0;
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ltd);
};



template <class F, class L>
class SFInterfaceMaker : public ScalarFieldInterfaceMaker
{
public:

  virtual ScalarFieldInterfaceHandle make(FieldHandle field)
  {
    F *tfield = dynamic_cast<F *>(field.get_rep());
    ASSERT(tfield);
    
    return ScalarFieldInterfaceHandle(scinew SFInterface<F, L>(tfield));
  }
};



template <class F, class L>
bool
SFInterface<F, L>::finterpolate(double &result, const Point &p) const
{
  typename F::mesh_handle_type mesh = field_->get_typed_mesh();

  typename L::array_type locs;
  vector<double> weights;
  mesh->get_weights(p, locs, weights);

  // weights is empty if point not found.
  if (weights.size() <= 0) return false;

  result = 0.0;
  for (unsigned int i = 0; i < locs.size(); i++)
  {
    typename F::value_type tmp;
    if (field_->value(tmp, locs[i]))
    {
      result += tmp * weights[i];
    }
  }

  return true;
}


template <class F, class L>
bool
SFInterface<F, L>::interpolate(double &result, const Point &p) const
{
  return finterpolate(result, p);
}


template <class F, class L>
bool
SFInterface<F, L>::interpolate_many(vector<double> &results,
				    const vector<Point> &points) const
{
  bool all_interped_p = true;
  results.resize(points.size());
  unsigned int i;
  for (i=0; i < points.size(); i++)
  {
    all_interped_p &=  interpolate(results[i], points[i]);
  }
  return all_interped_p;
}


template <class F, class L>
bool
SFInterface<F, L>::compute_min_max_aux(double &minout, double &maxout) const
{
  bool result = false;

  typename F::fdata_type::const_iterator bi, ei;
  bi = field_->fdata().begin();
  ei = field_->fdata().end();

  if (bi != ei)
  {
    result = true;
    minout = DBL_MAX;
    maxout = -DBL_MAX;
  }

  while (bi != ei)
  {
    typename F::value_type val = *bi;
    if (!result || val < minout) minout = val;
    if (!result || val > maxout) maxout = val;
    ++bi;
  }      

  return result;
}


template <class F, class L>
bool
SFInterface<F, L>::compute_min_max(double &minout, double &maxout, bool cache)
{
  if (cache)
  {
    std::pair<double, double> minmax;
    if (field_->get_property("minmax", minmax))
    {
      minout = minmax.first;
      maxout = minmax.second;
      return true;
    }
    else
    {
      field_->freeze();
      if (compute_min_max_aux(minmax.first, minmax.second))
      {
	field_->set_property("minmax", minmax, true);
	minout = minmax.first;
	maxout = minmax.second;
	return true;
      }
      else
      {
	return false;
      }
    }
  }
  else
  {
    return compute_min_max_aux(minout, maxout);
  }
}


template <class F, class L>
double
SFInterface<F, L>::find_closest(double &minout, const Point &p) const
{
  double mindist = DBL_MAX;
  typename F::mesh_handle_type mesh = field_->get_typed_mesh();
  Field::data_location d_at = field_->data_at();
  if (d_at == Field::NODE) mesh->synchronize(Mesh::NODES_E);
  else if (d_at == Field::CELL) mesh->synchronize(Mesh::CELLS_E);
  else if (d_at == Field::FACE) mesh->synchronize(Mesh::FACES_E);
  else if (d_at == Field::EDGE) mesh->synchronize(Mesh::EDGES_E);

  typename L::index_type index;
  typename L::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    Point c;
    mesh->get_center(c, *bi);
    const double dist = (p - c).length2();
    if (dist < mindist)
    {
      mindist = dist;
      index = *bi;
    }
    ++bi;
  }
  typename F::value_type val;
  field_->value(val, index);
  minout = (double)val;

  return mindist;
}



//! Should only be instantiated for fields with vector data.
template <class F, class L>
class VFInterface : public VectorFieldInterface {
public:
  VFInterface(LockingHandle<F> fld) :
    field_(fld)
  {}
  
  virtual bool compute_min_max(Vector &minout, Vector  &maxout,
			       bool cache = true);
  virtual bool interpolate(Vector &result, const Point &p) const;
  virtual bool interpolate_many(vector<Vector> &results,
				const vector<Point> &points) const;
  virtual double find_closest(Vector &result, const Point &p) const;
  virtual void io(Piostream&){};
private:
  bool compute_min_max_aux(Vector &minout, Vector  &maxout) const;
  bool finterpolate(Vector &result, const Point &p) const;

  LockingHandle<F>  field_;
};



class VectorFieldInterfaceMaker : public DynamicAlgoBase
{
public:
  virtual VectorFieldInterfaceHandle make(FieldHandle field) = 0;
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ltd);
};



template <class F, class L>
class VFInterfaceMaker : public VectorFieldInterfaceMaker
{
public:

  virtual VectorFieldInterfaceHandle make(FieldHandle field)
  {
    F *tfield = dynamic_cast<F *>(field.get_rep());
    ASSERT(tfield);
    return VectorFieldInterfaceHandle(scinew VFInterface<F, L>(tfield));
  }
};



template <class F, class L>
bool
VFInterface<F, L>::finterpolate(Vector &result, const Point &p) const
{
  typename F::mesh_handle_type mesh = field_->get_typed_mesh();

  typename L::array_type locs;
  vector<double> weights;
  mesh->get_weights(p, locs, weights);

  // weights is empty if point not found.
  if (weights.size() <= 0) return false;

  result = Vector(0.0, 0.0, 0.0);
  for (unsigned int i = 0; i < locs.size(); i++)
  {
    typename F::value_type tmp;
    if (field_->value(tmp, locs[i]))
    {
      result += tmp * weights[i];
    }
  }

  return true;
}


template <class F, class L>
bool
VFInterface<F, L>::interpolate(Vector &result, const Point &p) const
{
  return finterpolate(result, p);
}


template <class F, class L>
bool
VFInterface<F, L>::interpolate_many(vector<Vector> &results,
				    const vector<Point> &points) const
{
  bool all_interped_p = true;
  results.resize(points.size());
  unsigned int i;
  for (i=0; i < points.size(); i++)
  {
    all_interped_p &=  interpolate(results[i], points[i]);
  }
  return all_interped_p;
}


template <class F, class L>
bool
VFInterface<F, L>::compute_min_max_aux(Vector &minout, Vector &maxout) const
{
  bool result = false;
  typename F::fdata_type::const_iterator bi, ei;
  bi = field_->fdata().begin();
  ei = field_->fdata().end();

  if (bi != ei)
  {
    result = true;
    minout.x(DBL_MAX);
    minout.y(DBL_MAX);
    minout.z(DBL_MAX);
    maxout.x(-DBL_MAX);
    maxout.y(-DBL_MAX);
    maxout.z(-DBL_MAX);
  }

  while (bi != ei)
  {
    typename F::value_type val = *bi;
    if (!result ||val.x() < minout.x()) minout.x(val.x());
    if (!result ||val.y() < minout.y()) minout.y(val.y());
    if (!result ||val.z() < minout.z()) minout.z(val.z());
    
    if (!result ||val.x() > maxout.x()) maxout.x(val.x());
    if (!result ||val.y() > maxout.y()) maxout.y(val.y());
    if (!result ||val.z() > maxout.z()) maxout.z(val.z());
    ++bi;
  }      

  return result;
}


template <class F, class L>
bool
VFInterface<F, L>::compute_min_max(Vector &minout, Vector &maxout, bool cache)
{
  if (cache)
  {
    std::pair<Vector, Vector> minmax;
    if (field_->get_property("minmax", minmax))
    {
      minout = minmax.first;
      maxout = minmax.second;
      return true;
    }
    else
    {
      field_->freeze();
      if (compute_min_max_aux(minmax.first, minmax.second))
      {
	field_->set_property("minmax", minmax, true);
	minout = minmax.first;
	maxout = minmax.second;
	return true;
      }
      else
      {
	return false;
      }
    }
  }
  else
  {
    return compute_min_max_aux(minout, maxout);
  }
}


template <class F, class L>
double
VFInterface<F, L>::find_closest(Vector &minout, const Point &p) const
{
  double mindist = DBL_MAX;
  typename F::mesh_handle_type mesh = field_->get_typed_mesh();
  Field::data_location d_at = field_->data_at();
  if (d_at == Field::NODE) mesh->synchronize(Mesh::NODES_E);
  else if (d_at == Field::CELL) mesh->synchronize(Mesh::CELLS_E);
  else if (d_at == Field::FACE) mesh->synchronize(Mesh::FACES_E);
  else if (d_at == Field::EDGE) mesh->synchronize(Mesh::EDGES_E);

  typename L::index_type index;
  typename L::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    Point c;
    mesh->get_center(c, *bi);
    const double dist = (p - c).length2();
    if (dist < mindist)
    {
      mindist = dist;
      index = *bi;
    }
    ++bi;
  }
  field_->value(minout, index);

  return mindist;
}


//! Should only be instantiated for fields with tensor data.
template <class F, class L>
class TFInterface : public TensorFieldInterface {
public:
  TFInterface(LockingHandle<F> fld) :
    field_(fld)
  {}

  virtual bool interpolate(Tensor &result, const Point &p) const;
  virtual bool interpolate_many(vector<Tensor> &results,
				const vector<Point> &points) const;
  virtual double find_closest(Tensor &result, const Point &p) const;
    virtual void io(Piostream&){};
private:
  bool finterpolate(Tensor &result, const Point &p) const;

  LockingHandle<F> field_;
};



class TensorFieldInterfaceMaker : public DynamicAlgoBase
{
public:
  virtual TensorFieldInterfaceHandle make(FieldHandle field) = 0;
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ltd);
};



template <class F, class L>
class TFInterfaceMaker : public TensorFieldInterfaceMaker
{
public:

  virtual TensorFieldInterfaceHandle make(FieldHandle field)
  {
    F *tfield = dynamic_cast<F *>(field.get_rep());
    ASSERT(tfield);
    return TensorFieldInterfaceHandle(scinew TFInterface<F, L>(tfield));
  }
};



template <class F, class L>
bool
TFInterface<F, L>::finterpolate(Tensor &result, const Point &p) const
{
  typename F::mesh_handle_type mesh = field_->get_typed_mesh();

  typename L::array_type locs;
  vector<double> weights;
  mesh->get_weights(p, locs, weights);

  // weights is empty if point not found.
  if (weights.size() <= 0) return false;

  result = Tensor(0);
  for (unsigned int i = 0; i < locs.size(); i++)
  {
    typename F::value_type tmp;
    if (field_->value(tmp, locs[i]))
    {
      result += tmp * weights[i];
    }
  }

  return true;
}


template <class F, class L>
bool
TFInterface<F, L>::interpolate(Tensor &result, const Point &p) const
{
  return finterpolate(result, p);
}


template <class F, class L>
bool
TFInterface<F, L>::interpolate_many(vector<Tensor> &results,
				    const vector<Point> &points) const
{
  bool all_interped_p = true;
  results.resize(points.size());
  unsigned int i;
  for (i=0; i < points.size(); i++)
  {
    all_interped_p &=  interpolate(results[i], points[i]);
  }
  return all_interped_p;
}


template <class F, class L>
double
TFInterface<F, L>::find_closest(Tensor &minout, const Point &p) const
{
  double mindist = DBL_MAX;
  typename F::mesh_handle_type mesh = field_->get_typed_mesh();
  Field::data_location d_at = field_->data_at();
  if (d_at == Field::NODE) mesh->synchronize(Mesh::NODES_E);
  else if (d_at == Field::CELL) mesh->synchronize(Mesh::CELLS_E);
  else if (d_at == Field::FACE) mesh->synchronize(Mesh::FACES_E);
  else if (d_at == Field::EDGE) mesh->synchronize(Mesh::EDGES_E);


  typename L::index_type index;
  typename L::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    Point c;
    mesh->get_center(c, *bi);
    const double dist = (p - c).length2();
    if (dist < mindist)
    {
      mindist = dist;
      index = *bi;
    }
    ++bi;
  }

  field_->value(minout, index);

  return mindist;
}


} // end namespace SCIRun


#endif // Datatypes_FieldInterface_h


