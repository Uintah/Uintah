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
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   May 2000
//
//  Copyright (C) 2000 SCI Institute
//
//
//
// To add an Interface, a new class should be created in this file,
// and the appropriate query pure virtual should be added in the Field class.
//


#ifndef Datatypes_FieldInterface_h
#define Datatypes_FieldInterface_h

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Datatypes/Field.h>

namespace SCIRun {


class ScalarFieldInterface {
public:

  ScalarFieldInterface() {}
  ScalarFieldInterface(const ScalarFieldInterface&) {}
  virtual ~ScalarFieldInterface() {}

  virtual bool compute_min_max(double &minout, double &maxout) const = 0;
  virtual bool interpolate(double &result, const Point &p) const = 0;
  virtual bool interpolate_many(vector<double> &results,
				const vector<Point> &points) const = 0;

  virtual void find_closest(double &result, const Point &p) const = 0;
};


//! Should only be instantiated for fields with scalar data.
template <class F>
class SFInterface : public ScalarFieldInterface {
public:
  SFInterface(const F *fld) :
    fld_(fld)
  {}
  
  virtual bool compute_min_max(double &minout, double &maxout) const;
  virtual bool interpolate(double &result, const Point &p) const;
  virtual bool interpolate_many(vector<double> &results,
				const vector<Point> &points) const;
  virtual void find_closest(double &result, const Point &p) const;
private:

  bool finterpolate(double &result, const Point &p) const;

  const F        *fld_;
};


template <class F>
bool
SFInterface<F>::finterpolate(double &result, const Point &p) const
{
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch(fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::NONE:
    return false;
  }
  return true;
}


template <class F>
bool
SFInterface<F>::interpolate(double &result, const Point &p) const
{
  return finterpolate(result, p);
}


template <class F>
bool
SFInterface<F>::interpolate_many(vector<double> &results,
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


template <class F>
bool
SFInterface<F>::compute_min_max(double &minout, double &maxout) const
{
  bool result = false;
  minout = 1.0e6;
  maxout = -1.0e6;
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch (fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Node::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val < minout) minout = val;
	  if (val > maxout) maxout = val;
	  result = true;
	}
	++bi;
      }      
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Edge::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val < minout) minout = val;
	  if (val > maxout) maxout = val;
	  result = true;
	}
	++bi;
      }      
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Face::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val < minout) minout = val;
	  if (val > maxout) maxout = val;
	  result = true;
	}
	++bi;
      }      
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Cell::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val < minout) minout = val;
	  if (val > maxout) maxout = val;
	  result = true;
	}
	++bi;
      }      
    }
    break;
    
  case F::NONE:
    break;
  }
  return result;
}


template <class F>
void
SFInterface<F>::find_closest(double &minout, const Point &p) const
{
  double mindist = 1.0e6;
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch (fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::index_type index;
      typename F::mesh_type::Node::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Node::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      typename F::value_type val;
      fld_->value(val, index);
      minout = (double)val;
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::index_type index;
      typename F::mesh_type::Edge::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Edge::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      typename F::value_type val;
      fld_->value(val, index);
      minout = (double)val;
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::index_type index;
      typename F::mesh_type::Face::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Face::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      typename F::value_type val;
      fld_->value(val, index);
      minout = (double)val;
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::index_type index;
      typename F::mesh_type::Cell::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Cell::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      typename F::value_type val;
      fld_->value(val, index);
      minout = (double)val;
    }
    break;
    
  case F::NONE:
    break;
  }
}


class VectorFieldInterface {
public:
  VectorFieldInterface() {}
  VectorFieldInterface(const VectorFieldInterface&) {}
  virtual ~VectorFieldInterface() {}

  virtual bool compute_min_max(Vector &minout, Vector &maxout) const = 0;
  virtual bool interpolate(Vector &result, const Point &p) const = 0;
  virtual bool interpolate_many(vector<Vector> &results,
				const vector<Point> &points) const = 0;
  virtual void find_closest(Vector &result, const Point &p) const = 0;
};



//! Should only be instantiated for fields with scalar data.
template <class F>
class VFInterface : public VectorFieldInterface {
public:
  VFInterface(const F *fld) :
    fld_(fld)
  {}
  
  virtual bool compute_min_max(Vector &minout, Vector  &maxout) const;
  virtual bool interpolate(Vector &result, const Point &p) const;
  virtual bool interpolate_many(vector<Vector> &results,
				const vector<Point> &points) const;
  virtual void find_closest(Vector &result, const Point &p) const;

private:
  bool finterpolate(Vector &result, const Point &p) const;

  const F        *fld_;
};


template <class F>
bool
VFInterface<F>::finterpolate(Vector  &result, const Point &p) const
{
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch(fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::NONE:
    return false;
  }
  return true;
}


template <class F>
bool
VFInterface<F>::interpolate(Vector &result, const Point &p) const
{
  return finterpolate(result, p);
}


template <class F>
bool
VFInterface<F>::interpolate_many(vector<Vector> &results,
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


template <class F>
bool
VFInterface<F>::compute_min_max(Vector  &minout, Vector  &maxout) const
{
  static const Vector MaxVector(1.0e6, 1.0e6, 1.0e6);
  static const Vector MinVector(-1.0e6, -1.0e6, -1.0e6);

  bool result = false;
  minout = MaxVector;
  maxout = MinVector;
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch (fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Node::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val.x() < minout.x()) minout.x(val.x());
	  if (val.y() < minout.y()) minout.y(val.y());
	  if (val.z() < minout.z()) minout.z(val.z());

	  if (val.x() > maxout.x()) maxout.x(val.x());
	  if (val.y() > maxout.y()) maxout.y(val.y());
	  if (val.z() > maxout.z()) maxout.z(val.z());
	  result = true;
	}
	++bi;
      }      
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Edge::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val.x() < minout.x()) minout.x(val.x());
	  if (val.y() < minout.y()) minout.y(val.y());
	  if (val.z() < minout.z()) minout.z(val.z());

	  if (val.x() > maxout.x()) maxout.x(val.x());
	  if (val.y() > maxout.y()) maxout.y(val.y());
	  if (val.z() > maxout.z()) maxout.z(val.z());
	  result = true;
	}
	++bi;
      }      
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Face::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val.x() < minout.x()) minout.x(val.x());
	  if (val.y() < minout.y()) minout.y(val.y());
	  if (val.z() < minout.z()) minout.z(val.z());

	  if (val.x() > maxout.x()) maxout.x(val.x());
	  if (val.y() > maxout.y()) maxout.y(val.y());
	  if (val.z() > maxout.z()) maxout.z(val.z());
	  result = true;
	}
	++bi;
      }      
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Cell::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	typename F::value_type val;
	if (fld_->value(val, *bi))
	{
	  if (val.x() < minout.x()) minout.x(val.x());
	  if (val.y() < minout.y()) minout.y(val.y());
	  if (val.z() < minout.z()) minout.z(val.z());

	  if (val.x() > maxout.x()) maxout.x(val.x());
	  if (val.y() > maxout.y()) maxout.y(val.y());
	  if (val.z() > maxout.z()) maxout.z(val.z());
	  result = true;
	}
	++bi;
      }      
    }
    break;
    
  case F::NONE:
    break;
  }
  return result;
}


template <class F>
void
VFInterface<F>::find_closest(Vector &minout, const Point &p) const
{
  double mindist = 1.0e6;
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch (fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::index_type index;
      typename F::mesh_type::Node::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Node::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      fld_->value(minout, index);
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::index_type index;
      typename F::mesh_type::Edge::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Edge::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      fld_->value(minout, index);
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::index_type index;
      typename F::mesh_type::Face::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Face::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      fld_->value(minout, index);
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::index_type index;
      typename F::mesh_type::Cell::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Cell::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      fld_->value(minout, index);
    }
    break;
    
  case F::NONE:
    break;
  }
}



class TensorFieldInterface
{
public:
  TensorFieldInterface() {}
  TensorFieldInterface(const TensorFieldInterface&) {}
  virtual ~TensorFieldInterface() {}

  virtual bool interpolate(Tensor &result, const Point &p) const = 0;
  virtual bool interpolate_many(vector<Tensor> &results,
				const vector<Point> &points) const = 0;
  virtual void find_closest(Tensor &result, const Point &p) const = 0;
};


//! Should only be instantiated for fields with scalar data.
template <class F>
class TFInterface : public TensorFieldInterface {
public:
  TFInterface(const F *fld) :
    fld_(fld)
  {}

  virtual bool interpolate(Tensor &result, const Point &p) const;
  virtual bool interpolate_many(vector<Tensor> &results,
				const vector<Point> &points) const;
  virtual void find_closest(Tensor &result, const Point &p) const;
  
private:
  bool finterpolate(Tensor &result, const Point &p) const;

  const F        *fld_;
};


template <class F>
bool
TFInterface<F>::finterpolate(Tensor &result, const Point &p) const
{
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch(fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::array_type locs;
      vector<double> weights;
      mesh->get_weights(p, locs, weights);

      // weights is empty if point not found.
      if (weights.size() <= 0) return false;

      result = 0;
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	typename F::value_type tmp;
	if (fld_->value(tmp, locs[i]))
	{
	  result += tmp * weights[i];
	}
      }
    }
    break;

  case F::NONE:
    return false;
  }
  return true;
}



template <class F>
bool
TFInterface<F>::interpolate(Tensor &result, const Point &p) const
{
  return finterpolate(result, p);
}


template <class F>
bool
TFInterface<F>::interpolate_many(vector<Tensor> &results,
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


template <class F>
void
TFInterface<F>::find_closest(Tensor &minout, const Point &p) const
{
  double mindist = 1.0e6;
  typename F::mesh_handle_type mesh = fld_->get_typed_mesh();
  switch (fld_->data_at())
  {
  case F::NODE:
    {
      typename F::mesh_type::Node::index_type index;
      typename F::mesh_type::Node::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Node::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      fld_->value(minout, index);
    }
    break;

  case F::EDGE:
    {
      typename F::mesh_type::Edge::index_type index;
      typename F::mesh_type::Edge::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Edge::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      fld_->value(minout, index);
    }
    break;

  case F::FACE:
    {
      typename F::mesh_type::Face::index_type index;
      typename F::mesh_type::Face::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Face::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      fld_->value(minout, index);
    }
    break;

  case F::CELL:
    {
      typename F::mesh_type::Cell::index_type index;
      typename F::mesh_type::Cell::iterator bi; mesh->begin(bi);
      typename F::mesh_type::Cell::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	}
	++bi;
      }
      fld_->value(minout, index);
    }
    break;
    
  case F::NONE:
    break;
  }
}


} // end namespace SCIRun


#endif // Datatypes_FieldInterface_h


