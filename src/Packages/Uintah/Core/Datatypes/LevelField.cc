#include "LevelField.h"

namespace Uintah {

using SCIRun::Vector;

template<>
void make_minmax_thread<Matrix3>::run()
{
  sema_->up();
}

template<>
void make_minmax_thread<Vector>::run()
{
  sema_->up();
}

template <> 
bool LevelField<Matrix3>::get_gradient(Vector &, const Point &p)
{
  return false;
}
template <> 
bool LevelField<Vector>::get_gradient(Vector &, const Point &p)
{
  return false;
}

template <>
bool LevelField<Vector>::minmax( pair<double, double> & mm) const
{
  return false;
}

template <>
bool LevelField<Matrix3>::minmax( pair<double, double> & mm) const
{
  return false;
}


template <>
bool LevelFieldSFI<double>::interpolate( double& result, const Point &p) const
{
  return fld_->interpolate(result, p);
}

template <>
bool LevelFieldSFI<float>::interpolate( double& result, const Point &p) const
{
  bool success;
  float result_;
  success = fld_->interpolate( result_, p);
  if( success ) result = double(result_);
  return success;
}
template <>
bool LevelFieldSFI<long>::interpolate( double& result, const Point &p) const
{
  bool success;
  long result_;
  success = fld_->interpolate( result_, p);
  if( success ) result = double(result_);
  return success;
}


template <> ScalarFieldInterface *
LevelField<double>::query_scalar_interface() const
{
  return scinew LevelFieldSFI<double>(this);
}
template <> ScalarFieldInterface *
LevelField<float>::query_scalar_interface() const
{
  return scinew LevelFieldSFI<float >(this);
}


template <> ScalarFieldInterface *
LevelField<long>::query_scalar_interface() const
{
  return scinew LevelFieldSFI<long>(this);
}

template <>
VectorFieldInterface*
LevelField<Vector>::query_vector_interface() const 
{
  return scinew VFInterface<LevelField<Vector> >(this);
}



} // end namespace Uintah
