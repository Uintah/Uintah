#include "LevelField.h"

namespace Uintah {


template<>
void LevelField<Matrix3>::make_minmax_thread::run()
{
  sema_->up();
}
template<>
void LevelField<Vector>::make_minmax_thread::run()
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
bool LevelFieldSFI<int>::interpolate( double& result, const Point &p) const
{
  bool success;
  int result_;
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
} // end namespace Uintah
