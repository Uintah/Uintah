/*
 *  ScalarTransform1D.h: Transfer function for scalars
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef RTRT_Classlib_ScalarTransform_h
#define RTRT_Classlib_ScalarTransform_h 1

#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {
  
// LType is the lookup type - it should be a scalar
// RType is the result type - can be any type
template <class LType, class RType>
class ScalarTransform1D {
  Array1<RType> *results;
  LType min, max;
  double linear_factor;
  bool isscaled;


  inline int bound(const int val, const int min, const int max) const {
    return (val>min?(val<max?val:max):min);
  }

public:
  ScalarTransform1D(Array1<RType> *results=0);
  ScalarTransform1D(const Array1<RType> &r);
  ~ScalarTransform1D() {}

  // access members

  ////////////////////////////////////////////////////////////
  // does no checking for whether val lies between min and max
  inline int get_lookup_index(const LType val) const {
    //return (int)((val - min) * linear_factor * (results->size()-1) + 0.5);
    return (int)((val - min) * linear_factor * (results->size()-1));
  }
  inline RType lookup(const LType val) const {
    return (*results)[get_lookup_index(val)];
  }
    
  ////////////////////////////////////////////////////////////
  // same as lookup() though it clams val between min and max
  inline int get_bound_lookup_index(const LType val) const {
    //int index=(int)((val - min) * linear_factor * (results->size()-1) + 0.5);
    int index = (int)((val - min) * linear_factor * (results->size()-1));
    return bound(index, 0, results->size()-1);
    
  }
  inline RType lookup_bound(const LType val) const {
    return (*results)[get_bound_lookup_index(val)];
  }


  ////////////////////////////////////////////////////////////
  // looks up with the passed in min and max
  inline int get_lookup_index(const LType val, const LType _min,
			      const LType _max) const {
    //int index=(int)((val - _min) / (_max - _min) * (results->size()-1) + 0.5)
    int index = (int)((val - _min) / (_max - _min) * (results->size()-1));
    return bound(index, 0, results->size()-1);
  }
  inline RType lookup(const LType val, const LType _min,
		      const LType _max) const {
    return (*results)[get_lookup_index(val,_min,_max)];
  }


  ////////////////////////////////////////////////////////////
  // clamps index to fit in the range of good values
  inline RType lookup_with_index(const int index) const {
    return (*results)[bound(index, 0, results->size()-1)];
  }


  ////////////////////////////////////////////////////////////
  // returns &results[i], no bounds checking
  inline RType& operator[](const int index) const {
    return (*results)[index];
  }

  ////////////////////////////////////////////////////////////
  // returns the interpolated value of
  inline RType interpolate(const LType val) const {
    double step = (val - min) * linear_factor * (results->size()-1);
    int x_low = bound((int)step, 0, results->size()-2);
    double x_weight_low = (x_low+1) - step;
    return (*results)[x_low] * x_weight_low +
      (*results)[x_low+1] * (1 - x_weight_low);
  }
  
  Array1<RType> *get_results_ptr() {
    return results;
  }
  void set_results_ptr(Array1<RType> *_results) {
    results = _results;
  }
  
  // min and max operators
  void get_min_max(LType &_min, LType &_max) const;
  void scale(LType _min, LType _max);
  bool is_scaled() const { return isscaled; } 

  inline int size() const { return results->size(); }
};
		 
template <class LType, class RType>
ScalarTransform1D<LType,RType>::ScalarTransform1D(Array1<RType> *results):
  results(results), min(0), max(1), linear_factor(1), isscaled(false)
{}

template <class LType, class RType>
ScalarTransform1D<LType,RType>::ScalarTransform1D(const Array1<RType> &r):
  min(0), max(1), linear_factor(1), isscaled(false)
{
  results = new Array1<RType>(r);
}

// min and max operators
template <class LType, class RType>
void
ScalarTransform1D<LType,RType>::get_min_max(LType &_min, LType &_max) const {
  _min = min;
  _max = max;
}

template <class LType, class RType>
void ScalarTransform1D<LType,RType>::scale(LType _min, LType _max) {
  min = _min;
  max = _max;
  // make sure that min and max are in the right order
  if (min > max) {
    LType t = min;
    min = max;
    max = t;
  }
  // if min and max equal each other set linear_factor to 0
  if (min == max)
    linear_factor = 0;
  else
    linear_factor = 1.0 / (max - min);

  isscaled = true;
}

} // end namespace rtrt

#endif // RTRT_Classlib_ScalarTransform_h
