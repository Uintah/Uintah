/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#ifndef SpatialOps_FieldComparisons_h
#define SpatialOps_FieldComparisons_h

#include <spatialops/Nebo.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <sstream>

// Boost includes //
#include <boost/math/special_functions/next.hpp>

#define FIELDCOMPARISONS_ABS_ERROR_CONST .000001

/**
 * @file FieldComparisons.h
 * @ingroup fields
 * @brief Comparison operators
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 */
namespace SpatialOps{

template<typename FieldT>
class FieldComparisonHelper;

/**
 * @fn template<typename FieldT> bool field_not_equal(const FieldT&, const FieldT&, double)
 * @brief Returns if f1 is element-wise not equal to f2 within a certain relative
 * tolerance.
 * @ingroup fields
 * This function simply calls field_equal and negates it.
 * \c return !field_equal(f1, f2, error, error_abs);
 * error_abs is defined as default to be the L2 norm of \c f1 multiplied by \c error
 * and \c FIELDCOMPARISONS_ABS_ERROR_CONST.
 *
 * WARNING: Undefined behavior if f1 is a field of all 0's.
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param f1 -- Field 1
 * @param f2 -- Field 2
 * @param error -- Allowable percentage of error. i.e. 1% = .01
 */
template<typename FieldT>
bool field_not_equal(const FieldT& f1, const FieldT& f2, double error=0.0) {
  double error_abs = error ? nebo_norm(f1)*error*FIELDCOMPARISONS_ABS_ERROR_CONST : 0;
  return !field_equal(f1, f2, error, error_abs);
}

/**
 * @brief Returns if f1 is element-wise not equal to f2 within a certain relative
 * tolerance.
 *
 * This function simply calls field_equal and negates it.
 * \c return !field_equal(f1, f2, error, error_abs);
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param f1 -- Field 1
 * @param f2 -- Field 2
 * @param error -- Allowable percentage of error. i.e. 1% = .01
 * @param error_abs -- Allowable absolute error passed on to \c field_equal
 */
template<typename FieldT>
bool field_not_equal(const FieldT& f1, const FieldT& f2, double error, const double error_abs) {
  return !field_equal(f1, f2, error, error_abs);
}
//------------------------------------------------------------------

/**
 * @fn template<typename FieldT> bool field_equal(const FieldT&, const FieldT&, double)
 *
 * @brief Returns if f1 is element-wise equal to f2 within a certain relative
 * tolerance.
 *
 * This function returns the result of |f1 - f2|/(error_abs + |f1|) > error element wise.
 * error_abs is defined as default to be the L2 norm of \c f1 multiplied by \c error
 * and \c FIELDCOMPARISONS_ABS_ERROR_CONST.
 *
 * WARNING: Undefined behavior if f1 is a field of all 0's.
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param f1 -- Field 1
 * @param f2 -- Field 2
 * @param error -- Allowable percentage of error. i.e. 1% = .01
 */
template<typename FieldT>
bool field_equal(const FieldT& f1, const FieldT& f2, double error=0.0)
{
  double error_abs = error ? nebo_norm(f1)*error*FIELDCOMPARISONS_ABS_ERROR_CONST : 0;
  return field_equal(f1, f2, error, error_abs);
}

/**
 * @fn template<typename FieldT> bool field_equal(const FieldT&, const FieldT&, double, const double)
 * @brief Determines if f1 is element-wise equal to f2 within a certain relative
 * tolerance.
 *
 * This function returns the result of |f1 - f2|/(error_abs + |f1|) > error element wise.
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param f1 -- Field 1
 * @param f2 -- Field 2
 * @param error -- Allowable percentage of error. i.e. 1% = .01
 * @param errorAbs -- Allowable absolute error.  This term becomes significant
 * in the calculation as f1 approaches zero.
 */
template<typename FieldT>
bool field_equal(const FieldT& f1, const FieldT& f2, double error, const double errorAbs)
{
  const MemoryWindow& w1 = f1.window_with_ghost();
  const MemoryWindow& w2 = f2.window_with_ghost();

  if( w1 != w2 ){
    throw( std::runtime_error( "Attempted comparison between fields of unequal size." ) );
  }

  error = std::abs(error);
  const bool exactComparison = error == 0.0;

  SpatFldPtr<FieldT> tmp1, tmp2;
  typename FieldT::const_iterator if1, iend, if2;

  FieldComparisonHelper<FieldT>::init_iterator(f1, tmp1, if1, iend);
  FieldComparisonHelper<FieldT>::init_iterator(f2, tmp2, if2, iend);

  //do comparison
  for( ; if2 != iend; ++if1, ++if2 ){
    if( exactComparison ){
      if( *if1 != *if2 ) return false;
    }
    else {
      const double denom = std::abs(*if1) + errorAbs;
      if( std::abs(*if1 - *if2)/denom > error )  return false;
    }
  }
  return true;
}
//------------------------------------------------------------------

/**
 * @brief Returns if f1 is element-wise not equal to f2 within a certain absolute
 * tolerance.
 *
 * This function simply returns the negated result of field_equal_abs
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param f1 -- Field 1
 * @param f2 -- Field 2
 * @param error -- Allowable absolute value of error.
 */
template<typename FieldT>
bool field_not_equal_abs(const FieldT& f1, const FieldT& f2, double error=0.0) {
  return !field_equal_abs(f1, f2, error);
}
//------------------------------------------------------------------

/**
 * @brief Returns if f1 is element-wise equal to f2 within a certain absolute
 * tolerance.
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param f1 -- Field 1
 * @param f2 -- Field 2
 * @param error -- Allowable absolute value of error.
 */
template<typename FieldT>
bool field_equal_abs(const FieldT& f1, const FieldT& f2, double error=0.0)
{
  const MemoryWindow w1 = f1.window_with_ghost();
  const MemoryWindow w2 = f2.window_with_ghost();

  if(w1 != w2) {
    throw( std::runtime_error( "Attempted comparison between fields of unequal size." ) );
  }

  error = std::abs(error);
  const bool exactComparison = error == 0.0;

  SpatFldPtr<FieldT> tmp1, tmp2;
  typename FieldT::const_iterator if1, iend, if2;

  FieldComparisonHelper<FieldT>::init_iterator( f1, tmp1, if1, iend );
  FieldComparisonHelper<FieldT>::init_iterator( f2, tmp2, if2, iend );

  //do comparison
  for( ; if2 != iend; ++if1, ++if2 ){
    if( exactComparison ){
      if(*if1 != *if2) return false;
    }
    else {
      if( std::abs(*if1 - *if2) > error ) return false;
    }
  }
  return true;
}
//------------------------------------------------------------------

/**
 * @brief Returns if f1 is element-wise not equal to f2 within a certain number
 * of ulps.
 *
 * This function simply returns the negated result of field_equal_ulp
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param f1 -- Field 1
 * @param f2 -- Field 2
 * @param ulps -- Allowable difference in ulps
 */
template<typename FieldT>
bool field_not_equal_ulp(const FieldT& f1, const FieldT& f2, const unsigned int ulps) {
  return !field_equal_ulp(f1, f2, ulps);
}
//------------------------------------------------------------------

/**
 * @brief Returns if f1 is element-wise equal to f2 within a certain number
 * of ulps.
 *
 * This function determines the amount of ulps two floating point numbers are
 * off and compares them to the allowed tolerance.  Ulp stands for Unit in the
 * Last Place and is a measure of rounding error in floating point numbers.  A
 * more detailed article can be found at:
 * http://en.wikipedia.org/wiki/Unit_in_the_last_place
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param f1 -- Field 1
 * @param f2 -- Field 2
 * @param ulps -- Allowable difference in ulps
 */
template<typename FieldT>
bool field_equal_ulp(const FieldT& f1, const FieldT& f2, const unsigned int ulps)
{
  const MemoryWindow& w1 = f1.window_with_ghost();
  const MemoryWindow& w2 = f2.window_with_ghost();

  if(w1 != w2) {
    throw( std::runtime_error( "Attempted comparison between fields of unequal size." ) );
  }

  const bool exactComparison = ulps == 0;
  SpatFldPtr<FieldT> tmp1, tmp2;
  typename FieldT::const_iterator if1, iend, if2;

  FieldComparisonHelper<FieldT>::init_iterator( f1, tmp1, if1, iend );
  FieldComparisonHelper<FieldT>::init_iterator( f2, tmp2, if2, iend );

  //do comparison
  for( ; if2 != iend; ++if1, ++if2 ){
    if( exactComparison ){
      if( boost::math::float_distance(*if1, *if2) != 0) return false;
    }
    else {
      if (std::abs(boost::math::float_distance(*if1, *if2)) > ulps) return false;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////                                          /////////////////////////////
/////////////////////////////          SCALAR IMPLEMENTATION           /////////////////////////////
/////////////////////////////                                          /////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Returns if f1 is element-wise not equal to the scalar value d
 * within a certain relative tolerance.
 *
 * This function simply calls field_equal and negates it.
 * \c return !field_equal(d, f1, error, error_abs);
 * error_abs is defined as default to be the L2 norm of \c f1 multiplied by \c error
 * and \c FIELDCOMPARISONS_ABS_ERROR_CONST.
 *
 * WARNING: Undefined behavior if f1 is a field of all 0's.
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param d -- Scalar value
 * @param f1 -- Field 1
 * @param error -- Allowable percentage of error. i.e. 1% = .01
 */
template<typename FieldT>
bool field_not_equal(const double d, const FieldT& f1, double error=0.0) {
  double error_abs = error ? nebo_norm(f1)*error*FIELDCOMPARISONS_ABS_ERROR_CONST : 0;
  return !field_equal(d, f1, error, error_abs);
}

/**
 * @brief Returns if f1 is element-wise not equal to the scalar value d
 * within a certain relative tolerance.
 *
 * This function simply calls field_equal and negates it.
 * \c return !field_equal(d, f1, error, error_abs);
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param d -- Scalar value
 * @param f1 -- Field 1
 * @param error -- Allowable percentage of error. i.e. 1% = .01
 * @param error_abs -- Allowable absolute error passed on to \c field_equal
 */
template<typename FieldT>
bool field_not_equal(const double d, const FieldT& f1, double error, const double error_abs) {
  return !field_equal(d, f1, error, error_abs);
}
//------------------------------------------------------------------

/**
 * @brief Returns if f1 is element-wise equal to the scalar value d
 * within a certain relative tolerance.
 *
 * This function returns the result of |d - f1|/(error_abs + |d|) > error element wise.
 * error_abs is defined as default to be the L2 norm of \c f1 multiplied by \c error
 * and \c FIELDCOMPARISONS_ABS_ERROR_CONST.
 *
 * WARNING: Undefined behavior if f1 is a field of all 0's.
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param d -- Scalar value
 * @param f1 -- Field 1
 * @param error -- Allowable percentage of error. i.e. 1% = .01
 */
template<typename FieldT>
bool field_equal(const double d, const FieldT& f1, double error=0.0)
{
  double error_abs = error ? nebo_norm(f1)*error*FIELDCOMPARISONS_ABS_ERROR_CONST : 0;
  return field_equal(d, f1, error, error_abs);
}

/**
 * @fn template<typename FieldT> bool field_equal(const double, const FieldT&, double, const double )
 * @ingroup fields
 * @brief Returns if f1 is element-wise equal to the scalar value d
 * within a certain relative tolerance.
 *
 * This function returns the result of |d - f1|/(error_abs + |d|) > error element wise.
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param d -- Scalar value
 * @param f1 -- Field 1
 * @param error -- Allowable percentage of error. i.e. 1% = .01
 * @param errorAbs -- Allowable absolute error.  This term becomes significant
 * in the calculation as d approaches zero.
 */
template<typename FieldT>
bool field_equal(const double d, const FieldT& f1, double error, const double errorAbs)
{
  error = std::abs(error);
  const bool exactComparison = error == 0.0;

  SpatFldPtr<FieldT> tmp;
  typename FieldT::const_iterator if1, iend;

  FieldComparisonHelper<FieldT>::init_iterator(f1,tmp,if1,iend);

  //do comparison
  const double denom = std::abs(d) + errorAbs;
  for( ; if1 != iend; ++if1 ){
    if( exactComparison ){
      if( *if1 != d ) return false;
    }
    else{
      if( std::abs(d - *if1)/denom > error ) return false;
    }
  }
  return true;
}
//------------------------------------------------------------------

/**
 * @fn template<typename FieldT> bool field_not_equal_abs(const double, const FieldT&, const double )
 * @brief Returns if f1 is element-wise not equal to Scalar value d within a
 * certain absolute tolerance.
 *
 * This function simply returns the negated result of field_equal_abs
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param d -- Scalar value
 * @param f1 -- Field 1
 * @param error -- Allowable absolute value of error.
 */
template<typename FieldT>
bool field_not_equal_abs(const double d, const FieldT& f1, const double error=0.0) {
  return !field_equal_abs(d, f1, error);
}
//------------------------------------------------------------------

/**
 * @fn template<typename FieldT> bool field_equal_abs(const double, const FieldT&, double)
 *
 * @brief Returns if f1 is element-wise equal to Scalar value d within a
 * certain absolute tolerance.
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param d -- Scalar value
 * @param f1 -- Field 1
 * @param error -- Allowable absolute value of error.
 */
template<typename FieldT>
bool field_equal_abs(const double d, const FieldT& f1, double error=0.0)
{
  error = std::abs(error);
  const bool exactComparison = error == 0.0;

  SpatFldPtr<FieldT> tmp;
  typename FieldT::const_iterator if1, iend;

  FieldComparisonHelper<FieldT>::init_iterator(f1,tmp,if1,iend);

  //do comparison
  for( ; if1 != iend; ++if1 ){
    if( exactComparison ){
      if( *if1 != d ) return false;
    }
    else{
      if( std::abs(d - *if1)  > error ) return false;
    }
  }
  return true;
}
//------------------------------------------------------------------

/**
 * @fn template<typename FieldT> bool field_not_equal_ulp(const double, const FieldT&, const unsigned int)
 *
 * @brief Returns if f1 is element-wise not equal to Scalar value d within a
 * certain number of ulps.
 *
 * This function simply returns the negated result of field_equal_ulp
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param d -- Scalar value
 * @param f1 -- Field 1
 * @param ulps -- Allowable difference in ulps
 */
template<typename FieldT>
bool field_not_equal_ulp(const double d, const FieldT& f1, const unsigned int ulps) {
  return !field_equal_ulp(d, f1, ulps);
}
//------------------------------------------------------------------

/**
 * @fn template<typename FieldT> bool field_equal_ulp(const double, const FieldT&, const unsigned int)
 *
 * @brief Returns if f1 is element-wise equal to Scalar value d within a
 * certain number of ulps.
 *
 * This function determines the amount of ulps two floating point numbers are
 * off and compares them to the allowed tolerance.  Ulp stands for Unit in the
 * Last Place and is a measure of rounding error in floating point numbers.  A
 * more detailed article can be found at:
 * http://en.wikipedia.org/wiki/Unit_in_the_last_place
 *
 * WARNING: Slow in general and comparison with external fields will incur copy penalties.
 *
 * @tparam FieldT -- Any type of SpatialField
 * @param d -- Scalar value
 * @param f1 -- Field 1
 * @param ulps -- Allowable difference in ulps
 */
template<typename FieldT>
bool field_equal_ulp(const double d, const FieldT& f1, const unsigned int ulps)
{
  bool exactComparison = ulps == 0;

  SpatFldPtr<FieldT> tmp;
  typename FieldT::const_iterator if1, iend;

  FieldComparisonHelper<FieldT>::init_iterator(f1,tmp,if1,iend);

  //do comparison
  for( ; if1 != iend; ++if1 ){
    if( exactComparison ){
      if( boost::math::float_distance(d, *if1) != 0 ) return false;
    }
    else{
      if( std::abs(boost::math::float_distance(d, *if1)) > ulps ) return false;
    }
  }
  return true;
}

/**
 *  @class FieldComparisonHelper
 *  @ingroup structured
 *
 *  @brief static helper class for field comparison functions
 *
 *  This class contains private helper functions for the field_equal functions.
 *
 *  @tparam FieldT -- Any type of SpatialField
 *
 */
template<typename FieldT>
class FieldComparisonHelper
{
  /**
   * @brief Returns a field iterator to data in \c field or null.  This function
   * is only intended to be used by the functions in this file.
   *
   * This function will a \c const_iterator to the beginning and end of the data
   * in \c field or a copy of the data in \c field.  This function will copy the
   * data of \c field if \c field is not on local ram, and then assign the
   * iterators to the new memory created for the copy in \c localPtr.
   *
   * @tparam FieldT -- Any type of SpatialField
   *
   * @param field -- Field to return an iterator to.
   * @param fcopy -- In the case where the field is not on CPU, then this will
   * hold the CPU copy of the field on return.
   * @param ibegin -- to be populated with the begin() iterator
   * @param iend -- to be populated with the end() iterator
   */
  private:
    inline static void
    init_iterator( const FieldT& field,
                   SpatFldPtr<FieldT>& fcopy,
                   typename FieldT::const_iterator& ibegin,
                   typename FieldT::const_iterator& iend )
    {
      //we need to transfer the field to local ram to iterate
      if( IS_CPU_INDEX( field.active_device_index() ) ){
        ibegin = field.begin();
        iend   = field.end();
      }
#     ifdef ENABLE_CUDA
      else if( IS_GPU_INDEX(field.active_device_index()) ){
        fcopy = SpatialFieldStore::get<FieldT>(field);
        *fcopy <<= field;
        fcopy->add_device(CPU_INDEX);
        //
        // jcs hack to get things working.  There are two issues here:
        //
        //   1. in order to get begin() to use the const version, we need to have
        //      a const field, which "fcopy" is not.
        //
        //   2. For the non-const field to work with a non-const version of begin()
        //      we need to first set the active field location to the CPU.  This
        //      Seems to cause problems with the memory pool due to a bug that
        //      has not yet been resolved.
        //
        const FieldT* const fcopyA = &(*fcopy);
        ibegin = fcopyA->begin();
        iend   = fcopyA->end();
        return;
      }
#     endif
      else{
        throw std::runtime_error("Unrecognized field location (FieldComparisons.h FieldCopmarisonHelper::init_iterator()");
      }
    }

    friend bool field_equal<FieldT>(const FieldT& f1, const FieldT& f2, double error, const double error_abs);
    friend bool field_equal_abs<FieldT>(const FieldT& f1, const FieldT& f2, double error);
    friend bool field_equal_ulp<FieldT>(const FieldT& f1, const FieldT& f2, const unsigned int ulps);

    friend bool field_equal<FieldT>(const double d, const FieldT& f1, double error, const double error_abs);
    friend bool field_equal_abs<FieldT>(const double d, const FieldT& f1, double error);
    friend bool field_equal_ulp<FieldT>(const double d, const FieldT& f1, const unsigned int ulps);
};

} // namespace SpatialOps

#endif //SpatialOps_FieldComparisons_h
