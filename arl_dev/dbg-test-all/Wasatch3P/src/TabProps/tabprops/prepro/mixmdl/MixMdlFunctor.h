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

#ifndef MIXMDL_FUNCTOR
#define MIXMDL_FUNCTOR

#include <vector>

//====================================================================

/**
 *  @class FunctorDoubleBase
 *  @author James C. Sutherland
 *  @date   Feb, 2006
 *
 *  Provides a common interface to various functors used by the mixing
 *  models.  The rule is that the call signature for this functor
 *  requires a single double.
 */
class FunctorDoubleBase
{
public:
  FunctorDoubleBase(){nCalls_ = 0;}
  virtual ~FunctorDoubleBase(){}

  virtual double operator()(const double x) = 0;

  /** @brief Obtain the number of calls made to the functor */
  int get_n_calls() const {return nCalls_;}

  /** @brief Zero the call counter for the functor */
  inline void reset_n_calls(){ nCalls_=0; }

protected:
  int nCalls_;

};

//====================================================================

/**
 *  @class FunctorDouble
 *  @author James C. Sutherland
 *  @date   Feb, 2006
 *
 *  Provides functors for objects with functions that take a single double.
 */
template<class T1>
class FunctorDouble : public FunctorDoubleBase
{
public:
  FunctorDouble( T1* const obj,
		 double(T1::*f)(const double) )
    : FunctorDoubleBase(),
      obj_(obj),
      f_(f)
  {}

  inline double operator()(const double x)
  {
    ++nCalls_;
    return (obj_->*f_)(x);
  }

protected:
  T1* const obj_;
  double(T1::*f_)(const double);

};

//====================================================================

/**
 *  @class FunctorDoubleVec
 *  @author James C. Sutherland
 *  @date   Feb, 2006
 *
 *  Provides functors for objects with functions that take a vector
 *  of doubles.  At construction, the vector of inputs is provided,
 *  along with the index of the vector which will be populated when
 *  the functor is called.  The functor then calls through to the
 *  underlying function, providing the full vector as an argument.
 */
template<class T1>
class FunctorDoubleVec : public FunctorDoubleBase
{
public:
  /**
   *  @param obj : The object corresponding to the templated type for
   *  this functor.
   *
   *  @param f : the member function in obj that is to be called.
   *
   *  @param activeIndex : the index in the vector that is to be
   *  overwritten when the functor is called.
   *
   *  @param vals : the vector of values to provide to the underlying
   *  function.  When the functor is called, only the value
   *  corresponding to activeIndex will be changed.  The other values
   *  will be passed through.
   */
  FunctorDoubleVec<T1>( const T1* const obj,
			double(T1::*f)(const double*) const,
			const int activeIndex,
			const std::vector<double>& vals )
    : FunctorDoubleBase(),
      obj_(obj),
      f_(f),
      activeIndex_( activeIndex ),
      fInput_(vals)
  {}

  inline double operator()(const double x)
  {
    ++nCalls_;
    fInput_[activeIndex_] = x;
    return (obj_->*f_)(&fInput_[0]);
  }

  void reset_values( const std::vector<double> & vals ){ fInput_ = vals; }

private:
  const T1* const obj_;
  double(T1::*f_)(const double*) const;
  const int activeIndex_;
  std::vector<double> fInput_;
};

#endif  // MIXMDL_FUNCTOR
