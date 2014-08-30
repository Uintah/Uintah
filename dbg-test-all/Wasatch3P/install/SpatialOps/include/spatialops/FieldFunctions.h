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

#ifndef FieldFunctions_h
#define FieldFunctions_h

#include <spatialops/SpatialOpsConfigure.h>

#include <boost/static_assert.hpp>

#include <spatialops/SpatialOpsTools.h>
#include <spatialops/Nebo.h>

#include<cmath>


//====================================================================

namespace SpatialOps{


  //==================================================================

  namespace FFLocal{
    struct NULLPatch
    {
      struct FieldID{ FieldID(){} };
    };
  }

  //==================================================================


/**
 *  @class FieldFunction3D
 *  @author James C. Sutherland
 *  @date  September, 2007
 *
 *  @brief Supports implementation of 3D functions to assign values to
 *  SpatialField objects.
 *
 *  @par Template Parameters
 *
 *   <ul>
 *
 *   <li> \b FieldT The type of field that this function applies to.
 *
 *   <li> \b PatchT The patch type (if applicable) that this field
 *   lives on.  If no patch is specified, then field references for
 *   the spatial coordinates must be provded at construction.
 *
 *   \b PatchT must define the FieldID type for field identifiers.  It
 *   must also provide access to FieldManager objects.
 *
 *   Note that \b PatchT is an optional template parameter.  In the
 *   case where a patch is not being used, or where field references
 *   are available at construction, this template parameter may be
 *   omitted.
 *
 *   </ul>
 */
template<typename FieldT, typename PatchT=FFLocal::NULLPatch>
class FieldFunction3D
{
  typedef typename PatchT::FieldID  FieldID;
public:

  typedef FieldT  FieldType;
  typedef PatchT  PatchType;

  virtual void evaluate( FieldT& phi ) const =0;

  virtual void dx( FieldT& gradPhi ) const{ assert(0); }
  virtual void dy( FieldT& gradPhi ) const{ assert(0); }
  virtual void dz( FieldT& gradPhi ) const{ assert(0); }

  virtual void d2x( FieldT& gradPhi ) const{ assert(0); }
  virtual void d2y( FieldT& gradPhi ) const{ assert(0); }
  virtual void d2z( FieldT& gradPhi ) const{ assert(0); }


  const FieldT& get_x() const{set_fields(); return *x_;}
  const FieldT& get_y() const{set_fields(); return *y_;}
  const FieldT& get_z() const{set_fields(); return *z_;}

  virtual ~FieldFunction3D(){}

protected:

  /** Use this constructor when the PatchT was not specified. */
  FieldFunction3D( const FieldT& x, const FieldT& y, const FieldT& z );

  /** Use this constructor when the PatchT was specified. */
  FieldFunction3D( PatchT& p, const FieldID xid, const FieldID yid, const FieldID zid );

  /**
   *  In the case where field ids were provided and this is built
   *  using a patch, then this binds fields.  It must be called before
   *  using any of the fields.
   */
  void set_fields() const;

private:
  PatchT* const patch_;
  const FieldID xid_, yid_, zid_;
  mutable const FieldT* x_;
  mutable const FieldT* y_;
  mutable const FieldT* z_;
};

//====================================================================

/**
 *  @class FieldFunction2D
 *  @author James C. Sutherland
 *  @date  September, 2007
 *
 *  @brief Supports implementation of 2D functions to assign values to
 *  SpatialField objects.
 *
 *  See documentation on FieldFunction3D for more information about
 *  template parameters.
 */
template<typename FieldT, typename PatchT=FFLocal::NULLPatch>
class FieldFunction2D
{
  typedef typename PatchT::FieldID  FieldID;
public:

  typedef FieldT  FieldType;
  typedef PatchT  PatchType;

  virtual void evaluate( FieldT& phi ) const =0;


  virtual void dx( FieldT& gradPhi ) const{ const bool implemented=false; assert(implemented); }
  virtual void dy( FieldT& gradPhi ) const{ const bool implemented=false; assert(implemented); }

  virtual void d2x( FieldT& gradPhi ) const{ const bool implemented=false; assert(implemented); }
  virtual void d2y( FieldT& gradPhi ) const{ const bool implemented=false; assert(implemented); }


  const FieldT& get_x() const{set_fields(); return *x_;}
  const FieldT& get_y() const{set_fields(); return *y_;}

  virtual ~FieldFunction2D(){}

protected:

  FieldFunction2D( const FieldT& x, const FieldT& y );
  FieldFunction2D( PatchT& p, const FieldID xid, const FieldID yid, const FieldID zid );

  void set_fields() const;

private:
  PatchT* const patch_;
  const FieldID xid_, yid_;
  mutable const FieldT* x_;
  mutable const FieldT* y_;
};

//====================================================================

/**
 *  @class FieldFunction1D
 *  @author James C. Sutherland
 *  @date  September, 2007
 *
 *  @brief Supports implementation of 1D functions to assign values to
 *  SpatialField objects.
 *
 *  See documentation on FieldFunction3D for more information about
 *  template parameters.
 */
template< typename FieldT, typename PatchT=FFLocal::NULLPatch >
class FieldFunction1D
{
  typedef typename PatchT::FieldID  FieldID;
public:

  virtual void evaluate( FieldT& f ) const =0;

  virtual void dx ( FieldT& gradPhi ) const{ const bool implemented=false; assert(implemented); }
  virtual void d2x( FieldT& gradPhi ) const{ const bool implemented=false; assert(implemented); }

  const FieldT& get_x() const{set_fields(); return *x_;}

  virtual ~FieldFunction1D(){}

protected:

  FieldFunction1D( const FieldT& x );
  FieldFunction1D( PatchT& p, const FieldID xid );

  void set_fields() const;

private:
  PatchT* const patch_;
  const FieldID xid_;
  mutable const FieldT* x_;
};


//====================================================================


/**
 *   NOTE: this populates ghost fields as well.
 */
template< typename FieldT,
          typename PatchT=FFLocal::NULLPatch >
class LinearFunction1D : public FieldFunction1D<FieldT,PatchT>
{
  typedef typename PatchT::FieldID  FieldID;
public:

  LinearFunction1D( const FieldT& x,
                    const double slope,
                    const double intercept )
    : FieldFunction1D<FieldT,PatchT>(x),
      m_(slope),
      b_(intercept)
  {}

  LinearFunction1D( PatchT& p,
                    const FieldID xid,
                    const double slope,
                    const double intercept )
    : FieldFunction1D<FieldT,PatchT>(p,xid), m_(slope), b_(intercept)
  {}

  ~LinearFunction1D(){}

  void evaluate( FieldT& f ) const;
  void dx( FieldT& gradPhi ) const;
  void d2x( FieldT& gradPhi ) const;

protected:

private:
  const double m_, b_;
};


//====================================================================

/**
  *  @class SinFunction
 *  @author James C. Sutherland
 *  @date September, 2007
 *
 *  @brief Evaluates a*sin(b*x), where a is the amplitude and b is the
 *  period of the sin function.
 */
template< typename FieldT,
          typename PatchT=FFLocal::NULLPatch >
class SinFunction : public FieldFunction1D<FieldT,PatchT>
{
  typedef typename PatchT::FieldID FieldID;
 public:
  SinFunction( PatchT& p,
               const FieldID xid,
               const double amplitude,
               const double period,
               const double base=0.0 );
  SinFunction( const FieldT& x,
               const double amplitude,
               const double period,
               const double base=0.0 );
  ~SinFunction(){}
  void evaluate( FieldT& f ) const;
  void dx( FieldT& df ) const;
  void d2x( FieldT& d2f ) const;

 private:
  const double a_, b_, base_;
};

//====================================================================

/**
 *  @class GaussianFunction
 *  @author James C. Sutherland
 *  @date October, 2008
 *  @brief Evaluates a gaussian function,
 *  \f$
 *   f(x) = y_0 + a \exp\left( \frac{\left(x-x_0\right)^2 }{2\sigma^2} \right)
 *  \f$
 */
template< typename FieldT,
          typename PatchT=FFLocal::NULLPatch >
class GaussianFunction : public FieldFunction1D<FieldT,PatchT>
{
  typedef typename PatchT::FieldID FieldID;
 public:
  GaussianFunction( PatchT& p,
                    const FieldID xid,
                    const double a,
                    const double stddev,
                    const double mean,
                    const double yo=0.0 );
  GaussianFunction( const FieldT& x,
                    const double stddev,
                    const double mean,
                    const double xo,
                    const double yo=0.0 );
  ~GaussianFunction(){}
  void evaluate( FieldT& f ) const;
  void dx( FieldT& df ) const;
  void d2x( FieldT& d2f ) const;

 private:
  const double a_, b_, xo_, yo_;
};

//====================================================================

//====================================================================

/**
 *  @class HyperTanFunction
 *  @author Naveen Punati
 *  @date January, 2009
 *  @brief Hyperbolic tangent Function
 *
 * Implements the function
 *  \f[
 *     f(x) = \frac{A}{4} \left(1+\tanh\left(\frac{x-L_1}{w}\right)\right) \left(1-\tanh\left(\frac{x-L_2}{w}\right)\right)
 *  \f]
 * with
 *  - \f$A\f$ Amplitude of the function
 *  - \f$w\f$ width of the transition
 * - \f$L_1\f$ and \f$L_2\f$ Midpoint of transition from low to high
      (L1) and high to low (L2).
*/
template< typename FieldT,
          typename PatchT=FFLocal::NULLPatch >
class HyperTanFunction : public FieldFunction1D<FieldT,PatchT>
{
  typedef typename PatchT::FieldID FieldID;
 public:
  HyperTanFunction( const FieldT& x,
                    const double amplitude,
                    const double width,
                    const double L1,
                    const double L2 );
  ~HyperTanFunction(){}
 void evaluate( FieldT& f ) const;
 private:
  const double amplitude_, width_, L1_, L2_;
};

//====================================================================



// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//                           Implementations
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







//====================================================================


namespace FFLocal{
  template<typename FieldT, typename PatchT>
  struct FieldSetter
  {
    static void set( const FieldT* const f, PatchT* p, typename PatchT::FieldID id )
    {
      f = p->template field_manager<FieldT>().field_ref(id);
    }
  };

//====================================================================

  template<typename FieldT>
  struct FieldSetter<FieldT,NULLPatch>
  {
    static void set( const FieldT* const f, NULLPatch* p, typename NULLPatch::FieldID id ){}
  };
}


//====================================================================


//--------------------------------------------------------------------
template<typename FieldT,typename PatchT>
FieldFunction1D<FieldT,PatchT>::FieldFunction1D( const FieldT& x )
  : patch_( NULL ),
    x_( &x )
{}
//--------------------------------------------------------------------
template<typename FieldT,typename PatchT>
FieldFunction1D<FieldT,PatchT>::FieldFunction1D( PatchT& p, const FieldID xid )
  : patch_( &p ),
    xid_( xid )
{
  BOOST_STATIC_ASSERT( bool(!IsSameType<PatchT,FFLocal::NULLPatch>::result) );
  x_ = NULL;
}
//--------------------------------------------------------------------
template<typename FieldT,typename PatchT>
void
FieldFunction1D<FieldT,PatchT>::set_fields() const
{
  if(x_==NULL){
    FFLocal::FieldSetter<FieldT,PatchT>::set( x_, patch_, xid_ );
  }
}
//--------------------------------------------------------------------


//====================================================================


//--------------------------------------------------------------------
template<typename FieldT,typename PatchT>
FieldFunction2D<FieldT,PatchT>::FieldFunction2D( const FieldT& x, const FieldT& y )
  : patch_( NULL ),
    x_(&x), y_(&y)
{
}
//--------------------------------------------------------------------
template<typename FieldT,typename PatchT>
FieldFunction2D<FieldT,PatchT>::FieldFunction2D( PatchT& p, const FieldID xid, const FieldID yid, const FieldID zid )
  : patch_(&p),
  xid_(xid), yid_(yid)
{
  BOOST_STATIC_ASSERT( bool(!IsSameType<PatchT,FFLocal::NULLPatch>::result) );
  x_ = y_ = NULL;
}
//--------------------------------------------------------------------
template<typename FieldT,typename PatchT>
void
FieldFunction2D<FieldT,PatchT>::set_fields() const
{
  if(x_==NULL){
    FFLocal::FieldSetter<FieldT,PatchT>::set( x_, patch_, xid_ );
    FFLocal::FieldSetter<FieldT,PatchT>::set( y_, patch_, yid_ );
  }
}
//--------------------------------------------------------------------


//====================================================================


//--------------------------------------------------------------------
template<typename FieldT,typename PatchT>
FieldFunction3D<FieldT,PatchT>::FieldFunction3D( const FieldT& x, const FieldT& y, const FieldT& z )
  : patch_( NULL ),
    xid_(),   yid_(),   zid_(),
    x_( &x ), y_( &y ), z_( &z )
{}
//--------------------------------------------------------------------
template<typename FieldT,typename PatchT>
FieldFunction3D<FieldT,PatchT>::FieldFunction3D( PatchT& p, const FieldID xid, const FieldID yid, const FieldID zid )
  : patch_( &p ),
    xid_(xid), yid_(yid), zid_(zid)
{
  BOOST_STATIC_ASSERT( bool(!IsSameType<PatchT,FFLocal::NULLPatch>::result) );
  x_ = y_ = z_ = NULL;
}
//--------------------------------------------------------------------
template<typename FieldT,typename PatchT>
void
FieldFunction3D<FieldT,PatchT>::set_fields() const
{
  if(x_==NULL){
    FFLocal::FieldSetter<FieldT,PatchT>::set( x_, patch_, xid_ );
    FFLocal::FieldSetter<FieldT,PatchT>::set( y_, patch_, yid_ );
    FFLocal::FieldSetter<FieldT,PatchT>::set( z_, patch_, zid_ );
  }
}


//====================================================================


//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
LinearFunction1D<FieldT,PatchT>::
evaluate( FieldT& f ) const
{
  this->set_fields();
  const FieldT& x = this->get_x();
  f <<= x*m_ + b_;
}
//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
LinearFunction1D<FieldT,PatchT>::
dx( FieldT& f ) const
{
  f <<= m_;
}
//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
LinearFunction1D<FieldT,PatchT>::
d2x( FieldT& f ) const
{
  f <<= 0.0;
}
//--------------------------------------------------------------------


//====================================================================


//--------------------------------------------------------------------
template<typename FieldT, typename PatchT>
SinFunction<FieldT,PatchT>::
SinFunction( const FieldT& x,
             const double a,
             const double b,
             const double c )
  : FieldFunction1D<FieldT,PatchT>(x),
    a_( a ),
    b_( b ),
    base_( c )
{
}
//--------------------------------------------------------------------
template<typename FieldT, typename PatchT>
SinFunction<FieldT,PatchT>::
SinFunction( PatchT& p,
             const FieldID xid,
             const double a,
             const double b,
             const double c )
  : FieldFunction1D<FieldT,PatchT>( p, xid ),
    a_( a ),
    b_( b ),
    base_( c )
{
}
//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
SinFunction<FieldT,PatchT>::
evaluate( FieldT& f ) const
{
  this->set_fields();
  f <<= a_ * sin( this->get_x() * b_ ) + base_;
}
//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
SinFunction<FieldT,PatchT>::
dx( FieldT& f ) const
{
  this->set_fields();
  const FieldT& x = this->get_x();
  f <<= a_*b_ * cos(x*b_);
}
//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
SinFunction<FieldT,PatchT>::
d2x( FieldT& f ) const
{
  this->set_fields();
  const FieldT& x = this->get_x();
  f <<= -a_*b_*b_ * sin(x*b_);
}
//--------------------------------------------------------------------


//====================================================================


//--------------------------------------------------------------------
template<typename FieldT, typename PatchT>
GaussianFunction<FieldT,PatchT>::
GaussianFunction( const FieldT& x,
                  const double a,
                  const double b,
                  const double xo,
                  const double yo )
  : FieldFunction1D<FieldT,PatchT>(x),
    a_( a ),
    b_( 2.0*b*b ),  // b=2*sigma^2
    xo_( xo ),
    yo_( yo )
{
}
//--------------------------------------------------------------------
template<typename FieldT, typename PatchT>
GaussianFunction<FieldT,PatchT>::
GaussianFunction( PatchT& p,
                  const FieldID xid,
                  const double a,
                  const double b,
                  const double xo,
                  const double yo )
  : FieldFunction1D<FieldT,PatchT>( p, xid ),
    a_( a ),
    b_( 2.0*b*b ), // b=2*sigma^2
    xo_( xo ),
    yo_( yo )
{
}
//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
GaussianFunction<FieldT,PatchT>::
evaluate( FieldT& f ) const
{
  this->set_fields();
  const FieldT& x = this->get_x();
  f <<= a_*exp(-1.0/b_ * (x-xo_)*(x-xo_) ) + yo_;
}
//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
GaussianFunction<FieldT,PatchT>::
dx( FieldT& f ) const
{
  this->set_fields();
  const FieldT& x = this->get_x();
  f <<= -2/b_ * (x-xo_)*(x-xo_)*exp(-1.0/b_ * (x-xo_)*(x-xo_) ) + yo_;
}
//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
GaussianFunction<FieldT,PatchT>::
d2x( FieldT& f ) const
{
  this->set_fields();
  const FieldT& x = this->get_x();
  f <<= -2/b_ * ( a_*exp(-1.0/b_ * (x-xo_)*(x-xo_) ) ) * (1.0-2.0/b_*(x-xo_)*(x-xo_));
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
template<typename FieldT, typename PatchT>
HyperTanFunction<FieldT,PatchT>::
HyperTanFunction( const FieldT& x,
                  const double amplitude,
                  const double width,
                  const double L1,
                  const double L2 )
  : FieldFunction1D<FieldT,PatchT>(x),
    amplitude_( amplitude ),
    width_( width ),
    L1_( L1 ),
    L2_( L2 )
{
}
//------------------------------------------------------------------
template<typename FieldT, typename PatchT>
void
HyperTanFunction<FieldT,PatchT>::
evaluate( FieldT& f ) const
{
  this->set_fields();
  const FieldT& x = this->get_x();
  f <<= (amplitude_/2) * ( 1.0 + tanh((x-L1_)/width_)) * (1.0-0.5*(1.0+tanh((x-L2_)/width_)));
}
//------------------------------------------------------------------


} // namespace SpatialOps

#endif
