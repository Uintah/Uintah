/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#ifndef AreaFraction_Expr_h
#define AreaFraction_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>

/**
 *  \class   AreaFraction
 *  \author  Tony Saad
 *  \date    December, 2012
 *  \ingroup Expressions
 *
 *  \brief The AreaFraction expression provides an efficient and robust procedure
 to calculate x, y, and z area fractions for sharp-interface embedded geometries.
 By sharp interface we mean that no cells are partially filled by a solid, e.g. 
 stairstepping. Given a cell centered volume fraction field that represents sharp
 embedded geometry, the AreaFraction expression calculates the corresponding area
 fractions at the boundaries of the embedded geometry. These are also referred to
 as staggered volume fractions, i.e. xvolFraction, yvolFraction, and zvolFraction.
 The process of calculating the area fractions uses a clean algebraic operation
 on the svolFraction field. The procedure is as follows. Given \f$\phi\f$ as
 the cell centered fluid volume fraction such that \f$\phi = 0\f$ inside a solid and
 \f$\phi = 1\f$ outside the solid, our objective is to calculate the x,y,z aera fractions.
 we first interpolate \f$\phi\f$ to the corresponding staggered direction. 
 This will result in values of 0.5 at the fluid-solid interface. Because we only 
 deal with sharp interfaces in Wasatch, we set all those values (i.e. 0.5) to 0.
 */
template< typename DestT >
class AreaFraction
: public Expr::Expression<DestT>
{
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, DestT >::type InpterpSrcT2DestT;
  
  DECLARE_FIELD(SVolField, src_)
  const bool valid_;
  const InpterpSrcT2DestT* interpSrcT2DestTOp_;
  
  AreaFraction( const Expr::Tag& srctag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param srctag  Tag of the source field
     *  \param desttag Tag of the destination field
     */
    Builder( const Expr::Tag& result,
            const Expr::Tag& srctag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag srct_;
  };
  
  ~AreaFraction();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};


template< typename DestT >
AreaFraction<DestT>::
AreaFraction( const Expr::Tag& srctag )
: Expr::Expression<DestT>(),
valid_(srctag != Expr::Tag())
{
  this->set_gpu_runnable( true );
  
  if (valid_)  src_ = this->template create_field_request<SVolField>(srctag);
}

//--------------------------------------------------------------------

template<typename DestT >
AreaFraction<DestT>::
~AreaFraction()
{}

//--------------------------------------------------------------------

template<typename DestT >
void
AreaFraction<DestT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( valid_ ) interpSrcT2DestTOp_ = opDB.retrieve_operator<InpterpSrcT2DestT>();
}

//--------------------------------------------------------------------

template<typename DestT >
void
AreaFraction<DestT>::
evaluate()
{
  using namespace SpatialOps;
  using SpatialOps::operator<<=;
  if (!valid_) return;
  const SVolField& src = src_->field_ref();
  DestT& destResult = this->value();
  destResult <<= 1.0; // this will ensure that the boundaries have an area fraction of 1.0. Wall BCs are NOT handled by volume fractions rather by the BCHelperTools.
  destResult <<= cond ( (*interpSrcT2DestTOp_)(src) < 1.0, 0.0 )
                      ( 1.0 );
}

//--------------------------------------------------------------------

template<typename DestT >
AreaFraction<DestT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& srctag )
: ExpressionBuilder(result),
  srct_( srctag )
{}

//--------------------------------------------------------------------

template<typename DestT >
Expr::ExpressionBase*
AreaFraction<DestT>::Builder::build() const
{
  return new AreaFraction<DestT>( srct_ );
}

//--------------------------------------------------------------------

#endif // AreaFraction_Expr_h
