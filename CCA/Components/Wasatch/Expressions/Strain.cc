/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/Strain.h>

#include <spatialops/OperatorDatabase.h>

//====================================================================

template< typename StrainT, typename Vel1T, typename Vel2T >
Strain<StrainT,Vel1T,Vel2T>::
Strain( const Expr::Tag& vel1Tag,
        const Expr::Tag& vel2Tag )
  : Expr::Expression<StrainT>()
{
  this->set_gpu_runnable( true );
  
   u1_ = this->template create_field_request<Vel1T>(vel1Tag);
   u2_ = this->template create_field_request<Vel2T>(vel2Tag);
}

//--------------------------------------------------------------------

template< typename StrainT, typename Vel1T, typename Vel2T >
Strain<StrainT,Vel1T,Vel2T>::
~Strain()
{}

//--------------------------------------------------------------------

template< typename StrainT, typename Vel1T, typename Vel2T >
void
Strain<StrainT,Vel1T,Vel2T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  vel1GradOp_   = opDB.retrieve_operator<Vel1GradT  >();
  vel2GradOp_   = opDB.retrieve_operator<Vel2GradT  >();
}

//--------------------------------------------------------------------

template< typename StrainT, typename Vel1T, typename Vel2T >
void
Strain<StrainT,Vel1T,Vel2T>::
evaluate()
{
  using namespace SpatialOps;
  StrainT& strain = this->value();
  strain <<= 0.0; // avoid potential garbage in extra/ghost cells
  strain <<= 0.5 * ( (*vel1GradOp_)(u1_->field_ref()) + (*vel2GradOp_)(u2_->field_ref()) );
}

//--------------------------------------------------------------------

template< typename StrainT, typename Vel1T, typename Vel2T >
Strain<StrainT,Vel1T,Vel2T>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& vel1Tag,
                  const Expr::Tag& vel2Tag,
                  const Expr::Tag& dilTag )
  : ExpressionBuilder(result),
    vel1t_    ( vel1Tag ),
    vel2t_    ( vel2Tag )
{}

//--------------------------------------------------------------------

template< typename StrainT, typename Vel1T, typename Vel2T >
Expr::ExpressionBase*
Strain<StrainT,Vel1T,Vel2T>::Builder::build() const
{
  return new Strain<StrainT,Vel1T,Vel2T>( vel1t_, vel2t_ );
}


//====================================================================


template< typename StrainT, typename VelT >
Strain<StrainT,VelT,VelT>::
Strain( const Expr::Tag& velTag,
        const Expr::Tag& dilTag )
  : Expr::Expression<StrainT>()
{
  this->set_gpu_runnable( true );
  
   u_   = this->template create_field_request<VelT     >(velTag);
   dil_ = this->template create_field_request<SVolField>(dilTag);
}

//--------------------------------------------------------------------

template< typename StrainT, typename VelT >
Strain<StrainT,VelT,VelT>::
~Strain()
{}

//--------------------------------------------------------------------

template< typename StrainT, typename VelT >
void
Strain<StrainT,VelT,VelT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  svolInterpOp_ = opDB.retrieve_operator<SVol2StrainInterpT>();
  velGradOp_    = opDB.retrieve_operator<VelGradT          >();
}

//--------------------------------------------------------------------

template< typename StrainT, typename VelT >
void
Strain<StrainT,VelT,VelT>::
evaluate()
{
  using namespace SpatialOps;
  StrainT& strain = this->value();
  strain <<= (*velGradOp_)(u_->field_ref()) - 1.0/3.0*((*svolInterpOp_)(dil_->field_ref()));
}

//--------------------------------------------------------------------

template< typename StrainT, typename VelT >
Strain<StrainT,VelT,VelT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& vel1Tag,
                  const Expr::Tag& vel2Tag,
                  const Expr::Tag& dilTag )
  : ExpressionBuilder(result),
    velt_ ( vel1Tag ),
    dilt_ ( dilTag  )
{}

//--------------------------------------------------------------------

template< typename StrainT, typename VelT >
Expr::ExpressionBase*
Strain<StrainT,VelT,VelT>::Builder::build() const
{
  return new Strain<StrainT,VelT,VelT>( velt_, dilt_ );
}

//====================================================================


//====================================================================
// Explicit template instantiation
#include <spatialops/structured/FVStaggered.h>
#define DECLARE_Strain( VOL )	\
  template class Strain< SpatialOps::FaceTypes<VOL>::XFace,	\
                         VOL,					\
                         SpatialOps::XVolField >;               \
  template class Strain< SpatialOps::FaceTypes<VOL>::YFace,	\
                         VOL,					\
                         SpatialOps::YVolField >;               \
  template class Strain< SpatialOps::FaceTypes<VOL>::ZFace,	\
                         VOL,					\
                         SpatialOps::ZVolField>;

DECLARE_Strain( SpatialOps::XVolField );
DECLARE_Strain( SpatialOps::YVolField );
DECLARE_Strain( SpatialOps::ZVolField );
//====================================================================
