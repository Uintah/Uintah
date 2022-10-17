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

#include <CCA/Components/Wasatch/Expressions/MomHat.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Wasatch.h>

template< typename FieldT, typename DirT>
MomHat<FieldT, DirT>::
MomHat(const Expr::Tag& partRHSTag,
       const Expr::Tag& momNTag,
       const Expr::Tag& volFracTag)
: Expr::Expression<FieldT>(),
  hasIntrusion_(volFracTag != Expr::Tag())
{
    this->set_gpu_runnable( true );
    const Expr::Tag dtTag = WasatchCore::TagNames::self().dt;
    const Expr::Tag rksTag = WasatchCore::TagNames::self().rkstage;
    dt_ = this->template create_field_request<SingleValue>(dtTag);
    rkstage_ = this->template create_field_request<SingleValue>(rksTag);
    momN_ = this->template create_field_request<FieldT>(momNTag);
    rhsPart_ = this->template create_field_request<FieldT>(partRHSTag);
    
    if( hasIntrusion_ )  volfrac_ = this->template create_field_request<FieldT>(volFracTag);
}

//--------------------------------------------------------------------

template< typename FieldT, typename DirT>
MomHat<FieldT, DirT>::
~MomHat()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DirT>
void
MomHat<FieldT, DirT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_ = opDB.retrieve_operator<Grad>();
  timeIntInfo_ = opDB.retrieve_operator<WasatchCore::TimeIntegrator>();
}

//--------------------------------------------------------------------

template< typename FieldT, typename DirT>
void
MomHat<FieldT, DirT>::
evaluate()
{

    const WasatchCore::TimeIntegrator& timeIntInfo = *timeIntInfo_;
    const double a2 = timeIntInfo.alpha[1];
    const double a3 = timeIntInfo.alpha[2];
    const double b2 = timeIntInfo.beta[1];
    const double b3 = timeIntInfo.beta[2];

    using namespace SpatialOps;
    FieldT& result = this->value();
    const FieldT& momN = momN_->field_ref();
    const FieldT&  rhsPart = rhsPart_->field_ref();
    const SingleValue& dt = dt_->field_ref();
    const SingleValue& rkStage = rkstage_->field_ref();
    
    result <<= cond ( rkStage == 1.0,     momN + dt * rhsPart             )
                    ( rkStage == 2.0,  a2*momN + b2*(result + dt*rhsPart) ) 
                    ( rkStage == 3.0,  a3*momN + b3*(result + dt*rhsPart) ) 
                    ( 0.0 ); // should never get here. 
    
    if( hasIntrusion_ ) result <<= volfrac_->field_ref() * result;
        
}

//--------------------------------------------------------------------
template< typename FieldT, typename DirT>
MomHat<FieldT, DirT>::
Builder::Builder(const Expr::Tag& result,
                 const Expr::Tag& partRHS,
                 const Expr::Tag& momN,
                 const Expr::Tag& volFracTag)
: ExpressionBuilder(result),
rhspartt_( partRHS ),
momNt_(momN),
volFract_(volFracTag)
{}
//--------------------------------------------------------------------

template< typename FieldT, typename DirT>
Expr::ExpressionBase*
MomHat<FieldT, DirT>::Builder::build() const
{
    return new MomHat<FieldT, DirT>(rhspartt_, momNt_,volFract_);
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class MomHat< SpatialOps::XVolField, SpatialOps::NODIR >;
template class MomHat< SpatialOps::YVolField, SpatialOps::NODIR >;
template class MomHat< SpatialOps::ZVolField, SpatialOps::NODIR >;
//==================================================================
