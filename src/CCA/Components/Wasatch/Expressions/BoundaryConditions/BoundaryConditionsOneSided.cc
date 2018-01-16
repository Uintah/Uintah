/*
 * The MIT License
 *
 * Copyright (c) 2018 The University of Utah
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

/**
 *  \file   BoundaryConditionsOneSided.cc
 *  \date   Jul 10, 2017
 *  \author james
 */

#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionsOneSided.h>
#include <spatialops/structured/SpatialMask.h>
#include <spatialops/NeboStencilBuilder.h>

namespace WasatchCore{

  // ###################################################################
  template< typename FieldT, typename DivOpT >
  void
  BCOneSidedConvFluxDiv<FieldT, DivOpT>::
  bind_operators(const SpatialOps::OperatorDatabase& opdb)
  {
    divOp_ = opdb.retrieve_operator<DivOpT>();
  }

  //---------------------------------------------------------------------
  template< typename FieldT, typename DivOpT >
  void
  BCOneSidedConvFluxDiv<FieldT, DivOpT>::
  evaluate()
  {
    FieldT& result = this->value();
    const FieldT& phi = phi_->field_ref();
    const FieldT& vel = u_->field_ref();
    masked_assign( *this->interiorSvolSpatialMask_, result, result - 1.0 * (*divOp_)( phi*vel) );
    masked_assign( *this->spatialMask_, result, 0.0 );
  }

  // ###################################################################
  template< typename FieldT, typename GradOpT >
  void
  BCOneSidedGradP<FieldT, GradOpT>::
  bind_operators(const SpatialOps::OperatorDatabase& opdb)
  {
    gradOp_ = opdb.retrieve_operator<GradOpT>();
  }

  //---------------------------------------------------------------------
  template< typename FieldT, typename GradOpT >
  void
  BCOneSidedGradP<FieldT, GradOpT>::
  evaluate()
  {
    FieldT& result = this->value();
    const FieldT& p = p_->field_ref();
    masked_assign( *this->interiorSvolSpatialMask_, result, result - 1.0 * (*gradOp_)( p ) );
    masked_assign( *this->spatialMask_, result, 0.0 );
  }

  // ###################################################################
  template< typename FieldT >
  void
  OneSidedDirichletBC<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;

    if( this->spatialMask_ ){
      FieldT& lhs =  this->value();
      masked_assign(*this->spatialMask_, lhs, bcValue_);
      masked_assign(convert<FieldT>(*this->spatialMask_, this->shiftSide_), lhs, bcValue_);
    }
  }

  // ###################################################################
  // EXPLICIT INSTANTIATION
#include <CCA/Components/Wasatch/FieldTypes.h>

  template class OneSidedDirichletBC<SVolField>;
  template class OneSidedDirichletBC<XVolField>;
  template class OneSidedDirichletBC<YVolField>;
  template class OneSidedDirichletBC<ZVolField>;

#define INSTANTIATE_ONE_SIDED_BC(VOLT,DIRT,B,C)                                                                                                                            \
    typedef typename SpatialOps::OneSidedOpTypeBuilder<SpatialOps::Gradient,SpatialOps::OneSidedStencil3<typename SpatialOps::UnitTriplet<DIRT>::type::Negate>,VOLT>::type B;\
    typedef typename SpatialOps::OneSidedOpTypeBuilder<SpatialOps::Gradient,SpatialOps::OneSidedStencil3<SpatialOps::UnitTriplet<DIRT>::type>,VOLT>::type C;                 \
    template class BCOneSidedConvFluxDiv<VOLT,B>;                                                                                                                            \
    template class BCOneSidedGradP      <VOLT,B>;                                                                                                                                  \
    template class BCOneSidedConvFluxDiv<VOLT,C>;                                                                                                                            \
    template class BCOneSidedGradP      <VOLT,C>;

  INSTANTIATE_ONE_SIDED_BC(SpatialOps::SVolField, SpatialOps::XDIR, BX, CX);
  INSTANTIATE_ONE_SIDED_BC(SpatialOps::SVolField, SpatialOps::YDIR, BY, CY);
  INSTANTIATE_ONE_SIDED_BC(SpatialOps::SVolField, SpatialOps::ZDIR, BZ, CZ);

} // namespace WasatchCore
