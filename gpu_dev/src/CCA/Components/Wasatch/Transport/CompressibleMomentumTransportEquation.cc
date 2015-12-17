/*
 * The MIT License
 *
 * Copyright (c) 2015 The University of Utah
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

#include <expression/Expression.h>

#include <CCA/Components/Wasatch/Transport/CompressibleMomentumTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ParseEquation.h>
#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>

namespace WasatchCore{



  //============================================================================

  template <typename MomDirT>
  CompressibleMomentumTransportEquation<MomDirT>::
  CompressibleMomentumTransportEquation( const Direction momComponent,
                                         const std::string velName,
                                         const std::string momName,
                                         const Expr::Tag densityTag,
                                         const Expr::Tag temperatureTag,
                                         const Expr::Tag mixMWTag,
                                         const double gasConstant,
                                         const Expr::Tag bodyForceTag,
                                         const Expr::Tag srcTermTag,
                                         GraphCategories& gc,
                                         Uintah::ProblemSpecP params,
                                         TurbulenceParameters turbParams )
  : MomentumTransportEquationBase<SVolField>(momComponent,
                                          velName,
                                          momName,
                                          densityTag,
                                          false,
                                          bodyForceTag,
                                          srcTermTag,
                                          gc,
                                          params,
                                          turbParams)
  {
    // todo:
    //  - strain tensor       // registered by MomentumTransportEquationBase
    //  - convective flux     // registered by the MomentumTransportEquationBase
    //  - turbulent viscosity // SHOULD be registered by the MomentumTransportEquationBase. NOT READY YET.
    //  - buoyancy? //

    Expr::ExpressionFactory& factory = *gc[ADVANCE_SOLUTION]->exprFactory;

    typedef IdealGasPressure<FieldT>::Builder Pressure;
    if (!factory.have_entry(TagNames::self().pressure)) {
      factory.register_expression( scinew Pressure(TagNames::self().pressure,
                                                   densityTag,
                                                   temperatureTag,
                                                   mixMWTag,
                                                   gasConstant) );

    }

    setup();
  }

  //----------------------------------------------------------------------------
  
  template <typename MomDirT>
  Expr::ExpressionID  CompressibleMomentumTransportEquation<MomDirT>::
  setup_rhs( FieldTagInfo&,
            const Expr::TagList& srcTags )
  {
    
    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    Expr::Tag volFracTag = vNames.vol_frac_tag<FieldT>();
    
    Expr::ExpressionFactory& factory = *this->gc_[ADVANCE_SOLUTION]->exprFactory;

    typedef typename MomRHS<SVolField, MomDirT>::Builder RHS;
    return factory.register_expression( scinew RHS( this->rhsTag_,
                                                   this->pressureTag_,
                                                   rhs_part_tag(this->solnVarTag_),
                                                   volFracTag ) );
  }

  //----------------------------------------------------------------------------

  template <typename MomDirT>
  CompressibleMomentumTransportEquation<MomDirT>::
  ~CompressibleMomentumTransportEquation()
  {}

  //============================================================================

  template class CompressibleMomentumTransportEquation< SpatialOps::XDIR>;
  template class CompressibleMomentumTransportEquation< SpatialOps::YDIR>;
  template class CompressibleMomentumTransportEquation< SpatialOps::ZDIR>;
} // namespace Wasatch



