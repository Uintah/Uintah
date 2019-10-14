/**
 *  \file   EnthalpyTransportEquation.cc
 *  \date   Nov 12, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2018 The University of Utah
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
 *
 */
#include <CCA/Components/Wasatch/Transport/EnthalpyTransportEquation.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>

namespace WasatchCore {

  class EnthDiffCoeff
   : public Expr::Expression<SVolField>
  {
    DECLARE_FIELDS(SVolField, thermCond_, cp_, turbVisc_)
    const double turbPr_;
    const bool isTurbulent_;

    EnthDiffCoeff( const Expr::Tag& thermCondTag,
                   const Expr::Tag& viscosityTag,
                   const Expr::Tag& turbViscTag,
                   const double turbPr )
    : Expr::Expression<SVolField>(),
      turbPr_      ( turbPr ),
      isTurbulent_ ( turbViscTag != Expr::Tag() )
    {
       thermCond_ = create_field_request<SVolField>(thermCondTag);
       cp_ = create_field_request<SVolField>(viscosityTag);
      if(isTurbulent_)  turbVisc_ = create_field_request<SVolField>(turbViscTag);
    }

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  @brief Build a EnthDiffCoeff expression
       *  @param resultTag the tag for the value that this expression computes
       *  @param thermCondTag the tag for the thermal conductivity
       *  @param cpTag the tag for the specific heat
       *  @param turbViscTag the tag for the turbulent viscosity.  If empty, no turbulent contribution is included.
       *  @param turbPrandtl the turbulent Prandtl number.
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& thermCondTag,
               const Expr::Tag& cpTag,
               const Expr::Tag& turbViscTag,
               const double turbPrandtl = 0.7 )
      : ExpressionBuilder( resultTag ),
        thermCondTag_( thermCondTag ),
        cpTag_       ( cpTag ),
        turbViscTag_ ( turbViscTag  ),
        turbPr_      ( turbPrandtl  )
      {}

      Expr::ExpressionBase* build() const{
        return new EnthDiffCoeff( thermCondTag_,cpTag_,turbViscTag_,turbPr_ );
      }

    private:
      const Expr::Tag thermCondTag_, cpTag_, turbViscTag_;
      const double turbPr_;
    };

    ~EnthDiffCoeff(){}

    void evaluate(){
      using namespace SpatialOps;
      SVolField& result = this->value();
      const SVolField& thermCond = thermCond_->field_ref();
      const SVolField& cp = cp_->field_ref();
      if( isTurbulent_ ) result <<= thermCond / cp + turbVisc_->field_ref()/turbPr_;
      else               result <<= thermCond / cp;
    }
  };

  //===========================================================================

  EnthalpyTransportEquation::
  EnthalpyTransportEquation( const std::string enthalpyName,
                             Uintah::ProblemSpecP params,
                             GraphCategories& gc,
                             const Expr::Tag densityTag,
                             const TurbulenceParameters& turbulenceParams,
                             std::set<std::string>& persistentFields )
  : ScalarTransportEquation<SVolField>( enthalpyName, params, gc, densityTag, turbulenceParams, persistentFields ),
    diffCoeffTag_( enthalpyName+"_diffusivity", Expr::STATE_NONE )
  {
    const Expr::Tag turbViscTag = enableTurbulence_ ? TagNames::self().turbulentviscosity : Expr::Tag();
    const Expr::Tag lambdaTag( parse_nametag( params->findBlock("ThermalConductivity")->findBlock("NameTag") ) );
    const Expr::Tag cpTag    ( parse_nametag( params->findBlock("HeatCapacity"       )->findBlock("NameTag") ) );
    gc[ADVANCE_SOLUTION]->exprFactory->register_expression(
        scinew EnthDiffCoeff::Builder( diffCoeffTag_, lambdaTag, cpTag, turbViscTag, turbulenceParams.turbPrandtl )
    );

    setup();
  }

  //---------------------------------------------------------------------------

  EnthalpyTransportEquation::~EnthalpyTransportEquation()
  {}

  //---------------------------------------------------------------------------

  void
  EnthalpyTransportEquation::
  setup_diffusive_flux( FieldTagInfo& info )
  {
    typedef DiffusiveVelocity< SpatialOps::SSurfXField >::Builder XFlux;
    typedef DiffusiveVelocity< SpatialOps::SSurfYField >::Builder YFlux;
    typedef DiffusiveVelocity< SpatialOps::SSurfZField >::Builder ZFlux;

    // jcs how will we determine if we have each direction on???
    const bool doX=true, doY=true, doZ=true;

    const TagNames& tagNames = TagNames::self();
    const Expr::Tag xDiffFluxTag( solnVarName_ + tagNames.diffusiveflux + "x", Expr::STATE_NONE );
    const Expr::Tag yDiffFluxTag( solnVarName_ + tagNames.diffusiveflux + "y", Expr::STATE_NONE );
    const Expr::Tag zDiffFluxTag( solnVarName_ + tagNames.diffusiveflux + "z", Expr::STATE_NONE );

    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    if( doX ){ info[DIFFUSIVE_FLUX_X]=xDiffFluxTag; factory.register_expression( scinew XFlux( xDiffFluxTag, primVarTag_, diffCoeffTag_, turbDiffTag_ ) ); }
    if( doY ){ info[DIFFUSIVE_FLUX_Y]=yDiffFluxTag; factory.register_expression( scinew YFlux( yDiffFluxTag, primVarTag_, diffCoeffTag_, turbDiffTag_ ) ); }
    if( doZ ){ info[DIFFUSIVE_FLUX_Z]=zDiffFluxTag; factory.register_expression( scinew ZFlux( zDiffFluxTag, primVarTag_, diffCoeffTag_, turbDiffTag_ ) ); }

    // if doing convection, we will likely have a pressure solve that requires
    // predicted scalar values to approximate the density time derivatives
    if( params_->findBlock("ConvectiveFlux") ){
      const std::string& suffix = tagNames.star;
      const Expr::Tag xFluxTagNew  ( solnVarName_ + suffix + tagNames.diffusiveflux + "x", Expr::STATE_NONE    );
      const Expr::Tag yFluxTagNew  ( solnVarName_ + suffix + tagNames.diffusiveflux + "y", Expr::STATE_NONE    );
      const Expr::Tag zFluxTagNew  ( solnVarName_ + suffix + tagNames.diffusiveflux + "z", Expr::STATE_NONE    );
      const Expr::Tag primVarTagNew( primVarTag_.name()  + suffix, Expr::STATE_NONE    );
      if( doX ){ infoNP1_[DIFFUSIVE_FLUX_X]=xFluxTagNew; factory.register_expression( scinew XFlux( xFluxTagNew, primVarTagNew, diffCoeffTag_, turbDiffTag_ ) ); }
      if( doY ){ infoNP1_[DIFFUSIVE_FLUX_Y]=yFluxTagNew; factory.register_expression( scinew YFlux( yFluxTagNew, primVarTagNew, diffCoeffTag_, turbDiffTag_ ) ); }
      if( doZ ){ infoNP1_[DIFFUSIVE_FLUX_Z]=zFluxTagNew; factory.register_expression( scinew ZFlux( zFluxTagNew, primVarTagNew, diffCoeffTag_, turbDiffTag_ ) ); }
    }
  }

  //---------------------------------------------------------------------------

  void
  EnthalpyTransportEquation::
  setup_source_terms( FieldTagInfo& info, Expr::TagList& srcTags )
  {
    // see if we have radiation - if so, plug it in
    if( params_->findBlock("RadiativeSourceTerm") ){
      info[SOURCE_TERM] = parse_nametag( params_->findBlock("RadiativeSourceTerm")->findBlock("NameTag") );
    }

    // populate any additional source terms pushed on this equation
    ScalarTransportEquation<SVolField>::setup_source_terms( info, srcTags );
  }

  //---------------------------------------------------------------------------

} /* namespace WasatchCore */
