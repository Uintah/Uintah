/*
 * The MIT License
 *
 * Copyright (c) 2016-2017 The University of Utah
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

#include <CCA/Components/Wasatch/Transport/SpeciesTransportEquation.h>

#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/LewisNumberSpeciesFlux.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbDiffFlux.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

// PoKiTT expressions that we require here
#include <pokitt/CanteraObjects.h>
#include <pokitt/transport/DiffusionCoeffMix.h>
#include <pokitt/transport/SpeciesDiffusion.h>

namespace WasatchCore{

typedef SpatialOps::SVolField FieldT;  // this is the field type we will support for species transport

//------------------------------------------------------------------------------

void
SpeciesTransportEquation::setup_diffusive_flux( FieldTagInfo& info )
{
  Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
  const TagNames& tagNames = TagNames::self();

  typedef typename FaceTypes<FieldT>::XFace XFaceT;
  typedef typename FaceTypes<FieldT>::YFace YFaceT;
  typedef typename FaceTypes<FieldT>::ZFace ZFaceT;

  bool isFirstDirection = true;

  for( Uintah::ProblemSpecP diffFluxParams=params_->findBlock("DiffusiveFlux");
       diffFluxParams != nullptr;
       diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") )
  {
    const std::string& primVarName = primVarTag_.name();

    std::string direction;
    diffFluxParams->getAttribute("direction",direction);

    for( std::string::iterator it = direction.begin(); it != direction.end(); ++it ){
      const std::string dir(1,*it);

      Expr::TagList diffFluxTags;
      for( int i=0; i<nspec_; ++i ){
        diffFluxTags.push_back( Expr::Tag( CanteraObjects::species_name(i) + tagNames.diffusiveflux + dir, Expr::STATE_NONE ) );
      }

      FieldSelector fs;
      if     ( dir=="X" ) fs=DIFFUSIVE_FLUX_X;
      else if( dir=="Y" ) fs=DIFFUSIVE_FLUX_Y;
      else if( dir=="Z" ) fs=DIFFUSIVE_FLUX_Z;
      else{
        std::ostringstream msg;
        msg << "Invalid direction selection for diffusive flux expression" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      info[fs] = diffFluxTags[specNum_];

      // Turbulent diffusive flux
      if( turbParams_.turbModelName != TurbulenceParameters::NOTURBULENCE ){
        const Expr::Tag turbFluxTag( "turb_diff_flux_"+primVarName, Expr::STATE_NONE );
        if( direction == "X" ){
          typedef TurbDiffFlux<XFaceT>::Builder TurbFlux;
          factory.register_expression( scinew TurbFlux( turbFluxTag, tagNames.turbulentviscosity, turbParams_.turbSchmidt, primVarTag_ ) );
        }
        if( direction == "Y" ){
          typedef TurbDiffFlux<YFaceT>::Builder TurbFlux;
          factory.register_expression( scinew TurbFlux( turbFluxTag, tagNames.turbulentviscosity, turbParams_.turbSchmidt, primVarTag_ ) );
        }
        if( direction == "Z" ){
          typedef TurbDiffFlux<ZFaceT>::Builder TurbFlux;
          factory.register_expression( scinew TurbFlux( turbFluxTag, tagNames.turbulentviscosity, turbParams_.turbSchmidt, primVarTag_ ) );
        }

        factory.attach_modifier_expression( turbFluxTag, diffFluxTags[specNum_] );
      }

      if( specNum_ != 0 ) continue; // the rest only gets done once since we build all fluxes into one expression

      //---------------------------
      // Constant Lewis number:
      // jcs need a way to get the cp and \lambda values in for Le specification...
      Uintah::ProblemSpecP lewisParams  = diffFluxParams->findBlock("LewisNumber");
      Uintah::ProblemSpecP mixAvgParams = diffFluxParams->findBlock("MixtureAveraged");
      if( lewisParams ){
        std::vector<double> lewisNumbers(nspec_,1.0);  // all species default to 1.0 unless otherwise specified
        for( Uintah::ProblemSpecP specParam=lewisParams->findBlock("Species");
            specParam != nullptr;
            specParam=lewisParams->findNextBlock("Species") )
        {
          std::string spnam;
          specParam->getAttribute( "name", spnam );
          specParam->getAttribute( "value", lewisNumbers[CanteraObjects::species_index(spnam)] );
        }

        const Expr::Tag thermCondTag = parse_nametag( lewisParams->findBlock("ThermalConductivity") );
        const Expr::Tag cpTag        = parse_nametag( lewisParams->findBlock("HeatCapacity"       ) );

        if( dir == "X" ){
          typedef LewisNumberDiffFluxes<XFaceT>::Builder DiffFlux;
          factory.register_expression( scinew DiffFlux( diffFluxTags, lewisNumbers, yiTags_, thermCondTag, cpTag ) );
        }
        else if( dir == "Y" ){
          typedef LewisNumberDiffFluxes<YFaceT>::Builder DiffFlux;
          factory.register_expression( scinew DiffFlux( diffFluxTags, lewisNumbers, yiTags_, thermCondTag, cpTag ) );
        }
        else if( dir == "Z" ){
          typedef LewisNumberDiffFluxes<ZFaceT>::Builder DiffFlux;
          factory.register_expression( scinew DiffFlux( diffFluxTags, lewisNumbers, yiTags_, thermCondTag, cpTag ) );
        }
        else{
          throw Uintah::ProblemSetupException( "Invalid direction encountered for diffusive flux spec.\n", __FILE__, __LINE__ );
        }

      } // constant lewis number

      //---------------------------
      // Mixture averaged
      else if( mixAvgParams ){

        Expr::TagList diffCoeffTags;
        for( int i=0; i<nspec_; ++i ){
          diffCoeffTags.push_back( Expr::Tag( CanteraObjects::species_name(i) + "_diff_coeff", Expr::STATE_NONE) );
        }

        // isotropic diffusion coefficients - only register once.
        if( isFirstDirection ){
          typedef pokitt::DiffusionCoeff<FieldT>::Builder DiffCoeff;
          factory.register_expression( scinew DiffCoeff( diffCoeffTags, temperatureTag_, tagNames.pressure, yiTags_, tagNames.mixMW ) );
          isFirstDirection = false;
        }

        if( dir == "X" ){
          typedef pokitt::SpeciesDiffFlux<XFaceT>::Builder DiffFlux;
          factory.register_expression( scinew DiffFlux( diffFluxTags, yiTags_, densityTag_, tagNames.mixMW, diffCoeffTags ) );
        }
        else if( dir == "Y" ){
          typedef pokitt::SpeciesDiffFlux<YFaceT>::Builder DiffFlux;
          factory.register_expression( scinew DiffFlux( diffFluxTags, yiTags_, densityTag_, tagNames.mixMW, diffCoeffTags ) );
        }
        else if( dir == "Z" ){
          typedef pokitt::SpeciesDiffFlux<ZFaceT>::Builder DiffFlux;
          factory.register_expression( scinew DiffFlux( diffFluxTags, yiTags_, densityTag_, tagNames.mixMW, diffCoeffTags ) );
        }
        else{
          throw Uintah::ProblemSetupException( "Invalid direction encountered for diffusive flux spec.\n", __FILE__, __LINE__ );
        }

      } // mixture averaged

      else{
        throw Uintah::ProblemSetupException( "Unsupported specification for species diffusive flux expression\n", __FILE__, __LINE__ );
      }

    } // loop over direction
  }
}

//------------------------------------------------------------------------------

} // namespace WasatchCore
