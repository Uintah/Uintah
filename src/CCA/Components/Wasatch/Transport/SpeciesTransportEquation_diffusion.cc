/*
 * The MIT License
 *
 * Copyright (c) 2016-2021 The University of Utah
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
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/Expressions/LewisNumberSpeciesFlux.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbDiffFlux.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/DualTimeMatrixManager.h>

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
SpeciesTransportEquation::
setup_diffusive_flux( FieldTagInfo& info )
{
  Expr::ExpressionFactory& factory   = *gc_[ADVANCE_SOLUTION]->exprFactory;
  Expr::ExpressionFactory& icFactory = *gc_[INITIALIZATION  ]->exprFactory;

  const TagNames& tagNames = TagNames::self();

  if( flowTreatment_ == LOWMACH ){
    if( turbParams_.turbModelName == TurbulenceParameters::NOTURBULENCE ){
      const std::string suffix = "";
      register_diffusive_flux_expressions( INITIALIZATION  , infoInit_, densityInitTag_, primVarInitTag_, yiInitTags_, Expr::STATE_NONE, suffix );
      register_diffusive_flux_expressions( ADVANCE_SOLUTION, infoNP1_ , densityNP1Tag_ , primVarNP1Tag_ , yiNP1Tags_ , Expr::STATE_NP1 , suffix );

      // set info for diffusive flux based on infoNP1_, add tag names to set of persistent fields
      const std::set<FieldSelector> fsSet = {DIFFUSIVE_FLUX_X, DIFFUSIVE_FLUX_Y, DIFFUSIVE_FLUX_Z};
      for( FieldSelector fs : fsSet ){
       if( infoNP1_.find(fs) != infoNP1_.end() ){
         const std::string diffFluxName = infoNP1_[fs].name();
         info[fs] = Expr::Tag(diffFluxName, Expr::STATE_N);
         persistentFields_.insert(diffFluxName);

         // Force diffusive flux expression on initialization graph.
         const Expr::ExpressionID id = icFactory.get_id( infoInit_[fs] );
         gc_[INITIALIZATION]->rootIDs.insert(id);

         // deal with nth species diffusive flux
         if( specNum_==0 ){
           FieldTagInfo infoSpecN;
           const std::string dir = (fs == DIFFUSIVE_FLUX_X) ? "X"
                                 : (fs == DIFFUSIVE_FLUX_Y) ? "Y"
                                 :                            "Z";
           const std::string specName = CanteraObjects::species_name( nspec_-1 );
           infoSpecN[fs] = Expr::Tag( specName + tagNames.diffusiveflux + dir + suffix, Expr::STATE_N );
           persistentFields_.insert(infoSpecN[fs].name());

           register_diffusive_flux_placeholders<FieldT>( factory, infoSpecN );
           const Expr::Tag initSpecNDiffFluxTag = Expr::Tag(infoSpecN[fs].name(), Expr::STATE_NONE );
           const Expr::ExpressionID id = icFactory.get_id( initSpecNDiffFluxTag );
           gc_[INITIALIZATION]->rootIDs.insert(id);
          }// if( specNum_==0 )
        }// if( infoNP1_... )
      }// for( ... )

      // Register placeholders for diffusive flux parameters at STATE_N
      register_diffusive_flux_placeholders<FieldT>( factory, info );
    } //if(turbParams_.turbModelName == TurbulenceParameters::NOTURBULENCE)

    /* if the low-Mach algorithm is used with a turbulence model, we need to compute
     * the diffusive flux at STATE_N and an estimate for diffusive flux at STATE_NP1.
     * We cannot calculate the STATE_NP1 value exactly attempting to do so would
     * induce a circular dependency like so:
     * diffusive-flux_NP1 --> (...) --> velocity_NP1 --> (...) --> diffusive-flux_NP1
     */
    else{
      const Expr::Context context = Expr::STATE_NONE;
      const std::string suffix = "-NP1-estimate";
      register_diffusive_flux_expressions( ADVANCE_SOLUTION, info    , densityTag_   , primVarTag_   , yiTags_   , context, ""     );
      register_diffusive_flux_expressions( ADVANCE_SOLUTION, infoNP1_, densityNP1Tag_, primVarNP1Tag_, yiNP1Tags_, context, suffix );
    }
  }
  else{
    register_diffusive_flux_expressions( ADVANCE_SOLUTION, info, densityTag_, primVarTag_, yiTags_, Expr::STATE_NONE, "" );
  }
}

//------------------------------------------------------------------------------

void
SpeciesTransportEquation::
register_diffusive_flux_expressions( const Category       cat,
                                     FieldTagInfo&        info,
                                     const Expr::Tag&     densityTag,
                                     const Expr::Tag&     primVarTag,
                                     const Expr::TagList& yiTags,
                                     const Expr::Context  context,
                                     const std::string    suffix )
{
  Expr::ExpressionFactory& factory = *gc_[cat]->exprFactory;
  const TagNames& tagNames = TagNames::self();

  typedef typename FaceTypes<FieldT>::XFace XFaceT;
  typedef typename FaceTypes<FieldT>::YFace YFaceT;
  typedef typename FaceTypes<FieldT>::ZFace ZFaceT;

  Expr::Tag pressureTag = Wasatch::flow_treatment() == LOWMACH
                          ? tagNames.thermodynamicPressure
                          : tagNames.pressure;

  if((cat == INITIALIZATION) && (flow_treatment() == LOWMACH)) pressureTag.reset_context(Expr::STATE_NONE);

  bool isFirstDirection = true;

  for( Uintah::ProblemSpecP diffFluxParams=params_->findBlock("DiffusiveFlux");
       diffFluxParams != nullptr;
       diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") )
  {
    const std::string& primVarName = primVarTag.name();

    std::string direction;
    diffFluxParams->getAttribute("direction",direction);

    for( std::string::iterator it = direction.begin(); it != direction.end(); ++it ){
      const std::string dir(1,*it);

      Expr::TagList diffFluxTags;
      for( int i=0; i<nspec_; ++i ){
        diffFluxTags.push_back( Expr::Tag( CanteraObjects::species_name(i) + tagNames.diffusiveflux + dir + suffix, context ) );
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
          factory.register_expression( scinew TurbFlux( turbFluxTag, tagNames.turbulentviscosity, turbParams_.turbSchmidt, primVarTag ) );
        }
        if( direction == "Y" ){
          typedef TurbDiffFlux<YFaceT>::Builder TurbFlux;
          factory.register_expression( scinew TurbFlux( turbFluxTag, tagNames.turbulentviscosity, turbParams_.turbSchmidt, primVarTag ) );
        }
        if( direction == "Z" ){
          typedef TurbDiffFlux<ZFaceT>::Builder TurbFlux;
          factory.register_expression( scinew TurbFlux( turbFluxTag, tagNames.turbulentviscosity, turbParams_.turbSchmidt, primVarTag ) );
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
          const Expr::ExpressionID id =
          factory.register_expression( scinew DiffFlux( diffFluxTags, lewisNumbers, yiTags, thermCondTag, cpTag ) );

//          if(cat == ADVANCE_SOLUTION) factory.cleave_from_parents( id );
        }
        else if( dir == "Y" ){
          typedef LewisNumberDiffFluxes<YFaceT>::Builder DiffFlux;
          const Expr::ExpressionID id =
          factory.register_expression( scinew DiffFlux( diffFluxTags, lewisNumbers, yiTags, thermCondTag, cpTag ) );

//          if(cat == ADVANCE_SOLUTION) factory.cleave_from_parents( id );
        }
        else if( dir == "Z" ){
          typedef LewisNumberDiffFluxes<ZFaceT>::Builder DiffFlux;
          const Expr::ExpressionID id =
          factory.register_expression( scinew DiffFlux( diffFluxTags, lewisNumbers, yiTags, thermCondTag, cpTag ) );

//          if(cat == ADVANCE_SOLUTION) factory.cleave_from_parents( id );
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
          diffCoeffTags.push_back( Expr::Tag( CanteraObjects::species_name(i) + "_diff_coeff" + suffix, Expr::STATE_NONE) );
        }

        // isotropic diffusion coefficients - only register once.
        if( isFirstDirection ){
          typedef pokitt::DiffusionCoeff<FieldT>::Builder DiffCoeff;
          factory.register_expression( scinew DiffCoeff( diffCoeffTags, temperatureTag_, pressureTag, yiTags, tagNames.mixMW ) );
          isFirstDirection = false;

          dualTimeMatrixInfo_.set_diffusivities( diffCoeffTags );
        }

        if     ( dir == "X" ){
          typedef pokitt::SpeciesDiffFlux<XFaceT>::Builder DiffFlux;
          factory.register_expression( scinew DiffFlux( diffFluxTags, yiTags, densityTag, tagNames.mixMW, diffCoeffTags ) );
        }
        else if( dir == "Y" ){
          typedef pokitt::SpeciesDiffFlux<YFaceT>::Builder DiffFlux;
          factory.register_expression( scinew DiffFlux( diffFluxTags, yiTags, densityTag, tagNames.mixMW, diffCoeffTags ) );
        }
        else if( dir == "Z" ){
          typedef pokitt::SpeciesDiffFlux<ZFaceT>::Builder DiffFlux;
          factory.register_expression( scinew DiffFlux( diffFluxTags, yiTags, densityTag, tagNames.mixMW, diffCoeffTags ) );
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
