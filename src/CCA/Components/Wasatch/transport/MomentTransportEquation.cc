/*
 * Copyright (c) 2012 The University of Utah
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

//-- Wasatch includes --//
#include "MomentTransportEquation.h"
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>

#include <CCA/Components/Wasatch/Expressions/PBE/Birth.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Growth.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/OstwaldRipening.h>

#include <CCA/Components/Wasatch/Expressions/PBE/QMOM.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/PBE/QuadratureClosure.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/transport/ScalarTransportEquation.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Wasatch {

  //------------------------------------------------------------------
  template< typename FieldT >
  void setup_growth_expression( Uintah::ProblemSpecP growthParams,
                               const std::string& basePhiName,
                               const std::string& thisPhiName,
                               const double momentOrder,
                               const int nEqs,
                               Expr::TagList& growthTags,
                               const Expr::TagList& weightsTagList,
                               const Expr::TagList& abscissaeTagList,
                               Expr::ExpressionFactory& factory)
  {
    Expr::Tag growthTag; // this tag will be populated
    Expr::ExpressionBuilder* builder = NULL;

    Expr::Tag growthCoefTag;
    if( growthParams->findBlock("GrowthCoefficientExpression") ){
      Uintah::ProblemSpecP nameTagParam = growthParams->findBlock("GrowthCoefficientExpression")->findBlock("NameTag");
      growthCoefTag = parse_nametag( nameTagParam );
    }

    std::string growthModel;
    growthParams->get("GrowthModel",growthModel);
    growthTag = Expr::Tag( thisPhiName + "_growth_" + growthModel, Expr::STATE_NONE);
    double constCoef = 1.0;
    if (growthParams->findBlock("PreGrowthCoefficient") )
      growthParams->get("PreGrowthCoefficient",constCoef);

    if (growthModel == "BULK_DIFFUSION") {      //g(r) = 1/r
      std::stringstream previousMomentOrderStr;
      previousMomentOrderStr << momentOrder - 2;
      const Expr::Tag phiTag( basePhiName + "_" + previousMomentOrderStr.str(), Expr::STATE_N );
      if ( momentOrder == 1 || momentOrder == 0) {
        // register a moment closure expression
        const Expr::Tag momentClosureTag( basePhiName + "_" + previousMomentOrderStr.str(), Expr::STATE_N );
        typedef typename QuadratureClosure<FieldT>::Builder MomClosure;
        factory.register_expression(scinew MomClosure(momentClosureTag,weightsTagList,abscissaeTagList,momentOrder));
      }
      typedef typename Growth<FieldT>::Builder growth;
      builder = scinew growth(growthTag, phiTag, growthCoefTag, momentOrder, constCoef);

    } else if (growthModel == "MONOSURFACE") { //g(r) = r^2
      std::stringstream nextMomentOrderStr;
      nextMomentOrderStr << momentOrder + 1;
      const Expr::Tag phiTag( basePhiName + "_" + nextMomentOrderStr.str(), Expr::STATE_N );
      if (nEqs == momentOrder + 1) {
        // register a moment closure expression
        const Expr::Tag momentClosureTag( basePhiName + "_" + nextMomentOrderStr.str(), Expr::STATE_N );
        typedef typename QuadratureClosure<FieldT>::Builder MomClosure;
        factory.register_expression(scinew MomClosure(momentClosureTag,weightsTagList,abscissaeTagList,momentOrder+1));
      }
      typedef typename Growth<FieldT>::Builder growth;
      builder = scinew growth(growthTag, phiTag, growthCoefTag, momentOrder, constCoef);

    } else if (growthModel == "CONSTANT") {   // g0
      std::stringstream currentMomentOrderStr;
      currentMomentOrderStr << momentOrder;
      const Expr::Tag phiTag( basePhiName + "_" + currentMomentOrderStr.str(), Expr::STATE_N );
      typedef typename Growth<FieldT>::Builder growth;
      builder = scinew growth(growthTag, phiTag, growthCoefTag, momentOrder, constCoef);
    }

    growthTags.push_back(growthTag);
    factory.register_expression( builder );

    if (growthParams->findBlock("OstwaldRipening") ){
      Uintah::ProblemSpecP ostwaldParams = growthParams->findBlock("OstwaldRipening");
      int nPts = nEqs/2;
      Expr::Tag ostwaldTag; // this tag will be populated
      Expr::ExpressionBuilder* builder2 = NULL;
      ostwaldTag = Expr::Tag( thisPhiName + "_Ostwald_Ripening", Expr::STATE_NONE );

      double Molec_Vol;
      double Surf_Eng;
      double Temperature;
      double expCoef;
      const double R = 8.314;
      ostwaldParams->get("MolecularVolume", Molec_Vol);
      ostwaldParams->get("SurfaceEnergy", Surf_Eng);
      ostwaldParams->get("Temperature", Temperature);
      double CFCoef = 1.0;
      if (ostwaldParams->findBlock("ConversionFactor") )
        ostwaldParams->get("ConversionFactor", CFCoef);  //converts small radii to SI

      double RCutoff = 0.1;                           //Does RCutoff need an expression in future?
      if (ostwaldParams->findBlock("RCutoff") )
        ostwaldParams->get("RCutoff",RCutoff);
      expCoef = 2.0*Molec_Vol*Surf_Eng/R/Temperature * CFCoef;

      typedef typename OstwaldRipening<FieldT>::Builder ostwald;
      builder2 = scinew ostwald(ostwaldTag, growthCoefTag, weightsTagList, abscissaeTagList, momentOrder, expCoef, RCutoff, constCoef, nPts);
      growthTags.push_back(ostwaldTag);
      factory.register_expression( builder2 );
    }
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void setup_birth_expression( Uintah::ProblemSpecP birthParams,
                              const std::string& basePhiName,
                              const std::string& thisPhiName,
                              const double momentOrder,
                              const int nEqs,
                              Expr::TagList& birthTags,
                              Expr::ExpressionFactory& factory)
  {
    //birth expr is of the form $\f J * B_0(S) * B(r^*) $\f
    //where r* can be an expression or constant
    Expr::Tag birthTag; // this tag will be populated
    Expr::ExpressionBuilder* builder = NULL;

    std::string birthModel;
    birthParams->get("BirthModel",birthModel);

    double preCoef = 1.0;
    if (birthParams->findBlock("PreBirthCoefficient") )
      birthParams->get("PreBirthCoefficient",preCoef);

    double stdDev = 1.0;
    if (birthParams->findBlock("StandardDeviation") )
      birthParams->get("StandardDeviation",stdDev);

    Expr::Tag birthCoefTag;
    //register coefficient expr if found
    if( birthParams->findBlock("BirthCoefficientExpression") ){
      Uintah::ProblemSpecP nameTagParam = birthParams->findBlock("BirthCoefficientExpression")->findBlock("NameTag");
      birthCoefTag = parse_nametag( nameTagParam );
    }

    Expr::Tag RStarTag;
   //register RStar Expr, or use const RStar if found
    double ConstRStar = 1.0;
    if( birthParams->findBlock("RStarExpression") ){
      Uintah::ProblemSpecP nameTagParam = birthParams->findBlock("RStarExpression")->findBlock("NameTag");
      RStarTag = parse_nametag( nameTagParam );
    } else {
      RStarTag = Expr::Tag ();
      if (birthParams->findBlock("ConstantRStar") )
        birthParams->get("ConstantRStar",ConstRStar);
    }

    birthTag = Expr::Tag( thisPhiName + "_birth_" + birthModel, Expr::STATE_NONE );
    typedef typename Birth<FieldT>::Builder birth;
    builder = scinew birth(birthTag, birthCoefTag, RStarTag, preCoef, momentOrder, birthModel, ConstRStar, stdDev);

    birthTags.push_back(birthTag);
    factory.register_expression( builder );
  }

  //------------------------------------------------------------------

  std::string
  get_population_name( Uintah::ProblemSpecP params )
  {
    std::string phiName;
    params->get("PopulationName",phiName);
    return phiName;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionID
  MomentTransportEquation<FieldT>::
  get_moment_rhs_id(Expr::ExpressionFactory& factory,
                    Uintah::ProblemSpecP params,
                    Expr::TagList& weightsTags,
                    Expr::TagList& abscissaeTags,
                    const double momentOrder)
  {
    std::stringstream momentOrderStr;
    momentOrderStr << momentOrder;
    const std::string PopulationName = get_population_name(params);
    int nEnv = 1;
    params->get( "NumberOfEnvironments", nEnv );
    const int nEqs = 2*nEnv;

    const std::string basePhiName = "m_" + PopulationName;
    const std::string thisPhiName = "m_" + PopulationName + "_" + momentOrderStr.str();
    const Expr::Tag thisPhiTag    = Expr::Tag( thisPhiName, Expr::STATE_N );

    //____________
    // start setting up the right-hand-side terms: these include expressions
    // for growth, nucleation, birth, death, and any other fancy source terms
    Expr::TagList rhsTags;
    typename ScalarRHS<FieldT>::FieldTagInfo info;

    //____________
    // Growth
    for( Uintah::ProblemSpecP growthParams=params->findBlock("GrowthExpression");
        growthParams != 0;
        growthParams=growthParams->findNextBlock("GrowthExpression") ){
      setup_growth_expression <FieldT>( growthParams,
                                        basePhiName,
                                        thisPhiName,
                                        momentOrder,
                                        nEqs,
                                        rhsTags,
                                        weightsTags,
                                        abscissaeTags,
                                        factory);
    }

    //_________________
    // Birth
    for( Uintah::ProblemSpecP birthParams=params->findBlock("BirthExpression");
        birthParams != 0;
        birthParams = birthParams->findNextBlock("BirthExpression") ){
      setup_birth_expression <FieldT>( birthParams,
                                       basePhiName,
                                       thisPhiName,
                                       momentOrder,
                                       nEqs,
                                       rhsTags,
                                       factory);
    }

    //_________________
    // Diffusive Fluxes
    for( Uintah::ProblemSpecP diffFluxParams=params->findBlock("DiffusiveFluxExpression");
        diffFluxParams != 0;
        diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFluxExpression") ){
      setup_diffusive_velocity_expression<FieldT>( diffFluxParams, thisPhiTag, factory, info );

    }

    //__________________
    // Convective Fluxes
    for( Uintah::ProblemSpecP convFluxParams=params->findBlock("ConvectiveFluxExpression");
        convFluxParams != 0;
        convFluxParams=convFluxParams->findNextBlock("ConvectiveFluxExpression") ){

      setup_convective_flux_expression<FieldT>( convFluxParams, thisPhiTag, factory, info );

    }

    //_____________
    // volume fraction for embedded boundaries Terms
    Expr::Tag volFracTag = Expr::Tag();
    if (params->findBlock("VolumeFractionExpression")) {
      volFracTag = parse_nametag( params->findBlock("VolumeFractionExpression")->findBlock("NameTag") );
    }

    Expr::Tag xAreaFracTag = Expr::Tag();
    if (params->findBlock("XAreaFractionExpression")) {
      xAreaFracTag = parse_nametag( params->findBlock("XAreaFractionExpression")->findBlock("NameTag") );
    }

    Expr::Tag yAreaFracTag = Expr::Tag();
    if (params->findBlock("YAreaFractionExpression")) {
      yAreaFracTag = parse_nametag( params->findBlock("YAreaFractionExpression")->findBlock("NameTag") );
    }

    Expr::Tag zAreaFracTag = Expr::Tag();
    if (params->findBlock("ZAreaFractionExpression")) {
      zAreaFracTag = parse_nametag( params->findBlock("ZAreaFractionExpression")->findBlock("NameTag") );
    }

    //
    // Because of the forms that the ScalarRHS expression builders are defined,
    // we need a density tag and a boolean variable to be passed into this expression
    // builder. So we just define an empty tag and a false boolean to be passed into
    // the builder of ScalarRHS in order to prevent any errors in ScalarRHS

    const Expr::Tag densT = Expr::Tag();
    const bool tempConstDens = false;
    const Expr::Tag rhsTag( thisPhiName + "_rhs", Expr::STATE_NONE );
    return factory.register_expression( scinew typename ScalarRHS<FieldT>::Builder(rhsTag,info,rhsTags,densT,
                                                                                   volFracTag,xAreaFracTag,yAreaFracTag,
                                                                                   zAreaFracTag,tempConstDens ));
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  MomentTransportEquation<FieldT>::
  MomentTransportEquation( const std::string thisPhiName,
                          const Expr::ExpressionID rhsID )
  : Wasatch::TransportEquation( thisPhiName, rhsID,
                                get_staggered_location<FieldT>() )
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  MomentTransportEquation<FieldT>::~MomentTransportEquation()
  {}

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionID
  MomentTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    return icFactory.get_id( Expr::Tag( this->solution_variable_name(),
                                                      Expr::STATE_N ) );
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void MomentTransportEquation<FieldT>::
  setup_initial_boundary_conditions(const GraphHelper& graphHelper,
                            const Uintah::PatchSet* const localPatches,
                            const PatchInfoMap& patchInfoMap,
                            const Uintah::MaterialSubset* const materials)
  {
    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
    const Expr::Tag phiTag( this->solution_variable_name(), Expr::STATE_N );
    if (factory.have_entry(phiTag)) {
      process_boundary_conditions<FieldT>( phiTag,
                                          this->solution_variable_name(),
                                          this->staggered_location(),
                                          graphHelper,
                                          localPatches,
                                          patchInfoMap,
                                          materials );
    }

  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void MomentTransportEquation<FieldT>::
  setup_boundary_conditions(const GraphHelper& graphHelper,
                            const Uintah::PatchSet* const localPatches,
                            const PatchInfoMap& patchInfoMap,
                            const Uintah::MaterialSubset* const materials)
  {

    // see BCHelperTools.cc
    process_boundary_conditions<FieldT>( Expr::Tag( this->solution_variable_name(),
                                                   Expr::STATE_N ),
                                        this->solution_variable_name(),
                                        this->staggered_location(),
                                        graphHelper,
                                        localPatches,
                                        patchInfoMap,
                                        materials );
  }

  //------------------------------------------------------------------

  //==================================================================
  // Explicit template instantiation
  template class MomentTransportEquation< SVolField >;
  template class MomentTransportEquation< XVolField >;
  template class MomentTransportEquation< YVolField >;
  template class MomentTransportEquation< ZVolField >;
  //==================================================================

} // namespace Wasatch

