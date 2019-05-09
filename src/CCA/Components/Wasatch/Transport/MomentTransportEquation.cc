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

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Transport/MomentTransportEquation.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>

#include <CCA/Components/Wasatch/Expressions/PBE/Aggregation.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Birth.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Growth.h>
#include <CCA/Components/Wasatch/Expressions/PBE/MultiEnvSource.h>
#include <CCA/Components/Wasatch/Expressions/PBE/MultiEnvAveMoment.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/OstwaldRipening.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/AggregationEfficiency.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/Dissolution.h>

#include <CCA/Components/Wasatch/Expressions/PBE/QMOM.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/PBE/QuadratureClosure.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

#include <CCA/Components/Wasatch/Transport/ScalarTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>

#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace WasatchCore {

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
    Expr::ExpressionBuilder* builder = nullptr;

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
      std::stringstream twoPreviousMomentOrderStr;
      twoPreviousMomentOrderStr << momentOrder - 2;
      const Expr::Tag phiTag( basePhiName + "_" + twoPreviousMomentOrderStr.str(), Expr::STATE_N );
      if ( momentOrder == 1 || momentOrder == 0) {
        // register a moment closure expression
        const Expr::Tag momentClosureTag( basePhiName + "_" + twoPreviousMomentOrderStr.str(), Expr::STATE_N );
        typedef typename QuadratureClosure<FieldT>::Builder MomClosure;
        
        if (!factory.have_entry( momentClosureTag ) )
          factory.register_expression(scinew MomClosure(momentClosureTag,weightsTagList,abscissaeTagList,momentOrder-2));
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
        
        if (!factory.have_entry( momentClosureTag ) )
          factory.register_expression(scinew MomClosure(momentClosureTag,weightsTagList,abscissaeTagList,momentOrder+1));
      }
      typedef typename Growth<FieldT>::Builder growth;
      builder = scinew growth(growthTag, phiTag, growthCoefTag, momentOrder, constCoef);

    } else if (growthModel == "CONSTANT" || growthModel == "KINETIC" ) {   // g0
      std::stringstream previousMomentOrderStr;
      previousMomentOrderStr << momentOrder - 1;
      const Expr::Tag phiTag( basePhiName + "_" + previousMomentOrderStr.str(), Expr::STATE_N );
      if (momentOrder == 0) { //register closure expr
        const Expr::Tag momentClosureTag( basePhiName + "_" + previousMomentOrderStr.str(), Expr::STATE_N );
        typedef typename QuadratureClosure<FieldT>::Builder MomClosure;
        
        if (!factory.have_entry( momentClosureTag ) )
          factory.register_expression(scinew MomClosure(momentClosureTag,weightsTagList,abscissaeTagList,momentOrder-1));
      }
      typedef typename Growth<FieldT>::Builder growth;
      builder = scinew growth(growthTag, phiTag, growthCoefTag, momentOrder, constCoef);
    }

    growthTags.push_back(growthTag);
    factory.register_expression( builder );
  }
  
  //-----------------------------------------------------------------
  template<typename FieldT>
  void setup_ostwald_expression( Uintah::ProblemSpecP ostwaldParams,
                                 const std::string& PopulationName,
                                 const Expr::TagList& weightsTagList,
                                 const Expr::TagList& abscissaeTagList,
                                 Expr::ExpressionFactory& factory)
  {
    Expr::Tag ostwaldTag, m0Tag; // this tag will be populated
    Expr::ExpressionBuilder* builder = nullptr;
    ostwaldTag = Expr::Tag( "SBar_" + PopulationName , Expr::STATE_NONE );
    //need to m0 to normalize wieghts to add to 1, otherwise there are counted twice when the actually growht term is computed
    m0Tag = Expr::Tag( "m_" + PopulationName + "_0", Expr::STATE_DYNAMIC);
    
    double Molec_Vol;
    double Surf_Eng;
    double Temperature;
    double expCoef;
    const double R = 8.314;
    ostwaldParams->get("MolecularVolume", Molec_Vol);
    ostwaldParams->get("SurfaceEnergy", Surf_Eng);
    ostwaldParams->get("Temperature", Temperature);

    double CFCoef, RCutoff, tolmanLength;
    ostwaldParams->getWithDefault("ConversionFactor",CFCoef,1.0); //converts small radii to SI
    ostwaldParams->getWithDefault("RCutoff",RCutoff,0.1);
    ostwaldParams->getWithDefault("TolmanLength",tolmanLength,0.0);
    expCoef = 2.0*Molec_Vol*Surf_Eng/R/Temperature / CFCoef;  //r is divided in this equation later
    
    typedef typename OstwaldRipening<FieldT>::Builder ostwald;
    builder = scinew ostwald(ostwaldTag, weightsTagList, abscissaeTagList, m0Tag, expCoef, tolmanLength, RCutoff);
    factory.register_expression( builder );
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
    Expr::ExpressionBuilder* builder = nullptr;

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

    typename Birth<FieldT>::BirthModel birthType = Birth<FieldT>::POINT;
    if (birthModel == "POINT") {
      birthType = Birth<FieldT>::POINT;
    } else if (birthModel == "UNIFORM") {
      birthType = Birth<FieldT>::UNIFORM;
    } else if (birthModel == "NORMAL") {
      birthType = Birth<FieldT>::NORMAL;
    }
    
    birthTag = Expr::Tag( thisPhiName + "_birth_" + birthModel, Expr::STATE_NONE );
    typedef typename Birth<FieldT>::Builder birth;
    builder = scinew birth(birthTag, birthCoefTag, RStarTag, preCoef, momentOrder, birthType, ConstRStar, stdDev);

    birthTags.push_back(birthTag);
    factory.register_expression( builder );
  }
  
  //------------------------------------------------------------------
  
  template<typename FieldT>
  void setup_death_expression( Uintah::ProblemSpecP deathParams,
                               const std::string& PopulationName,
                               const std::string& thisPhiName,
                               const double momentOrder,
                               const Expr::TagList& weightsTagList,
                               const Expr::TagList& abscissaeTagList,
                               Expr::TagList& deathTags,
                               Expr::ExpressionFactory& factory)
  {
    Expr::Tag ostwaldTag, deathTag, superSatTag;
    Expr::ExpressionBuilder* builder = nullptr;
    ostwaldTag = Expr::Tag( "SBar_" + PopulationName , Expr::STATE_NONE );
    deathTag = Expr::Tag(thisPhiName + "_death", Expr::STATE_NONE ); 
    
    double rCutoff, deathCoefficient;
    deathParams->get("CriticalRadius", rCutoff);
    deathParams->getWithDefault("DeathCoefficient", deathCoefficient, 1.0);
    Uintah::ProblemSpecP nameTagParam = deathParams->findBlock("Supersaturation")->findBlock("NameTag");
    superSatTag = parse_nametag( nameTagParam );
    
    typedef typename Dissolution<FieldT>::Builder death;
    builder = scinew death(deathTag, weightsTagList, abscissaeTagList, ostwaldTag, superSatTag, rCutoff, momentOrder, deathCoefficient);
    deathTags.push_back(deathTag);
    factory.register_expression( builder );
  }
  
  //------------------------------------------------------------------
  
  template<typename FieldT>
  void setup_aggregation_expression ( Uintah::ProblemSpecP aggParams,
                                      const std::string& thisPhiName,
                                      const std::string& PopulationName,
                                      const double momentOrder,
                                      const int nEnv,
                                      Expr::TagList& aggTags,
                                      const Expr::TagList& weightsTagList,
                                      const Expr::TagList& abscissaeTagList,
                                      Expr::ExpressionFactory& factory) 
  {
    std::string aggModel;
    double efficiencyCoef;
    Expr::Tag aggTag;
    Expr::ExpressionBuilder* builder = nullptr;
    
    Expr::Tag aggCoefTag;
    if( aggParams->findBlock("AggregationCoefficientExpression") ){
      Uintah::ProblemSpecP nameTagParam = aggParams->findBlock("AggregationCoefficientExpression")->findBlock("NameTag");
      aggCoefTag = parse_nametag( nameTagParam );
    }

    aggParams->getWithDefault("EfficiencyCoefficient", efficiencyCoef, 1.0);
    aggParams->get("AggregationModel", aggModel );
    
    bool useEffTags = false;
    Expr::TagList efficiencyTagList;
    if (aggParams->findBlock("SizeDependentEfficiency") ){
      useEffTags = true;
      Expr::Tag efficiencyTag;
      for (int i = 0; i<nEnv; i++) {
        for (int j = 0; j<nEnv; j++) {
          std::stringstream iStr,jStr ;
          iStr << i; jStr << j;
          efficiencyTag = Expr::Tag(PopulationName + "_" + aggModel + "_efficiency_" + iStr.str() + jStr.str(), Expr::STATE_NONE );     
          efficiencyTagList.push_back(efficiencyTag);
        }
      }
      if (momentOrder == 0) { //only register coefficient expression once per pop name
        Uintah::ProblemSpecP effParams = aggParams->findBlock("SizeDependentEfficiency");
        double lengthParam;
        Uintah::ProblemSpecP nameTagParam;
        Expr::Tag growthCoefTag;
        Expr::Tag densityTag;
        Expr::Tag dissipationTag;
        std::string growthModel;
        growthCoefTag = parse_nametag( nameTagParam = effParams->findBlock("GrowthCoefficientExpression")->findBlock("NameTag") );
        densityTag = parse_nametag( nameTagParam = effParams->findBlock("Density")->findBlock("NameTag") );
        dissipationTag = parse_nametag( nameTagParam = effParams->findBlock("EnergyDissipation")->findBlock("NameTag") );
        effParams->getWithDefault("LengthParam", lengthParam, 1.0);
        effParams->get("GrowthModel",growthModel);
        
        typedef typename AggregationEfficiency<FieldT>::Builder aggregationEfficiency;
        builder = scinew aggregationEfficiency(efficiencyTagList, abscissaeTagList, growthCoefTag, dissipationTag, densityTag, lengthParam, growthModel);
        factory.register_expression(builder);
      }
    }

    typename Aggregation<FieldT>::AggregationModel aggType = Aggregation<FieldT>::CONSTANT;
    if (aggModel == "CONSTANT") {
      aggType = Aggregation<FieldT>::CONSTANT;
    } else if (aggModel == "BROWNIAN") {
      aggType = Aggregation<FieldT>::BROWNIAN;
    } else if (aggModel == "HYDRODYNAMIC") {
      aggType = Aggregation<FieldT>::HYDRODYNAMIC;
    }
    
    aggTag = Expr::Tag( thisPhiName + "_agg_" + aggModel, Expr::STATE_NONE );
    typedef typename Aggregation<FieldT>::Builder aggregation;
    builder = scinew aggregation(aggTag, weightsTagList, abscissaeTagList, efficiencyTagList, aggCoefTag,  momentOrder, efficiencyCoef, aggType, useEffTags);
    aggTags.push_back(aggTag);
    factory.register_expression( builder );
  }
  //------------------------------------------------------------------
  
  std::string
  get_population_name( Uintah::ProblemSpecP params )
  {
    std::string name;
    params->get("PopulationName",name);
    return name;
  }

  std::string
  get_soln_var_name( Uintah::ProblemSpecP params,
                     const double momentOrder )
  {
    std::stringstream momentOrderStr;
    momentOrderStr << momentOrder;
    return std::string("m_" + get_population_name(params)+ "_" + momentOrderStr.str());
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  MomentTransportEquation<FieldT>::
  MomentTransportEquation( const std::string thisPhiName,
                           GraphCategories& gc,
                           const double momentOrder,
                           Uintah::ProblemSpecP params,
                           const double initialMoment )
  : WasatchCore::TransportEquation( gc,
                                get_soln_var_name(params,momentOrder),
                                get_staggered_location<FieldT>() ),
   params_( params ),
   populationName_ ( get_population_name(params) ),
   baseSolnVarName_( "m_" + populationName_ ),
   momentOrder_    ( momentOrder ),
   initialMoment_  ( initialMoment )
  {
    // jcs need to ensure tha this is executed before the base class calls setup()...
    params->get( "NumberOfEnvironments", nEnv_ );
    nEqn_ = 2*nEnv_;

    const bool realizableQMOM = params->findBlock("RealizableQMOM");

    Expr::TagList weightsAndAbscissaeTags;
    //
    // fill in the weights and abscissae tags
    //
    for( unsigned i=0; i<nEnv_; ++i ){
      std::stringstream envID;
      envID << i;
      weightsAndAbscissaeTags.push_back(Expr::Tag("w_" + populationName_ + "_" + envID.str(), Expr::STATE_NONE) );
      weightsAndAbscissaeTags.push_back(Expr::Tag("a_" + populationName_ + "_" + envID.str(), Expr::STATE_NONE) );
      weightsTags_  .push_back(Expr::Tag("w_" + populationName_ + "_" + envID.str(), Expr::STATE_NONE) );
      abscissaeTags_.push_back(Expr::Tag("a_" + populationName_ + "_" + envID.str(), Expr::STATE_NONE) );
    }

    if( momentOrder_ == 0 ){ // only register the qmom expression once
      //
      // construct the transported moments taglist. This will be used
      // to register the qmom expression
      //
      Expr::TagList transportedMomentTags;
      for( unsigned iEq=0; iEq<nEqn_; ++iEq ){
        std::stringstream strMomID;
        const double momentOrder = (double) iEq;
        strMomID << momentOrder;
        const std::string phiName = "m_" + populationName_ + "_" + strMomID.str();
        transportedMomentTags.push_back( Expr::Tag(phiName,Expr::STATE_DYNAMIC) );
      }
      typedef typename QMOM<FieldT>::Builder QMOMExpr;
      Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
      factory.register_expression( scinew QMOMExpr( weightsAndAbscissaeTags, transportedMomentTags, realizableQMOM) );
    }

    setup();

  }

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
    return icFactory.get_id( initial_condition_tag() );
  }

  //------------------------------------------------------------------
  
  template< typename FieldT >
  void MomentTransportEquation<FieldT>::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     WasatchBCHelper& bcHelper )
  {
    const Category taskCat = INITIALIZATION;
    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
    
    
    // multiply the initial condition by the volume fraction for embedded geometries
    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    if (vNames.has_embedded_geometry()) {
      
      const Expr::Tag& phiTag = solution_variable_tag();
      
      std::cout << "attaching modifier expression on " << phiTag << std::endl;
      //create modifier expression
      typedef ExprAlgebra<FieldT> ExprAlgbr;
      Expr::TagList theTagList;
      theTagList.push_back( vNames.vol_frac_tag<FieldT>() );
      Expr::Tag modifierTag = Expr::Tag( this->solution_variable_name() + "_init_cond_modifier", Expr::STATE_NONE);
      factory.register_expression( new typename ExprAlgbr::Builder(modifierTag,
                                                                   theTagList,
                                                                   ExprAlgbr::PRODUCT,
                                                                   true) );
      
      factory.attach_modifier_expression( modifierTag, phiTag );
    }
    
    if (factory.have_entry(solution_variable_tag()))
      bcHelper.apply_boundary_condition<FieldT>(solution_variable_tag(), taskCat);


  }

  //------------------------------------------------------------------
  
  template< typename FieldT >
  void MomentTransportEquation<FieldT>::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             WasatchBCHelper& bcHelper )
  {
    const Category taskCat = ADVANCE_SOLUTION;
    bcHelper.apply_boundary_condition<FieldT>(this->solution_variable_tag(), taskCat);
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  MomentTransportEquation<FieldT>::setup_diffusive_flux( FieldTagInfo& info )
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    for( Uintah::ProblemSpecP diffFluxParams=params_->findBlock("DiffusiveFlux");
        diffFluxParams != nullptr;
        diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") ) {
      const Expr::Tag turbDiffTag = Expr::Tag();
      setup_diffusive_velocity_expression<FieldT>( diffFluxParams, solnVarTag_, turbDiffTag, factory, info );
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  MomentTransportEquation<FieldT>::setup_convective_flux( FieldTagInfo& info )
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    for( Uintah::ProblemSpecP convFluxParams=params_->findBlock("ConvectiveFlux");
         convFluxParams != nullptr;
         convFluxParams=convFluxParams->findNextBlock("ConvectiveFlux") ) {
       setup_convective_flux_expression<FieldT>( convFluxParams, solnVarTag_, factory, info );
     }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  MomentTransportEquation<FieldT>::setup_source_terms( FieldTagInfo& info, Expr::TagList& rhsTags )
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;

    //____________
    //Multi Environment Mixing
    if( params_->findBlock("MultiEnvMixingModel") ){
      Expr::TagList multiEnvWeightsTags;
      const int numMixingEnv = 3;
      //create the tag list for the multi env weights
      const Expr::Tag baseTag = parse_nametag( params_->findBlock("MultiEnvMixingModel")->findBlock("NameTag") );
      for( int i=0; i<numMixingEnv; ++i ){
        std::stringstream wID;
        wID << i;
        multiEnvWeightsTags.push_back( Expr::Tag("w_"    + baseTag.name() + "_" + wID.str(), baseTag.context() ) );
        multiEnvWeightsTags.push_back( Expr::Tag("dwdt_" + baseTag.name() + "_" + wID.str(), baseTag.context() ) );
      }

      //register source term from mixing
      const Expr::Tag mixingSourceTag( solnVarName_ + "_mixing_source", Expr::STATE_NONE);
      typedef typename MultiEnvSource<FieldT>::Builder MixSource;
      factory.register_expression( scinew MixSource( mixingSourceTag, multiEnvWeightsTags, solnVarTag_, initialMoment_ ) );
      rhsTags.push_back( mixingSourceTag );

      //register averaged moment
      const Expr::Tag aveMomentTag( solnVarName_ + "_ave", Expr::STATE_NONE );
      typedef typename MultiEnvAveMoment<FieldT>::Builder AveMoment;
      factory.register_expression( scinew AveMoment( aveMomentTag, multiEnvWeightsTags, solnVarTag_, initialMoment_ ) );
    }

    //____________
    // Growth
    for( Uintah::ProblemSpecP growthParams=params_->findBlock("GrowthExpression");
        growthParams != nullptr;
        growthParams=growthParams->findNextBlock("GrowthExpression") ) {
      setup_growth_expression <FieldT>( growthParams, baseSolnVarName_, solnVarName_,
                                        momentOrder_, nEqn_,            rhsTags,
                                        weightsTags_, abscissaeTags_,   factory );
    }

    //_________________
    // Ostwald Ripening
    if( momentOrder_ == 0 ){ // only register sBar once
      if ( params_->findBlock("OstwaldRipening") ) {
        Uintah::ProblemSpecP ostwaldParams = params_->findBlock("OstwaldRipening");
        setup_ostwald_expression <FieldT>( ostwaldParams, populationName_,
                                           weightsTags_,  abscissaeTags_,
                                           factory );
      }
    }

    //_________________
    // Birth
    for( Uintah::ProblemSpecP birthParams=params_->findBlock("BirthExpression");
        birthParams != nullptr;
        birthParams = birthParams->findNextBlock("BirthExpression") ) {
      setup_birth_expression <FieldT>( birthParams,  baseSolnVarName_, solnVarName_,
                                       momentOrder_, nEqn_,            rhsTags,
                                       factory );
    }

    //_________________
    // Death
    if ( params_->findBlock("Dissolution") ) {
      Uintah::ProblemSpecP deathParams = params_->findBlock("Dissolution");
      setup_death_expression <FieldT>( deathParams,  populationName_, solnVarName_,
                                       momentOrder_, weightsTags_,    abscissaeTags_,
                                       rhsTags,      factory );
    }

    //_________________
    // Aggregation
    for( Uintah::ProblemSpecP aggParams=params_->findBlock("AggregationExpression");
        aggParams != nullptr;
        aggParams = aggParams->findNextBlock("AggregationExpression") ) {
      setup_aggregation_expression <FieldT>( aggParams,    solnVarName_,   populationName_,
                                             momentOrder_, nEnv_,          rhsTags,
                                             weightsTags_, abscissaeTags_, factory );
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  Expr::ExpressionID
  MomentTransportEquation<FieldT>::setup_rhs( FieldTagInfo& info,
                                              const Expr::TagList& srcTags )
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;

    // Because of the forms that the ScalarRHS expression builders are defined,
    // we need a density tag and a boolean variable to be passed into this expression
    // builder. So we just define an empty tag and a false boolean to be passed into
    // the builder of ScalarRHS in order to prevent any errors in ScalarRHS
    const Expr::Tag densT = Expr::Tag();
    const bool tempConstDens = false;
    return factory.register_expression( scinew typename ScalarRHS<FieldT>::Builder(rhsTag_,info,srcTags,densT,tempConstDens) );
  }

  //------------------------------------------------------------------

  //==================================================================
  // Explicit template instantiation
  template class MomentTransportEquation< SVolField >;
  //==================================================================

} // namespace WasatchCore

