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

#include <CCA/Components/Wasatch/Transport/SetupCoalModels.h>
#include <CCA/Components/Wasatch/Transport/ParticleSizeEquation.h>
#include <CCA/Components/Wasatch/Transport/PersistentParticleICs.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleDensity.h>
#include <CCA/Components/Wasatch/Expressions/RHSTerms.h>
#include <CCA/Components/Wasatch/Expressions/Particles/CellToParticleInterpExpr.h>
#include <CCA/Components/Wasatch/Expressions/Particles/ParticleToCellSrcExpr.h>
#include <CCA/Components/Wasatch/TagNames.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <pokitt/CanteraObjects.h>

/**
 *  \file SetupCoalModels.h
 *  \brief Performs tasks required to implement coal-related models
 */

namespace WasatchCore{

  SetupCoalModels::
  SetupCoalModels( Uintah::ProblemSpecP& particleSpec,
                   Uintah::ProblemSpecP& wasatchSpec,
                   Uintah::ProblemSpecP& coalSpec,
                   GraphCategories&      gc,
                   std::set<std::string> persistentFields )
    : gc_          ( gc           ),
      wasatchSpec_ ( wasatchSpec  ),
      particleSpec_( particleSpec ),
      coalSpec_    ( coalSpec     ),
      factory_     ( *(gc[ADVANCE_SOLUTION]->exprFactory) ),
      tagNames_    ( TagNames::self() ),
      pMassTag_    ( parse_nametag(particleSpec->findBlock("ParticleMass")) ),
      pSizeTag_    ( parse_nametag(particleSpec->findBlock("ParticleSize")) ),
      pDensTag_    ( parse_nametag(particleSpec->findBlock("ParticleDensity")) ),
      pTempTag_    ( parse_nametag(particleSpec->findBlock("ParticleTemperature")) ),
      gDensTag_    ( parse_nametag(particleSpec->findBlock("ParticleMomentum")->findBlock("GasProperties")->findBlock("GasDensity"  )) )
  {
    equationsSet_ = false;
    adaptors_.clear();
    Expr::ExpressionFactory& iFactory = *(gc[INITIALIZATION]->exprFactory);

    std::string pxName, pyName, pzName;
    Uintah::ProblemSpecP posSpec = particleSpec->findBlock("ParticlePosition");
    posSpec->getAttribute( "x", pxName );
    posSpec->getAttribute( "y", pyName );
    posSpec->getAttribute( "z", pzName );

    Expr::Context context = Expr::STATE_NONE;
    const Expr::Tag pxTag(pxName, Expr::STATE_DYNAMIC);
    const Expr::Tag pyTag(pyName, Expr::STATE_DYNAMIC);
    const Expr::Tag pzTag(pzName, Expr::STATE_DYNAMIC);
    pPosTags_ = tag_list(pxTag, pyTag, pzTag);

    const Expr::Tag gPressTag = tagNames_.pressure;
    const Expr::Tag gTempTag  = tagNames_.temperature;
    const Expr::Tag pReTag    = tagNames_.preynolds;
    const Expr::Tag mwTag     = tagNames_.mixMW;

    const Expr::Tag pSize0Tag(pSizeTag_.name()+"_init", Expr::STATE_DYNAMIC);
    const Expr::Tag pMass0Tag(pMassTag_.name()+"_init", Expr::STATE_DYNAMIC);
    const Expr::Tag pDens0Tag(pDensTag_.name()+"_init", context);
    persistentFields.insert( pSize0Tag.name() );
    persistentFields.insert( pMass0Tag.name() );
    persistentFields.insert( pDens0Tag.name() );

    // Setup equations to transport initial particle size and mass  ***********************//
    EquationBase*
    pSize0Eqn = scinew
    PersistentInitialParticleSize( pSize0Tag.name(),
                                   pPosTags_,
                                   pSize0Tag,
                                   gc_ );

    EquationBase*
    pMass0Eqn = scinew
    PersistentInitialParticleMass( pMass0Tag.name(),
                                   pPosTags_,
                                   pSizeTag_,
                                   particleSpec,
                                   gc_ );

    iFactory.register_expression( new ParticleDensity::Builder(pDens0Tag, pMass0Tag, pSize0Tag) );
    factory_.register_expression( new ParticleDensity::Builder(pDens0Tag, pMass0Tag, pSize0Tag) );

    adaptors_.push_back( scinew EqnTimestepAdaptor<ParticleField>(pSize0Eqn) );
    adaptors_.push_back( scinew EqnTimestepAdaptor<ParticleField>(pMass0Eqn) );

    //**************************************************************************************//

    Expr::Tag scTag("Schmidt_number", Expr::STATE_NONE);

    // Define non-species cell-to-particle tags *******************************************//
    const Expr::Tag gTempC2PTag  = get_c2p_tag( gTempTag. name() );
    const Expr::Tag gPressC2PTag = get_c2p_tag( gPressTag.name() );
    const Expr::Tag mwC2PTag     = get_c2p_tag( mwTag.    name() );

    //todo: Write an expression for the Schmidt Number
    factory_.register_expression( new Expr::ConstantExpr<ParticleField>::Builder( scTag, 1 ) );

    // interpolate gas properties to particle locations ***********************************//
    typedef CellToParticleInterpExpr<SVolField>::Builder C2PBuilder;
    factory_.register_expression( new C2PBuilder( gTempC2PTag , gTempTag , pSizeTag_, pPosTags_) );
    factory_.register_expression( new C2PBuilder( mwC2PTag    , mwTag    , pSizeTag_, pPosTags_) );
    factory_.register_expression( new C2PBuilder( gPressC2PTag, gPressTag, pSizeTag_, pPosTags_) );

    // Setup a map so the coal code knows what species tags to use *************************//
    Coal::SpeciesTagMap specTagMap;
    for( int i=0; i<CanteraObjects::number_species(); ++i ){
      const std::string& specName = CanteraObjects::species_name(i);
      const Coal::GasSpeciesName spec = Coal::gas_name_to_enum( specName );

      const Expr::Tag yiTag(specName          , Expr::STATE_NONE );
      yiTags_.push_back( yiTag );
      const Expr::Tag yiC2PTag = get_c2p_tag( specName );

      // interpolate species mass fraction from cell locations to particle locations
      factory_.register_expression( new C2PBuilder( yiC2PTag, yiTag, pSizeTag_, pPosTags_ ) );

      specTagMap[ spec ] = yiC2PTag;
    }
    // get info for coal ******************************************************************//
    std::string devModel, charModel, coalName;
    coalSpec->findBlock("DevolModel" )->getAttribute( "name", devModel  );
    coalSpec->findBlock("CharModel"  )->getAttribute( "name", charModel );
    coalSpec->findBlock("CoalType"   )->getAttribute( "name", coalName  );
    coalInterface_ = new Coal::CoalInterface<ParticleField>( gc_,
                                                             Coal::coal_type ( coalName  ),
                                                             DEV::devol_model( devModel  ),
                                                             CHAR::char_model( charModel ),
                                                             pSizeTag_,
                                                             pTempTag_,    gTempC2PTag,
                                                             mwC2PTag,    pDensTag_,
                                                             gPressC2PTag,pMassTag_,
                                                             pReTag,      scTag,
                                                             specTagMap,  pMass0Tag,
                                                             pDens0Tag,   pSize0Tag );

    const Coal::CoalEqVec eqns = coalInterface_->get_equations();
    for(size_t i=0; i<eqns.size(); i++){
      adaptors_.push_back( scinew EqnTimestepAdaptor<ParticleField>(eqns[i]) );
    }

    set_initial_conditions();
    setup_coal_src_terms();
  };

  SetupCoalModels::~SetupCoalModels(){ delete coalInterface_; }

  void
  SetupCoalModels::setup_cantera()
  {
    Uintah::ProblemSpecP specParams = wasatchSpec_->findBlock("SpeciesTransportEquations");
    std::string canteraInputFileName, canteraGroupName;
    specParams->get("CanteraInputFile",canteraInputFileName);
    specParams->get("CanteraGroup",    canteraGroupName    );

    CanteraObjects::Setup canteraSetup( "Mix", canteraInputFileName, canteraGroupName );
    CanteraObjects::setup_cantera( canteraSetup );
  }

  void
  SetupCoalModels::set_initial_conditions()
  {
    for( EquationAdaptors::const_iterator ia=adaptors_.begin(); ia!=adaptors_.end(); ++ia ){
      EqnTimestepAdaptorBase* const adaptor = *ia;
      EquationBase* eqn = adaptor->equation();
      try{
        proc0cout << "Setting initial conditions for coal equation: "
            << eqn->solution_variable_name()
            << std::endl;
        GraphHelper* const icGraphHelper = gc_[INITIALIZATION];
        icGraphHelper->rootIDs.insert( eqn->initial_condition( *icGraphHelper->exprFactory ) );
      }
      catch( std::runtime_error& e ){
        std::ostringstream msg;
        msg << e.what()
              << std::endl
              << "ERROR while setting initial conditions on coal equation "
              << eqn->solution_variable_name()
              << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }
    equationsSet_ = true;
  }

  void
  SetupCoalModels::setup_coal_src_terms()
  {
    typedef ParticleToCellSrcExpr<SVolField>::Builder P2CBuilder;

    // Connect coal source terms to gas species RHSs **************************************//
    for( size_t i=0; i<yiTags_.size(); ++i ){
      const std::string          specName = yiTags_[i].name();
      const Coal::GasSpeciesName spec     = Coal::gas_name_to_enum( specName );

      TagList coalSrcTags = coalInterface_->gas_species_source_term(spec,false);

      if( coalSrcTags.empty() ) continue;

      proc0cout<<"\nregistering particle-to-cell source expression for " << specName << std::endl;

      const Expr::Tag p2cSpeciesSrcTag = get_p2c_src_tag( specName );
      factory_.register_expression( new P2CBuilder( p2cSpeciesSrcTag,
                                                    coalSrcTags,
                                                    pSizeTag_,
                                                    pPosTags_ ) );

      factory_.attach_dependency_to_expression( p2cSpeciesSrcTag,
                                                get_rhs_tag( "rho_" + specName ),
                                                Expr::ADD_SOURCE_EXPRESSION );
    }

    const std::string tarName = tagNames_.tar.name();
    const Tag p2cTarSrcTag = get_p2c_src_tag( tarName );
    factory_.register_expression( new P2CBuilder( p2cTarSrcTag,
                                                  coalInterface_->tar_source_term(),
                                                  pSizeTag_,
                                                  pPosTags_ ) );

    factory_.attach_dependency_to_expression( p2cTarSrcTag,
                                              get_rhs_tag( "rho_" + tarName ),
                                              Expr::ADD_SOURCE_EXPRESSION );

    // Connect coal species source terms to coal species RHSs *****************************//
    const TagList coalSpeciesTags = coalInterface_->gas_species_source_terms();
    Uintah::ProblemSpecP constDensityParams = wasatchSpec_->findBlock("Density")
                                                              ->findBlock("Constant");

    if( constDensityParams ){
      proc0cout<<"\n\n"
               <<"***************** WARNING ****************************\n"
               <<"Constant density implemented.\n"
               <<"Partcle-to-gas source terms for density are ignored.  \n"
               <<"******************************************************\n";
    }
    else if( !coalSpeciesTags.empty() ){
      const Tag p2cDensitySrcTag = get_p2c_src_tag( gDensTag_.name() );
      factory_.register_expression( new P2CBuilder( p2cDensitySrcTag,
                                                    coalSpeciesTags,
                                                    pSizeTag_,
                                                    pPosTags_ ) );

      factory_.attach_dependency_to_expression( p2cDensitySrcTag,
                                                get_rhs_tag( gDensTag_.name() ),
                                                Expr::ADD_SOURCE_EXPRESSION );
    }

    // Add production rates from coal rxns to particle mass *******************************//
    const Expr::TagList pMassSrcTags = coalInterface_->particle_mass_rhs_taglist();

    if( !pMassSrcTags.empty() ){
      for( size_t i=0; i<pMassSrcTags.size(); ++i ){
        factory_.attach_dependency_to_expression( pMassSrcTags[i],
                                                  get_rhs_tag( pMassTag_.name() ),
                                                  Expr::ADD_SOURCE_EXPRESSION);
      }
    }
    // Connect coal energy source term to coal gas energy RHS ***************************//
    const Tag coalTempSrcTag = coalInterface_->particle_temperature_rhs_tag();
    if( coalTempSrcTag  != Tag() ){
      factory_.attach_dependency_to_expression( coalTempSrcTag,
                                                get_rhs_tag( pTempTag_.name() ),
                                                Expr::ADD_SOURCE_EXPRESSION );

      // Connect the coal energy term to gas phase energy RHS ****************************//
      const Tag coalEnergyToGasTag = coalInterface_->heat_released_to_gas_tag();
      if( coalEnergyToGasTag!= Tag() ){
        const Tag p2cEnergySrcTag=get_p2c_src_tag( coalEnergyToGasTag.name() );

        factory_.register_expression( new P2CBuilder( p2cEnergySrcTag,
                                                      coalEnergyToGasTag,
                                                      pSizeTag_,
                                                      pPosTags_ ) );

        const Tag gasEnergyTag = tagNames_.totalinternalenergy;
        factory_.attach_dependency_to_expression( p2cEnergySrcTag,
                                                  get_rhs_tag( gasEnergyTag.name() ),
                                                  Expr::ADD_SOURCE_EXPRESSION );

      }
    }
  }

  EquationAdaptors
  SetupCoalModels::get_adaptors()
  {
    if( !equationsSet_ ){
      throw Uintah::ProblemSetupException( "\nVector of equations must be populated before it is used. \n", __FILE__, __LINE__ );
    }
    return adaptors_;
  }

  const Expr::Tag
  SetupCoalModels::get_rhs_tag( std::string fieldName ){
    return Expr::Tag(fieldName + "_rhs", Expr::STATE_NONE );
  }

  const Expr::Tag
  SetupCoalModels::get_c2p_tag( std::string fieldName ){
    return Expr::Tag("c2p_" + fieldName, Expr::STATE_NONE );
  }

  const Expr::Tag
  SetupCoalModels::get_p2c_src_tag( std::string fieldName ){
    return Expr::Tag("p2cSrc_" + fieldName, Expr::STATE_NONE );
  }

} // namespace WasatchCore

