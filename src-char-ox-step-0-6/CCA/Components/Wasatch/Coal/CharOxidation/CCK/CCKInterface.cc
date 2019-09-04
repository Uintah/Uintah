#include <stdexcept>
#include <sstream>

#include <expression/ClipValue.h>

#include <CCA/Components/Wasatch/Coal/Devolatilization/CPD/c0_fun.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>

// includes for char conversion kinetics (CCK) model
#include "CCK.h"
#include "LogFrequencyDistributionRHS.h"
#include "InitialDevolatilizedDensity.h"
#include "InitialCoreDensity.h"
#include "DevolatilizedDensity.h"
#include "DevolatilizedMassFracs.h"
#include "AshFilm.h"
#include "CoreDensity.h"
#include "CharConversion.h"
#include "CharSpeciesRHS.h"
#include "ParticleDiameter.h"
#include "ThermalAnnealing.h"
#include "CCKInterface.h"

using std::ostringstream;
using std::endl;

namespace CCK{

  template< typename FieldT >
  CCKInterface<FieldT>::
  CCKInterface( WasatchCore::GraphCategories& gc,
                const Tag& pDiamTag,
                const Tag& pTempTag,
                const Tag& gTempTag,
                const Tag& co2MassFracTag,
                const Tag& coMassFracTag,
                const Tag& o2MassFracTag,
                const Tag& h2MassFracTag,
                const Tag& h2oMassFracTag,
                const Tag& ch4MassFracTag,
                const Tag& mixMWTag,
                const Tag& gPressTag,
                const Tag& pMassTag,
                const Tag& pMass0Tag,
                const Tag& pDens0Tag,
                const Tag& pDiam0Tag,
                const Tag& volatilesTag,
                const Coal::CoalType coalType,
                const DEV::DevModel devModel )
    : gc_                  ( gc                ),

      gTempTag_            ( gTempTag  ),
      mixMWTag_            ( mixMWTag  ),
      gPressTag_           ( gPressTag ),

      o2MassFracTag_       ( o2MassFracTag  ),
      h2oMassFracTag_      ( h2oMassFracTag ),
      h2MassFracTag_       ( h2MassFracTag  ),
      ch4MassFracTag_      ( ch4MassFracTag ),
      co2MassFracTag_      ( co2MassFracTag ),
      coMassFracTag_       ( coMassFracTag  ),
      co2_rhsTag_          ( Coal::StringNames::self().char_co2_rhs,            STATE_NONE ),
      co_rhsTag_           ( Coal::StringNames::self().char_co_rhs,             STATE_NONE ),
      o2_rhsTag_           ( Coal::StringNames::self().char_o2_rhs,             STATE_NONE ),
      h2o_rhsTag_          ( Coal::StringNames::self().char_h2o_rhs,            STATE_NONE ),
      h2_rhsTag_           ( Coal::StringNames::self().char_h2_rhs,             STATE_NONE ),
      ch4_rhsTag_          ( Coal::StringNames::self().char_ch4_rhs,            STATE_NONE ),

      pDiam0Tag_           ( pDiam0Tag    ),
      ashPorosityTag_      ( Coal::StringNames::self().ash_porosity,            STATE_NONE ),
      ashThicknessTag_     ( Coal::StringNames::self().ash_thickness,           STATE_NONE ),
      coreDiamTag_         ( Coal::StringNames::self().core_diameter,           STATE_NONE ),
      coreDensityTag_      ( Coal::StringNames::self().core_density,            STATE_NONE ),
      thermAnnealTag_      ( Coal::StringNames::self().therm_anneal,            STATE_NONE ),
      charConversionTag_   ( Coal::StringNames::self().char_conversion,         STATE_NONE ),
      ashDensityTag_       ( Coal::StringNames::self().ash_density,             STATE_NONE ),
      ashMassFracTag_      ( Coal::StringNames::self().ash_mass_frac,           STATE_NONE ),
      charMassFracTag_     ( Coal::StringNames::self().char_mass_frac,          STATE_NONE ),
      devolDensityTag_     ( "p_DevolatilizedDensity",                          STATE_NONE ),
      pDiamModTag_         ( "cck_p_size",                                      STATE_NONE ),
      devolDens0Tag_       ( "Initial_p_DevolatilizedDensity",                  STATE_NONE ),
      coreDens0Tag_        ( "Initial_"+Coal::StringNames::self().core_density, STATE_NONE ),
      pTempTag_            ( pTempTag     ),
      pDens0Tag_           ( pDens0Tag    ),
      pDiamTag_            ( pDiamTag     ),
      pMassTag_            ( pMassTag     ),
      pMass0Tag_           ( pMass0Tag    ),
      volatilesTag_        (volatilesTag       ),
      devolAshMassFracTag_ ( Coal::StringNames::self().ash_mass_frac +"_d",     STATE_NONE ),
      devolCharMassFracTag_( Coal::StringNames::self().char_mass_frac+"_d",     STATE_NONE ),
      sNames_              ( Coal::StringNames::self() ),
      initDevChar_         ( devModel == DEV::CPDM ),
      cckData_             ( coalType             )
  {
    parse_equations();
    set_tags();
    register_expressions();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  CCKInterface<FieldT>::
  parse_equations()
  {
    // Get initial mass fraction of char within coal volatiles
    double c0 = 0.0;
    if( initDevChar_ ){
      c0 = CPD::c0_fun(cckData_.get_C(), cckData_.get_O());
    }

     // Calculate initial mass fraction of char within coal
     double char0 = cckData_.get_fixed_C()+ cckData_.get_vm()*c0;

    std::cout << std::endl
              << "Initial char mass fraction in coal volatiles is : "
              << cckData_.get_vm()*c0
              << std::endl
              << "Initial mass fraction of char in coal is        : "
              << char0 << std::endl;

    charEqn_ = new Coal::CoalEquation( sNames_.char_mass, pMassTag_, char0, gc_ );

    // setup ODEs for frequency distribution used for thermal annealing model
    const CHAR::Vec eD    = cckData_.get_eD_vec();
    const double s  = cckData_.get_neD_std_dev();
    const double mu = cckData_.get_neD_mean();

    logFrequencyEqns_.clear();
    for( size_t i = 0; i<eD.size(); ++i){

      const double lnf = -pow(log(eD[i]) - mu, 2)/(2*pow(s,2))
                       - log(sqrt(2*PI)*s*eD[i]);

      std::ostringstream frqName;
      frqName    << sNames_.log_freq_dist << "_" << i;

      logFrequencyEqns_.push_back( new Coal::CoalEquation(frqName.str(), lnf, gc_ ) );
    }

    // collect equations into a single vector
    eqns_.clear();
    eqns_.push_back( charEqn_ );
    eqns_.insert( eqns_.end(), logFrequencyEqns_.begin(), logFrequencyEqns_.end() );
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  CCKInterface<FieldT>::
  set_tags()
  {
    charMassTag_      = charEqn_->solution_variable_tag();
    charMass_rhsTag_  = charEqn_->rhs_tag();
    heteroCo2Tag_     = Tag( sNames_.char_gasifco2,   STATE_NONE );
    heteroH2oTag_     = Tag( sNames_.char_gasifh2o,   STATE_NONE );
    oxidation_rhsTag_ = Tag( sNames_.char_oxid_rhs,   STATE_NONE );
    co2CoRatioTag_    = Tag( sNames_.char_coco2ratio, STATE_NONE );

    speciesSrcTags_.clear();
    speciesSrcTags_.push_back( co2_rhsTag_ );
    speciesSrcTags_.push_back( co_rhsTag_  );
    speciesSrcTags_.push_back( o2_rhsTag_  );
    speciesSrcTags_.push_back( h2_rhsTag_  );
    speciesSrcTags_.push_back( h2o_rhsTag_ );

    // order here matters: [ CO2, CO, O2, H2, H2O, CH4 ]
    massFracTags_.clear();
    massFracTags_.push_back( co2MassFracTag_ );
    massFracTags_.push_back( coMassFracTag_  );
    massFracTags_.push_back( o2MassFracTag_  );
    massFracTags_.push_back( h2MassFracTag_  );
    massFracTags_.push_back( h2oMassFracTag_ );
    massFracTags_.push_back( ch4MassFracTag_ );

    // gas property tags
    gasTags_.clear();
    gasTags_.push_back( gTempTag_       ); // 0
    gasTags_.push_back( gPressTag_      ); // 1
    gasTags_.push_back( mixMWTag_       ); // 2

    particleTags_.clear();
    particleTags_.push_back( pTempTag_          ); //0
    particleTags_.push_back( pMass0Tag_         ); //1
    particleTags_.push_back( pMassTag_          ); //2
    particleTags_.push_back( charMassTag_       ); //3
    particleTags_.push_back( charConversionTag_ ); //4
    particleTags_.push_back( coreDensityTag_    ); //5
    particleTags_.push_back( coreDiamTag_       ); //6
    particleTags_.push_back( pDiamTag_          ); //7
    particleTags_.push_back( ashPorosityTag_    ); //8
    particleTags_.push_back( ashThicknessTag_   ); //9
    particleTags_.push_back( thermAnnealTag_    ); //10

    // char depletion tags
    charDepletionTags_.clear();
    // these are in (kg char consumed by species i)/sec
    charDepletionTags_.push_back( heteroCo2Tag_                                 );
    charDepletionTags_.push_back( Tag("char_Depletion_CO", STATE_NONE ) );
    charDepletionTags_.push_back( oxidation_rhsTag_                              );
    charDepletionTags_.push_back( Tag(sNames_.char_gasifh2,  STATE_NONE ) );
    charDepletionTags_.push_back( heteroH2oTag_                                 );

    // CO2-CO ratio (mole basis)
    charDepletionTags_.push_back( co2CoRatioTag_ );

    // RHS tags (kg/s)
    char_rhsTags_.clear();
    char_rhsTags_.push_back( co2_rhsTag_      ); //CO2
    char_rhsTags_.push_back( co_rhsTag_       ); //CO
    char_rhsTags_.push_back( o2_rhsTag_       ); //O2
    char_rhsTags_.push_back( h2_rhsTag_       ); //H2
    char_rhsTags_.push_back( h2o_rhsTag_      ); //H2O
    char_rhsTags_.push_back( ch4_rhsTag_      ); //H2O
    char_rhsTags_.push_back( charMass_rhsTag_ ); //char

    // tags for log(freqency distribution) RHSs
    logFreqDistTags_.clear(); logFreqDistRHSTags_.clear();
    for( size_t i = 0; i<logFrequencyEqns_.size(); ++i )
    {
      logFreqDistTags_.   push_back( logFrequencyEqns_[i]->solution_variable_tag() );
      logFreqDistRHSTags_.push_back( logFrequencyEqns_[i]->rhs_tag()               );
    }
  }
  //------------------------------------------------------------------

  template< typename FieldT >
  void
  CCKInterface<FieldT>::
  register_expressions()
  {
    Expr::ExpressionFactory& factory = *(gc_[WasatchCore::ADVANCE_SOLUTION]->exprFactory);

    factory.register_expression( new typename CCK::InitialDevolatilizedDensity<FieldT>::
                                 Builder(devolDens0Tag_, pDens0Tag_, cckData_ ));

    factory.register_expression( new typename CCK::DevolatilizedDensity<FieldT>::
                                 Builder( devolDensityTag_, coreDensityTag_, ashMassFracTag_,
                                          devolAshMassFracTag_, charMassFracTag_, ashDensityTag_,
                                          cckData_ ));

    factory.register_expression( new typename CCK::ParticleDiameter<FieldT>::
                                 Builder( pDiamModTag_,  charMassTag_, pMass0Tag_,
                                          devolDensityTag_, devolDens0Tag_, pDiam0Tag_,
                                          cckData_ ));

    factory.attach_modifier_expression( pDiamModTag_,
                                        pDiamTag_,
                                        ALL_PATCHES, true) ;

    factory.register_expression( new typename CCK::InitialCoreDensity<FieldT>::
                                 Builder( coreDens0Tag_, devolDens0Tag_, pDiam0Tag_, cckData_ ));

    factory.register_expression( new typename CCK::CoreDensity<FieldT>::
                                 Builder( coreDensityTag_, charConversionTag_, coreDens0Tag_,
                                          cckData_ ));

    factory.register_expression( new typename CCK::AshFilm<FieldT>::
                                 Builder( Expr::tag_list(ashPorosityTag_, ashThicknessTag_, coreDiamTag_),
                                          pDiamTag_, devolDensityTag_, devolDens0Tag_, devolAshMassFracTag_,
                                          cckData_ ));

    factory.register_expression( new typename CCK::DevolatilizedMassFracs<FieldT>::
                                 Builder( Expr::tag_list( charMassFracTag_,      ashMassFracTag_,
                                                          devolCharMassFracTag_, devolAshMassFracTag_ ),
                                                          pMass0Tag_, pMassTag_, charMassTag_, cckData_ ));

    factory.register_expression( new typename Expr::ConstantExpr<FieldT>::
                                 Builder( ashDensityTag_,
                                          cckData_.get_nonporous_ash_density()
                                         *(1 - cckData_.get_min_ash_porosity()) ) );

    factory.register_expression( new typename CCK::LogFrequencyDistributionRHS<FieldT>::
                                 Builder( logFreqDistRHSTags_, pTempTag_,       pMass0Tag_,
                                          pMassTag_,          volatilesTag_, cckData_ ));

    factory.register_expression( new typename CCK::ThermalAnnealing<FieldT>::
                                 Builder( thermAnnealTag_, logFreqDistTags_, cckData_ ));

    factory.register_expression( new typename CCK::CCKModel<FieldT>::
                                 Builder( charDepletionTags_,  massFracTags_, gasTags_, particleTags_,
                                          cckData_ ) );

    factory.register_expression( new typename CCK::CharConversion<FieldT>::
                                 Builder( charConversionTag_, charMassTag_, pMass0Tag_, cckData_ ));

    factory.register_expression( new typename CCK::CharSpeciesRHS<FieldT>::
                                 Builder( char_rhsTags_, charDepletionTags_, cckData_ ));
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  const Tag
  CCKInterface<FieldT>::
  gas_species_src_tag( const CHAR::CharGasSpecies spec ) const
  {
    if( spec == CHAR::O2 ) return o2_rhsTag_;
    if( spec == CHAR::CO2) return co2_rhsTag_;
    if( spec == CHAR::CO ) return co_rhsTag_;
    if( spec == CHAR::H2 ) return h2_rhsTag_;
    if( spec == CHAR::H2O) return h2o_rhsTag_;
    if( spec == CHAR::CH4) return ch4_rhsTag_;
    return Tag();
  }

//==========================================================================
// Explicit template instantiation for supported versions of this expression
template class CCKInterface< SpatialOps::Particle::ParticleField >;
//==========================================================================


} // namespace CCK
