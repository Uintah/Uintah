#include "CPDInterface.h"

#include <stdexcept>
#include <sstream>

#include <boost/foreach.hpp>

#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/Nebo.h>

#include <expression/ClipValue.h>

// expressions we build here
#include "L_RHS.h"
#include "CPDData.h"
#include "C_RHS.h"
#include "Gi_RHS.h"
#include "Deltai_RHS.h"
#include "c0_fun.h"
#include "kb.h"
#include "kg_i.h"
#include "dy_gi.h"
#include "MvRHS.h"
#include "Mv.h"
#include "TarProductionRate.h"

using std::vector;
using std::cout;
using std::endl;
using std::ostringstream;
using std::string;

namespace CPD{

//------------------------------------------------------------------

vector<string> taglist_names( const Expr::TagList& tags )
{
  vector<string> fieldNames;
  for(  Expr::TagList::const_iterator it=tags.begin(); it!=tags.end(); ++it ){
    fieldNames.push_back( it->name() );
  }
  return fieldNames;
}
//------------------------------------------------------------------
	
CPDSpecies gas_dev2cpd( const DEV::DEVSpecies cspec )
{
  CPDSpecies s;
  switch( cspec ){
    case DEV::CO2:  s=CO2;             break;
    case DEV::H2O:  s=H2O;             break;
    case DEV::CO :  s=CO;              break;
    case DEV::HCN:  s=HCN;             break;
    case DEV::NH3:  s=NH3;             break;
    case DEV::CH4:  s=CH4;             break;
    case DEV::H  :  s=H;               break;
    default:   s=INVALID_SPECIES; break;
  }
  return s;
}
	
//--------------------------------------------------------------------

template< typename FieldT >
CPDInterface<FieldT>::CPDInterface( GraphCategories& gc,
                                    const CoalType   coalType,
                                    const Expr::Tag  pTempTag,
                                    const Expr::Tag  pMassTag,
                                    const Expr::Tag  pMass0Tag )
  : DEV::DevolatilizationBase(),
    cpdInfo_   ( coalType ),
    coalComp_  ( cpdInfo_.get_coal_composition() ),
    c0_        ( c0_fun( coalComp_.get_C(), coalComp_.get_O() )),
    vMassFrac0_( coalComp_.get_vm() ),
    tar0_      ( tar_0(cpdInfo_, c0_) ),
    pTempTag_  ( pTempTag  ),
    pMassTag_  ( pMassTag  ),
    pMass0Tag_ ( pMass0Tag ),
    kbTag_     ( Coal::StringNames::self().cpd_kb, Expr::STATE_NONE ),
    sNames_    ( Coal::StringNames::self() ),
    gc_        ( gc )
{
  if( pTempTag == Expr::Tag() ){
    ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl
        << "temperature tag is invalid." << endl;
    throw std::runtime_error( msg.str() );
  }

  haveRegisteredExprs_ = false;

  parse_equations();
  set_tags();
  register_expressions();
}

//----------------------------------------------------------------------

template< typename FieldT >
void
CPDInterface<FieldT>::
parse_equations()
{
  gEqns_.    clear();
  deltaEqns_.clear();

  gTags_.    clear();
  deltaTags_.clear();

  g_rhsTags_.    clear();
  delta_rhsTags_.clear();

  const vector<double> delta0 = deltai_0(cpdInfo_, c0_ );

  /*
   * Parse equations for coal functional groups
   */
  const int nspec = cpdInfo_.get_nspec();
  for( int i=0; i<nspec; ++i ){
    ostringstream name1, name2;
    name1 << sNames_.cpd_g     << i;
    name2 << sNames_.cpd_delta << i;

    gEqns_.    push_back( new Coal::CoalEquation(name1.str(), 0.0, gc_) );
    deltaEqns_.push_back( new Coal::CoalEquation(name2.str(), pMassTag_, delta0[i] , gc_) );

    gTags_.    push_back( gEqns_    [i]->solution_variable_tag() );
    deltaTags_.push_back( deltaEqns_[i]->solution_variable_tag() );

    g_rhsTags_.    push_back( gEqns_    [i]->rhs_tag() );
    delta_rhsTags_.push_back( deltaEqns_[i]->rhs_tag() );
  }

  /*
   * Parse equations for volatiles, tar, labile bridge, and labile bridge population.
   */
  const double lFrac0 = cpdInfo_.get_l0_mass() * vMassFrac0_; // initial mass fraction of labile bridge in the particle.

  volatilesEqn_    = new Coal::CoalEquation( sNames_.dev_mv, pMassTag_, vMassFrac0_*(1.0-c0_), gc_ );
  tarEqn_          = new Coal::CoalEquation( sNames_.cpd_tar, pMassTag_, vMassFrac0_*tar0_,    gc_ );
  lbEqn_           = new Coal::CoalEquation( sNames_.cpd_l,  pMassTag_, lFrac0,                gc_ );
  lbPopulationEqn_ = new Coal::CoalEquation( sNames_.cpd_lbPopulation, cpdInfo_.get_lbPop0(),  gc_ );


  // put pointers to equations in a vector
  eqns_.clear();
  eqns_.insert( eqns_.end(), gEqns_.begin(),     gEqns_.end()     );
  eqns_.insert( eqns_.end(), deltaEqns_.begin(), deltaEqns_.end() );
  eqns_.push_back( volatilesEqn_    );
  eqns_.push_back( tarEqn_          );
  eqns_.push_back( lbEqn_           );
  eqns_.push_back( lbPopulationEqn_ );
}
//----------------------------------------------------------------------
template< typename FieldT >
void
CPDInterface<FieldT>::
set_tags()
{
  // set tags for functional group rate constants
  const int nspec = cpdInfo_.get_nspec();
  for( int i=0; i<nspec; ++i ){

    ostringstream name1;
    name1 << sNames_.cpd_kg << i;
    kgTags_.push_back( Expr::Tag(name1.str(),Expr::STATE_NONE) );
  }

  // set tags for composition and species sources.
  speciesSrcTags_.clear();
  const SpeciesSum& specSum = SpeciesSum::self();
  const int ncomp = specSum.get_ncomp();
  for( int i=1; i<=ncomp; i++ ){

    ostringstream name2;
    name2 << sNames_.cpd_dy << i;
    speciesSrcTags_.push_back( Expr::Tag(name2.str(),Expr::STATE_NONE) );
  }

  // set tags needed for particle-to-cell source terms (and a few others)
  charSrcTag_      = Expr::Tag(sNames_.cpd_charProd_rhs, Expr::STATE_NONE);

  lbTag_           = lbEqn_           ->solution_variable_tag();
  lb_rhsTag_       = lbEqn_           ->rhs_tag();

  lbpTag_          = lbPopulationEqn_->solution_variable_tag();
  lbp_rhsTag_      = lbPopulationEqn_->rhs_tag();

  tarTag_          = tarEqn_          ->solution_variable_tag();
  tarSrcTag_       = tarEqn_          ->rhs_tag();

  volatilesTag_    = volatilesEqn_    ->solution_variable_tag();
  volatilesSrcTag_ = volatilesEqn_    ->rhs_tag();
}

//----------------------------------------------------------------------

template< typename FieldT >
void
CPDInterface<FieldT>::
register_expressions()
{
  Expr::ExpressionFactory& factory = *(gc_[WasatchCore::ADVANCE_SOLUTION]->exprFactory);

  factory.register_expression( new typename kb<FieldT>::
                                                Builder(kbTag_, pTempTag_, cpdInfo_));

  factory.register_expression( new typename kg_i<FieldT>::
                                                Builder(kgTags_, gTags_ ,pTempTag_, pMass0Tag_, cpdInfo_) );

  factory.register_expression( new typename TarProductionRate<FieldT>::
                                                Builder(tarSrcTag_, lbpTag_, lbp_rhsTag_,
                                                        pMass0Tag_, vMassFrac0_, tar0_, cpdInfo_ ) );

  factory.register_expression( new typename L_RHS<FieldT>::
                                                Builder(lb_rhsTag_, kbTag_, lbTag_ ) );

  factory.register_expression( new typename C_RHS<FieldT>::
                                                Builder(charSrcTag_,  kbTag_, lbTag_, cpdInfo_ ));

  factory.register_expression( new typename Gi_RHS<FieldT>::
                                                Builder(g_rhsTags_, kbTag_, kgTags_,deltaTags_,
                                                        lbTag_ ,cpdInfo_) );

  factory.register_expression( new typename Deltai_RHS<FieldT>::
                                                Builder(delta_rhsTags_, kbTag_, kgTags_, deltaTags_,
                                                        lbTag_ ,cpdInfo_) );

  factory.register_expression( new typename MvRHS<FieldT>::
                                                Builder(volatilesSrcTag_, charSrcTag_, speciesSrcTags_ ));

  factory.register_expression( new typename dy_gi<FieldT>::
                                                Builder(speciesSrcTags_, g_rhsTags_ ) );

  factory.register_expression( new typename L_RHS<FieldT>::
                                                Builder(lbp_rhsTag_, kbTag_, lbpTag_ ) );

  // adds tar production rate to volatiles production rate
  factory.attach_dependency_to_expression( tarSrcTag_, volatilesSrcTag_ );

  // Ensure non-negative values
  typedef Expr::ClipValue<FieldT> Clipper;
  BOOST_FOREACH( Coal::CoalEquation* const& eqn, eqns_ ){
    Expr::Tag tag = eqn->solution_variable_tag();
    const Expr::Tag clip( tag.name()+"_clip", Expr::STATE_NONE );
    factory.register_expression( new typename Clipper::Builder( clip, 0.0, 0.0, Clipper::CLIP_MIN_ONLY ) );
    factory.attach_modifier_expression( clip, tag );
  }

  haveRegisteredExprs_ = true;
}

//--------------------------------------------------------------------

template< typename FieldT >
const Expr::Tag
CPDInterface<FieldT>::
gas_species_src_tag( const DEV::DEVSpecies devspec ) const
{
  const CPDSpecies spec = gas_dev2cpd(devspec);
  if( spec == INVALID_SPECIES ) return Expr::Tag();
  return speciesSrcTags_[ spec ];
}

//====================================================================
// Explicit template instantiation
template class CPDInterface< SpatialOps::Particle::ParticleField >;
//====================================================================

} // namespace CPD
