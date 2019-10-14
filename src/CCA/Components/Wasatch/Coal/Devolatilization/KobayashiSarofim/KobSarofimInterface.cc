#include <CCA/Components/Wasatch/Coal/Devolatilization/KobayashiSarofim/KobSarofimInterface.h>

#include <stdexcept>
#include <sstream>

#include <CCA/Components/Wasatch/TimeStepper.h>

#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/Nebo.h>

// expressions we build here

using std::vector;
using std::endl;
using std::ostringstream;
using std::string;

namespace SAROFIM{

//------------------------------------------------------------------	
  
KobSarofimSpecies gas_dev2sarrofim( const DEV::DEVSpecies cspec )
{
  KobSarofimSpecies s;
  switch( cspec ){
    case DEV::CO:   s=CO;             break;
    case DEV::H2:   s=H2;             break;
    default:   s=INVALID_SPECIES;break;
  }
  return s;
}
	
//------------------------------------------------------------------
template< typename FieldT >
KobSarofimInterface<FieldT>::
KobSarofimInterface( WasatchCore::GraphCategories& gc,
                     const Coal::CoalType& ct,
                     const Expr::Tag pTempTag,
                     const Expr::Tag pMassTag,
                     const Expr::Tag pMass0Tag )
  : DEV::DevolatilizationBase(),
    gc_         ( gc        ),
    coalComp_   ( ct        ),
    pTempTag_   ( pTempTag  ),
    pMassTag_   ( pMassTag  ),
    pMass0Tag_  ( pMass0Tag ),
    sarofimData_( coalComp_ ),
    sNames_     ( Coal::StringNames::self() )
{
	
  if( pTempTag == Expr::Tag() || pMassTag == Expr::Tag() || pMass0Tag ==Expr::Tag()){
    ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl
        << "Tags entered into KobSarofimInterface are invalid." << endl;
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
KobSarofimInterface<FieldT>::
parse_equations()
{
  volatilesEqn_ = new Coal::CoalEquation( sNames_.dev_mv, pMassTag_,
                                          coalComp_.get_vm(), gc_ );

  hydrogenEqn_  = new Coal::CoalEquation( "Hydrogen_Volatile_Element",
                                          sarofimData_.get_hydrogen_coefficient(), gc_ );

  oxygenEqn_    = new Coal::CoalEquation( "Oxygen_Volatile_Element",
                                          sarofimData_.get_oxygen_coefficient(),   gc_ );
  eqns_.clear();
  eqns_.push_back( volatilesEqn_ );
  eqns_.push_back( hydrogenEqn_  );
  eqns_.push_back( oxygenEqn_    );

  eqnsParsed_ = true;
}

//--------------------------------------------------------------------

template< typename FieldT >
void
KobSarofimInterface<FieldT>::
set_tags()
{
  volatilesTag_    = volatilesEqn_->solution_variable_tag();
  volatilesSrcTag_ = volatilesEqn_->rhs_tag();

  charSrcTag_      = Expr::Tag( sNames_.dev_char,         Expr::STATE_NONE );
  tarSrcTag_       = Expr::Tag( sNames_.sarofim_tar_src,  Expr::STATE_NONE );

  elementTags_.clear();
  elementTags_.push_back( hydrogenEqn_->solution_variable_tag() );
  elementTags_.push_back( oxygenEqn_  ->solution_variable_tag() );

  dElementTags_.clear();
  dElementTags_.push_back( hydrogenEqn_->rhs_tag() );
  dElementTags_.push_back( oxygenEqn_  ->rhs_tag() );

  speciesSrcTags_.push_back( Expr::Tag("sarofim_CO_rhs",   Expr::STATE_NONE) );
  speciesSrcTags_.push_back( Expr::Tag("sarofim_H2_rhs",   Expr::STATE_NONE) );

  mvCharTags_.clear();
  mvCharTags_.push_back( volatilesSrcTag_         ); // 0  Volatile RHS
  mvCharTags_.push_back( charSrcTag_              ); // 1  Char RHS

  mvCharTags_.push_back( speciesSrcTags_[0]    );  // 2 CO   Consumption rate ( for gas Phase )
  mvCharTags_.push_back( speciesSrcTags_[1]    );  // 3 H2   Consumption rate ( for gas Phase )
  mvCharTags_.push_back( tarSrcTag_            );  // 4 tar  Consumption rate ( for gas Phase )

  mvCharTags_.push_back( dElementTags_[0]);        // 4 Hydrogen Consumption rate ( for gas Phase )
  mvCharTags_.push_back( dElementTags_[1]);        // 5 Oxygen Consumption rate   ( for gas Phase )
}

//--------------------------------------------------------------------

template< typename FieldT >
void
KobSarofimInterface<FieldT>::
register_expressions()
{
  Expr::ExpressionFactory& factory = *(gc_[WasatchCore::ADVANCE_SOLUTION]->exprFactory);

  volatilesSrcID_   = factory.register_expression( new typename KobayashiSarofim <FieldT>::
                                                       Builder(mvCharTags_, pTempTag_, volatilesTag_,
                                                               elementTags_, sarofimData_ ));

  haveRegisteredExprs_ = true;
}

//--------------------------------------------------------------------

template< typename FieldT >
const Expr::Tag
KobSarofimInterface<FieldT>::
gas_species_src_tag( const DEV::DEVSpecies devspec ) const
{
	const KobSarofimSpecies spec = gas_dev2sarrofim(devspec); 
  if( spec == INVALID_SPECIES ) return Expr::Tag();
  return speciesSrcTags_[spec ];
}

//--------------------------------------------------------------------
//====================================================================
// Explicit template instantiation
template class KobSarofimInterface< SpatialOps::Particle::ParticleField >;
//====================================================================

} // namespace SARROFIM
