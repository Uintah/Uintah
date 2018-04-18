#include <CCA/Components/Wasatch/Coal/Devolatilization/SingleRate/SingleRateInterface.h>

#include <stdexcept>
#include <sstream>

#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/Nebo.h>

#include <CCA/Components/Wasatch/Coal/Devolatilization/SingleRate/SingleRateData.h>


// expressions we build here

using std::vector;
using std::endl;
using std::ostringstream;
using std::string;

namespace SNGRATE{

//------------------------------------------------------------------	
  
  SNGRATE::SingleRateSpecies gas_dev2singlerate( const DEV::DEVSpecies cspec )
  {
    SNGRATE::SingleRateSpecies s;
    switch( cspec ){
      case DEV::CO:   s=CO;             break;
      case DEV::H2:   s=H2;             break;
      default:   s=INVALID_SPECIES;break;
    }
    return s;
  }
	
//--------------------------------------------------------------------

template< typename FieldT >
SingleRateInterface<FieldT>::
SingleRateInterface( GraphCategories& gc,
                     const Coal::CoalType& ct,
                     const Expr::Tag pTempTag,
                     const Expr::Tag pMassTag,
                     const Expr::Tag pMass0Tag,
                     const bool isDAE )
  : DEV::DevolatilizationBase(),
    gc_            ( gc        ),
    coalComp_      ( ct        ),
    pTempTag_      ( pTempTag  ),
    pMassTag_      ( pMassTag  ),
    pMass0Tag_     ( pMass0Tag ),
    isDAE_         ( isDAE     ),
    singleRateData_( coalComp_ ),
    sNames_        ( Coal::StringNames::self() )
{
	
  if( pTempTag == Expr::Tag() || pMassTag == Expr::Tag() || pMass0Tag ==Expr::Tag()){
    ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl
        << "Tags entered into SingleRateInterface are invalid." << endl;
    throw std::runtime_error( msg.str() );
  }

  parse_equations();
  set_tags();
  register_expressions();
}

//----------------------------------------------------------------------

template< typename FieldT >
void
SingleRateInterface<FieldT>::
parse_equations()
{
  volatilesEqn_ = new Coal::CoalEquation( sNames_.dev_mv, pMassTag_,
                                          coalComp_.get_vm(), gc_ );

  eqns_.clear();
  eqns_.push_back( volatilesEqn_ );
}

//----------------------------------------------------------------------

template< typename FieldT >
void
SingleRateInterface<FieldT>::
set_tags()
{
  // Base variables
  volatilesSrcTag_   = volatilesEqn_->rhs_tag();
  volatilesTag_      = volatilesEqn_->solution_variable_tag();
  tarSrcTag_      = Expr::Tag(sNames_.singlerate_tar_src, Expr::STATE_NONE );
  charSrcTag_     = Expr::Tag(sNames_.dev_char,           Expr::STATE_NONE );


  speciesSrcTags_.clear();
  speciesSrcTags_.push_back(Expr::Tag("singlerate_CO_rhs", Expr::STATE_NONE));
  speciesSrcTags_.push_back(Expr::Tag("singlerate_H2_rhs", Expr::STATE_NONE));

  mvCharTags_.clear();
  mvCharTags_.push_back(volatilesSrcTag_); // 0- Volatile RHS
  mvCharTags_.push_back( speciesSrcTags_[0]     );  // 1
  mvCharTags_.push_back( speciesSrcTags_[1]     );  // 2
  mvCharTags_.push_back( tarSrcTag_ );  // 3

  haveRegisteredExprs_ = false;
}

//----------------------------------------------------------------------

template< typename FieldT >
void
SingleRateInterface<FieldT>::
register_expressions()
{
  Expr::ExpressionFactory& factory = *(gc_[WasatchCore::ADVANCE_SOLUTION]->exprFactory);

  volatilesSrcID_ =
  factory.register_expression( new typename SingleRateModel<FieldT>::
                                                Builder(mvCharTags_, pTempTag_, volatilesTag_,
                                                        pMass0Tag_, coalComp_.get_vm(), singleRateData_, isDAE_));

  // Char will be registered as a Constant Experssion
  factory.register_expression( new typename Expr::ConstantExpr<FieldT>::Builder( charSrcTag_, 0.0));

  haveRegisteredExprs_ = true;
}

//--------------------------------------------------------------------

template< typename FieldT >
const Expr::Tag
SingleRateInterface<FieldT>::
gas_species_src_tag( const DEV::DEVSpecies devspec ) const
{
  const SingleRateSpecies spec = gas_dev2singlerate(devspec);
  if( spec == INVALID_SPECIES ) return Expr::Tag();
  return speciesSrcTags_[ spec ];
}

//------------------------------------------------------------------

//====================================================================
// Explicit template instantiation
template class SingleRateInterface< SpatialOps::Particle::ParticleField >;
//====================================================================

} // namespace SNGRATE
