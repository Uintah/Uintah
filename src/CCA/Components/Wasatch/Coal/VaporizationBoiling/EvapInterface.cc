#include <stdexcept>
#include <sstream>

#include <CCA/Components/Wasatch/Coal/StringNames.h>

#include <CCA/Components/Wasatch/Coal/VaporizationBoiling/EvapInterface.h>
#include <CCA/Components/Wasatch/Coal/VaporizationBoiling/Vaporization_RHS.h>

#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/Nebo.h>

#include <cantera/thermo/VPStandardStateTP.h>
#include <cantera/thermo/PDSS_Water.h>

#include <expression/ClipValue.h>

using std::ostringstream;
using std::endl;

namespace EVAP{

  //------------------------------------------------------------------

  template< typename FieldT >
  EvapInterface<FieldT>::
  EvapInterface( WasatchCore::GraphCategories& gc,
                 const Expr::Tag tempGTag,
                 const Expr::Tag tempPTag,
                 const Expr::Tag diamPTag,
                 const Expr::Tag rePTag,
                 const Expr::Tag scGTag,
                 const Expr::Tag waterMasFracTag,
                 const Expr::Tag totalMWTag,
                 const Expr::Tag gasPressureTag,
                 const Expr::Tag pMassTag,
                 const Coal::CoalType ct )
    : gc_             ( gc              ),
      coalComp_       ( ct              ),
      ct_             ( ct              ),
      tempGTag_       ( tempGTag        ),
      tempPTag_       ( tempPTag        ),
      diamPTag_       ( diamPTag        ),
      rePTag_         ( rePTag          ),
      scGTag_         ( scGTag          ),
      waterMasFracTag_( waterMasFracTag ),
      totalMWTag_     ( totalMWTag      ),
      gasPressureTag_ ( gasPressureTag  ),
      pMassTag_       ( pMassTag        ),
      eqnsParsed_     ( false           )
  {
    ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl;
    bool foundError = false;

    if( tempGTag        == Expr::Tag() ){ foundError=true;  msg << "gas temperature tag is invalid." << endl; }
    if( tempPTag        == Expr::Tag() ){ foundError=true;  msg << "particle temperature tag is invalid" << endl; }
    if( diamPTag        == Expr::Tag() ){ foundError=true;  msg << "particle diameter tag is invalid" << endl; }
    if( rePTag          == Expr::Tag() ){ foundError=true;  msg << "particle reynolds number tag is invalid" << endl; }
    if( scGTag          == Expr::Tag() ){ foundError=true;  msg << "schmidt number tag is invalid" << endl; }
    if( waterMasFracTag == Expr::Tag() ){ foundError=true;  msg << "water mass fraction tag is invalid" << endl; }
    if( totalMWTag      == Expr::Tag() ){ foundError=true;  msg << "total molecular weight tag is invalid" << endl; }
    if( gasPressureTag  == Expr::Tag() ){ foundError=true;  msg << "gas pressure tag is invalid" << endl; }
    if( pMassTag        == Expr::Tag() ){ foundError=true;  msg << "particle mass tag is invalid" << endl; }

    if( foundError ){
      throw std::runtime_error( msg.str() );
    }

    parse_equations();
    register_expressions();

  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  EvapInterface<FieldT>::
  parse_equations()
  {
    proc0cout << "\n\nEvapInterface::parse_equations() called. \n\n";
    const Coal::StringNames& sNames = Coal::StringNames::self();;
    evapEqn_ = new Coal::CoalEquation( sNames.moisture, pMassTag_, coalComp_.get_moisture(), gc_ );

    eqns_.clear();
    eqns_.push_back( evapEqn_ );

    eqnsParsed_ = true;
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  EvapInterface<FieldT>::
  register_expressions()
  {
    if( !eqnsParsed_ ){
      ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << endl <<endl
          << "EvapInterface::parse_equations() must be run before"
          << " evaporation expressions are registered \n\n.";

      throw std::runtime_error( msg.str() );
    }

    Expr::ExpressionFactory& factory = *(gc_[WasatchCore::ADVANCE_SOLUTION]->exprFactory);

    factory.register_expression(
        new typename Vaporization_RHS<FieldT>::Builder( evapEqn_->rhs_tag(),
                                                        tempGTag_, tempPTag_, diamPTag_, rePTag_,
                                                        scGTag_, waterMasFracTag_, totalMWTag_,
                                                        gasPressureTag_, evapEqn_->solution_variable_tag() ),
                                                        true );

    // ensure that values do not become negative.
    const Expr::Tag clip( evapEqn_->solution_variable_tag().name()+"_clip", Expr::STATE_NONE );
    typedef Expr::ClipValue<FieldT> Clipper;
    factory.register_expression( new typename Clipper::Builder( clip, 0.0, 0.0, Clipper::CLIP_MIN_ONLY ) );
    factory.attach_modifier_expression( clip, evapEqn_->solution_variable_tag() );
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  Expr::Tag
  EvapInterface<FieldT>::
  gas_species_src_tag( const EvapSpecies spec ) const
  {
    if( spec == INVALID_SPECIES ) return Expr::Tag();
    return evapEqn_->rhs_tag();
  }

  //------------------------------------------------------------------

  //========================================================================
  // Explicit template instantiation for supported versions of this expression
  template class EvapInterface< SpatialOps::Particle::ParticleField >;
  //========================================================================


} // namespace EVAP
