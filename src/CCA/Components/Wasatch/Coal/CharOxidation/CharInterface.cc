#include "CharInterface.h"

#include <expression/ClipValue.h>

#include <stdexcept>
#include <sstream>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

using std::ostringstream;
using std::endl;
using std::cout;

namespace CHAR{

  template< typename FieldT >
  CharInterface<FieldT>::
  CharInterface( WasatchCore::GraphCategories& gc,
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
                 const Tag& pDensTag,
                 const Tag& gPressTag,
                 const Tag& pMassTag,
                 const Tag& pMass0Tag,
                 const Tag& pDens0Tag,
                 const Tag& pDiam0Tag,
                 const Tag& volatilesTag,
                 const Coal::CoalType coalType,
                 const DEV::DevModel  dvmodel,
                 const CharModel      chmodel )
  : gc_( gc )
  {

    // Ensure tags passed to CharInterface are valid.
    bool foundError = false;
    ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl;
    if( pDiamTag          == Tag() ){ foundError=true;  msg << "Particle diameter tag is invalid." << endl; }
    if( pMassTag          == Tag() ){ foundError=true;  msg << "Particle mass tag is invalid" << endl; }
    if( pDens0Tag         == Tag() ){ foundError=true;  msg << "Initial particle mass tag is invalid" << endl; }
    if( pDensTag          == Tag() ){ foundError=true;  msg << "Particle density tag is invalid" << endl; }
    if( pDens0Tag         == Tag() ){ foundError=true;  msg << "Initial particle density tag is invalid" << endl; }
    if( volatilesTag      == Tag() ){ foundError=true;  msg << "Volatiles tag is invalid" << endl; }
    if( pTempTag          == Tag() ){ foundError=true;  msg << "Particle temperature tag is invalid." << endl; }
    if( gTempTag          == Tag() ){ foundError=true;  msg << "Gas temperature tag is invalid." << endl; }
    if( o2MassFracTag     == Tag() ){ foundError=true;  msg << "O2 mass fraction tag is invalid" << endl; }
    if( co2MassFracTag    == Tag() ){ foundError=true;  msg << "CO2 mass fraction tag is invalid" << endl; }
    if( coMassFracTag     == Tag() ){ foundError=true;  msg << "CO mass fraction tag is invalid" << endl; }
    if( h2MassFracTag     == Tag() ){ foundError=true;  msg << "H2 mass fraction tag is invalid" << endl; }
    if( h2oMassFracTag    == Tag() ){ foundError=true;  msg << "H2O mass fraction tag is invalid" << endl; }
    if( ch4MassFracTag    == Tag() ){ foundError=true;  msg << "CH4 mass fraction tag is invalid" << endl; }
    if( mixMWTag          == Tag() ){ foundError=true;  msg << "mixture molecular weight tag is invalid" << endl; }
    if( gPressTag         == Tag() ){ foundError=true;  msg << "Pressure tag is invalid" << endl; }

    if( foundError ) throw std::invalid_argument( msg.str() );

    switch (chmodel) {
    case LH:
    case FRACTAL:
      lhModel_ = new LH::LangmuirInterface<FieldT>
                     ( gc,
                       pDiamTag,
                       pTempTag,
                       gTempTag,
                       co2MassFracTag,
                       o2MassFracTag,
                       h2oMassFracTag,
                       mixMWTag,
                       pDensTag,
                       gPressTag,
                       pMassTag,
                       pMass0Tag,
                       coalType,
                       dvmodel,
                       chmodel );

      charModel_ = lhModel_;
      break;

    case FIRST_ORDER:
      firstOrderModel_ = new FOA::FirstOrderInterface<FieldT>
                     ( gc,
                       pDiamTag,
                       pTempTag,
                       gTempTag,
                       co2MassFracTag,
                       coMassFracTag,
                       o2MassFracTag,
                       h2MassFracTag,
                       h2oMassFracTag,
                       ch4MassFracTag,
                       mixMWTag,
                       pDensTag,
                       gPressTag,
                       pMassTag,
                       pMass0Tag,
                       coalType,
                       dvmodel );

      charModel_ = firstOrderModel_;
      break;

    case CCK:
      cckModel_ = new CCK::CCKInterface<FieldT>
                     ( gc,
                       pDiamTag,
                       pTempTag,
                       gTempTag,
                       co2MassFracTag,
                       coMassFracTag,
                       o2MassFracTag,
                       h2MassFracTag,
                       h2oMassFracTag,
                       ch4MassFracTag,
                       mixMWTag,
                       gPressTag,
                       pMassTag,
                       pMass0Tag,
                       pDens0Tag,
                       pDiam0Tag,
                       volatilesTag,
                       coalType,
                       dvmodel );

      charModel_ = cckModel_;
      break;

    case INVALID_CHARMODEL:
      throw std::invalid_argument( "Invalid model in CharInterface constructor" );
    }

    clip_char_mass();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void
  CharInterface<FieldT>::
  clip_char_mass()
  {
   Expr::ExpressionFactory& factory = *(gc_[WasatchCore::ADVANCE_SOLUTION]->exprFactory);
   cout<<std::endl<<"registering char mass clipping expression...     \n";

   // Ensure that values of Char do not become negative.
   const Tag charMassTag = charModel_->char_mass_tag();
   const Tag charMassClip( charMassTag.name()+"_clip", STATE_NONE );
   typedef Expr::ClipValue<FieldT> Clipper;
   factory.register_expression( new typename Clipper::
                                Builder( charMassClip, 0.0, 0.0, Clipper::CLIP_MIN_ONLY ) );

   factory.attach_modifier_expression( charMassTag, charMassClip );
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  const TagList&
  CharInterface<FieldT>::
  gas_species_src_tags()
  {
    return charModel_->gas_species_src_tags();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  Tag
  CharInterface<FieldT>::
  gas_species_src_tag( const CharGasSpecies spec )
  {
    return charModel_->gas_species_src_tag(spec);
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  const Tag&
  CharInterface<FieldT>::
  char_mass_tag()
  {
    return charModel_->char_mass_tag();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  const Tag&
  CharInterface<FieldT>::
  char_consumption_rate_tag()
  {
    return charModel_->char_consumption_rate_tag();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  const Tag&
  CharInterface<FieldT>::
  char_gasification_co2_rate_tag()
  {
    return charModel_->char_gasification_co2_rate_tag();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  const Tag&
  CharInterface<FieldT>::
  char_gasification_h2o_rate_tag()
  {
    return charModel_->char_gasification_h2o_rate_tag();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  const Tag&
  CharInterface<FieldT>::
  co2coratio_tag()
  {
    return charModel_->co2coratio_tag();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  const Tag&
  CharInterface<FieldT>::
  oxidation_tag()
  {
    return charModel_->oxidation_tag();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  Coal::CoalEqVec
  CharInterface<FieldT>::
  get_equations() const
  {
    return charModel_->get_equations();
  }

//==========================================================================
// Explicit template instantiation for supported versions of this expression
template class CharInterface< SpatialOps::Particle::ParticleField >;
//==========================================================================


} // namespace CHAR
