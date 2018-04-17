#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationInterface.h>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/Nebo.h>

// expressions we build here

using std::vector;
using std::endl;
using std::ostringstream;
using std::string;

namespace DEV{


  template< typename FieldT >
  DevolatilizationInterface<FieldT>::
  DevolatilizationInterface( GraphCategories& gc,
                             const Coal::CoalType ct,
                             const DevModel  dvm,
                             const Expr::Tag pTempTag,
                             const Expr::Tag pMassTag,
                             const Expr::Tag pMass0Tag )
  : ct_( ct ),
    pTempTag_      ( pTempTag ),
    pMassTag_ ( pMassTag ),
    pMass0Tag_( pMass0Tag )
  {
    switch (dvm) {
      case CPDM:
        cpdModel_ = new CPD::CPDInterface<FieldT>(gc, ct, pTempTag, pMassTag, pMass0Tag);
        devModel_ = cpdModel_;
        break;
      case KOBAYASHIM:
        kobModel_ = new SAROFIM::KobSarofimInterface<FieldT>(gc, ct, pTempTag, pMassTag, pMass0Tag);
        devModel_ = kobModel_;
        break;
      case SINGLERATE:
        singleRateModel_ = new SNGRATE::SingleRateInterface<FieldT>(gc, ct, pTempTag, pMassTag, pMass0Tag, false);
        devModel_ = singleRateModel_;
        break;
      case DAE:
        singleRateModel_ = new SNGRATE::SingleRateInterface<FieldT>(gc, ct, pTempTag, pMassTag, pMass0Tag, true);
        devModel_ = singleRateModel_;
        break;

      case INVALID_DEVMODEL:
        throw std::invalid_argument( "Invalid model in DevolatilizationInterface constructor" );
    }
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  const Expr::Tag
  DevolatilizationInterface<FieldT>::
  gas_species_src_tag(const DEVSpecies spec)
  {  
    return devModel_->gas_species_src_tag(spec);
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  const Expr::TagList&
  DevolatilizationInterface<FieldT>::
  gas_species_src_tags()
  {    
    return devModel_->gas_species_src_tags();
  }
  
  //--------------------------------------------------------------------


  template< typename FieldT > // jtm
  const Expr::Tag&
  DevolatilizationInterface<FieldT>::
  tar_production_rate_tag()
  {
    return devModel_->tar_production_rate_tag();
  }
  

  //--------------------------------------------------------------------


  template< typename FieldT >
  const Expr::Tag&
  DevolatilizationInterface<FieldT>::
  char_production_rate_tag()
  {
    return devModel_->char_production_rate_tag();
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  const Expr::Tag&
  DevolatilizationInterface<FieldT>::
  volatile_consumption_rate_tag()
  {
    return devModel_->volatile_consumption_rate_tag();
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  const 
  Expr::Tag&
  DevolatilizationInterface<FieldT>::
  volatiles_tag()
  {
    return devModel_->volatiles_tag();
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  Coal::CoalEqVec
  DevolatilizationInterface<FieldT>::
  get_equations() const
  {
    return devModel_->get_equations();
  }

  //====================================================================
  // Explicit template instantiation
  template class DevolatilizationInterface< SpatialOps::Particle::ParticleField >;
  //====================================================================

} // namespace DEV
