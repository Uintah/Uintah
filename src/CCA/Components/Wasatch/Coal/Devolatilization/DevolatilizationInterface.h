#ifndef DEVOLATILIZATION_h
#define DEVOLATILIZATION_h

/**
 *  \ingroup Devolatilization - KobayashiSarofim
 *  \brief   Provide an interface to be connected to an devolatilization model 
 *           
 *
 *  \author  Babak Goshayeshi
 *  \date    Jun 2012 
 *
 *  
 *  \how to specify the devolatiliztion model : 
 * 
 */

#include <map>
#include <expression/ExprLib.h>
#include <expression/Tag.h>
#include <expression/ExpressionID.h>

#include <CCA/Components/Wasatch/Coal/CoalData.h>
#include <CCA/Components/Wasatch/Coal/StringNames.h>

#include <CCA/Components/Wasatch/Coal/Devolatilization/CPD/CPDInterface.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/KobayashiSarofim/KobSarofimInterface.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/SingleRate/SingleRateInterface.h>

#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>


using WasatchCore::GraphCategories;

namespace DEV{



  /**
   *  \ingroup DEV
   *  \class DevolatilizationInterface
   *  \brief Provides an interface Devolatilization models
   */

  template< typename FieldT >
  class DevolatilizationInterface
  {

    DEV::DevolatilizationBase*  devModel_;
    CPD::CPDInterface<FieldT>*  cpdModel_;
    SAROFIM::KobSarofimInterface<FieldT>* kobModel_;
    SNGRATE::SingleRateInterface<FieldT>* singleRateModel_;


    const Coal::CoalType ct_;
    const Expr::Tag pTempTag_, pMassTag_, pMass0Tag_;

    DevolatilizationInterface(); // no copying
    DevolatilizationInterface& operator=( const DevolatilizationInterface& ); // no assignment


  public:
    /**
     *  \param coalType     the CoalType for this coal.
     *  \param pTempTag     Particle temperature
     *  \param pMassTag     Particle mass
     *  \param pMass0Tag    Initial particle mass
     */
    DevolatilizationInterface( GraphCategories& gc,
                               const Coal::CoalType coalType,
                               const DevModel  dvm,
                               const Expr::Tag pTempTag,
                               const Expr::Tag pMassTag,
                               const Expr::Tag pMass0Tag );


    /**
     *  \brief obtain the TagList for the gas phase species production rates (kg/s)
     */
    const Expr::TagList& gas_species_src_tags();

    /**
     *  \brief obtain the Tag for production rate of specified species
     */
    const Expr::Tag gas_species_src_tag(const DEVSpecies spec );

    /**
     *  \brief obtain the Tag for production rate of tar
     */
    const Expr::Tag& tar_production_rate_tag();


    const Expr::Tag& char_production_rate_tag();

    /**
     *  \brief obtain the Tag for volatile consumption rate rhs ( to add in rhs of particle mass change and volatile mass )
     */
    const Expr::Tag& volatile_consumption_rate_tag();

    /**
     *  \brief obtain the Tag of volatile matter
     */
    const Expr::Tag& volatiles_tag();

    /**
     *  \brief retrieve the vector of equation objects for the selected devolatilization model
     */
    Coal::CoalEqVec get_equations() const;

  };

} // namespace DEV

#endif // DEVOLATILIZATION_h
