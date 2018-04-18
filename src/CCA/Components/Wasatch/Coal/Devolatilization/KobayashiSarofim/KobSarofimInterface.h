#ifndef DEV_KOBSAROFIM_INTERFACE_h
#define DEV_KOBSAROFIM_INTERFACE_h

/**
 *  \class   KobSarofimInterface
 *  \ingroup Devolatilization - KobayashiSarofim
 *  \brief   Provide an interface to connect Kobbayashi-Sarofim two-step model
 *           as a devolatiization model to coal models. 
 *
 *  \author  Babak Goshayeshi
 *  \date    Jun 2012 
 *
 *  \param   coalType        : the CoalType for this coal.
 *  \param   pTempTag      : particle temperature.
 *  \param   prtMassTag       : particle mass
 *  \param   initialprtmastag : initial particle mass 
 */

#include <map>
#include <expression/ExprLib.h>
#include <expression/Tag.h>
#include <expression/ExpressionID.h>

#include <CCA/Components/Wasatch/Coal/StringNames.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>

#include <CCA/Components/Wasatch/Coal/Devolatilization/KobayashiSarofim/KobayashiSarofim.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/KobayashiSarofim/KobSarofimData.h>

namespace DEV { class DevolatilizationBase;}

namespace SAROFIM{
  

  /**
   *  \ingroup SAROFIM
   *  \class KobSarofimInterface
   *  \brief Provides an interface to the Kobayashi-Sarofim devolatilization model
   */
  template< typename FieldT >
  class KobSarofimInterface: public DEV::DevolatilizationBase
  {
    WasatchCore::GraphCategories& gc_;

    const Coal::CoalComposition coalComp_;
    const Expr::Tag pTempTag_, pMassTag_, pMass0Tag_;
    const KobSarofimInformation sarofimData_;
    const Coal::StringNames& sNames_;

    Expr::TagList dElementTags_, elementTags_, mvCharTags_;

    Coal::CoalEquation  *hydrogenEqn_, *oxygenEqn_;

    bool haveRegisteredExprs_;
    bool eqnsParsed_;

    KobSarofimInterface(); // no copying
    KobSarofimInterface& operator=( const KobSarofimInterface& ); // no assignment

    /**
     *  \brief Parse ODEs relevant to the Kobayashi-Sarofim model.
     */
    void parse_equations();

    /**
     *  \brief set tags that are not set in constructor initializer list
     */
    void set_tags();

    /**
     *  \brief Registers expressions relevant to Kobayashi-Sarofim model
     */
    void register_expressions();


  public:
    /**
     *  \param coalType the CoalType for this coal.
     *  \param pTempTag    Particle temperature
     *  \param prtMassTag     Particle mass
     *  \param initialprtmastag Initial particle mass
     */
    KobSarofimInterface( WasatchCore::GraphCategories& gc,
                         const Coal::CoalType& coalType,
                         const Expr::Tag pTempTag,
                         const Expr::Tag prtMassTag,
                         const Expr::Tag initialprtmastag );

    /**
     *  \brief obtain the Tag for production rate of specified species
     */
    const Expr::Tag gas_species_src_tag( const DEV::DEVSpecies devspec ) const;

  };

} // namespace SAROFIM

#endif // DEV_KOBSAROFIM_INTERFACE_h
