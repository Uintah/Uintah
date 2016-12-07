#ifndef CharInterface_h
#define CharInterface_h


/**
 *  \file CharInterface.h
 *  \defgroup CharOxidation Char Oxidation Model
 *
 */
#include <expression/ExprLib.h>

#include <CCA/Components/Wasatch/Coal/StringNames.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>

#include "LangmuirHinshelwood/LangmuirInterface.h"
#include "FirstOrderArrhenius/FirstOrderInterface.h"
#include "CCK/CCKInterface.h"

namespace CHAR{
	

  /**
   *  \ingroup CharOxidation
   *  \class CharInterface
   *
   *  \brief Provides an interface to the Char oxidation models
   *
   */
  template< typename FieldT >
  class CharInterface
  {
    CharBase* charModel_;
    LH::LangmuirInterface   <FieldT>* lhModel_;
    FOA::FirstOrderInterface<FieldT>* firstOrderModel_;
    CCK::CCKInterface       <FieldT>* cckModel_;

    WasatchCore::GraphCategories& gc_;

    /**
     *  \brief registers char mass clipping expression for the
     *         specified char oxidation model
     */
    void clip_char_mass();

    CharInterface(); // no copying
    CharInterface& operator=( const CharInterface& );  // no assignment

  public:

    /**
     *  \param pDiamTag       particle diameter
     *  \param pTempTag       particle temperature
     *  \param gTempTag       gas phase temperature
     *  \param co2MassFracTag gas-phase CO2 mass fraction at the particle surface
     *  \param coMassFracTag  gas-phase CO mass fraction at the particle surface
     *  \param o2MassFracTag  gas-phase O2 mass fraction at the particle surface
     *  \param h2MassFracTag  gas-phase H2 mass fraction at the particle surface
     *  \param h2oMassFracTag gas-phase H2O mass fraction at the particle surface
     *  \param ch4MassFracTag gas-phase CH4 mass fraction at the particle surface
     *  \param mixMWTag       gas-phase mixture molecular weight at the particle surface
     *  \param pDensTag       gas-phase mixture mass density at the particle surface
     *  \param gPressTag      gas phase pressure
     *  \param pMassTag       particle mass
     *  \param pMass0Tag      initial particle mass
     *  \param pDens0Tag      initial particle density
     *  \param pDiam0Tag      initial particle diameter
     *  \param volatilesTag   volatile mass within the coal
     *  \param coalType       the name of the coal
     *  \param devModel       devolatilization model
     *  \param chModel        char oxidation model
     */
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
                   const DEV::DevModel devModel,
                   const CharModel chModel );

    /**
     *  \brief obtain the Tag for the char mass (kg)
     */
    const Tag& char_mass_tag();

    /**
     *  \brief obtain the Tag for the char consumption rate.
     */
    const Tag& char_consumption_rate_tag();

    /**
     *  \brief obtain the Tag for the char consumption rate due to reaction of
     *          \f$ C_{(char)}+CO_{2}\rightarrow 2CO \f$
     */
    const Tag& char_gasification_co2_rate_tag();

    /**
     *  \brief obtain the Tag for the char consumption rate due to reaction of
     *          \f$ C_{(char)}+H_{2}O\rightarrowCO+H_{2} \f$
     */
    const Tag& char_gasification_h2o_rate_tag();

    /**
     *  \brief Obtain the ratio of CO2/CO due to char oxidation tag.
     */
    const Tag& co2coratio_tag();

    /**
     *  \brief Obtain the char consumption due to char oxidation
     */
    const Tag& oxidation_tag();

    /**
     *  \brief Obtain the list of all gas species production rate tags (kg/s)
     */
    const TagList& gas_species_src_tags();

    /**
     *  \brief obtain the Tag for the requested species
     */
    Tag gas_species_src_tag( const CharGasSpecies spec );

    /**
     *  \brief retrieve the vector of equation objects for the selected char model
     */
    Coal::CoalEqVec get_equations() const;

  };

} // namespace CHAR

#endif // CharInterface_h
