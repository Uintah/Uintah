#ifndef CCKInterface_h
#define CCKInterface_h

#include <expression/ExprLib.h>

#include <CCA/Components/Wasatch/Coal/StringNames.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharBase.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharData.h>

#include "CCKData.h"

namespace CCK{


  /**
   *  \ingroup CharOxidation
   *  \class   LangmuirInterface
   *  \author  Josh McConnell
   *  \date    December 2015
   *
   *  \brief Provides an interface for the CCK model
   *
   */
  template< typename FieldT >
  class CCKInterface: public CHAR::CharBase
  {
    WasatchCore::GraphCategories& gc_;

    //particle and gas property tags;
    const Tag gTempTag_, mixMWTag_, gPressTag_;

    //mass fractions and species RHSs
    const Tag o2MassFracTag_, h2oMassFracTag_, h2MassFracTag_, ch4MassFracTag_, co2MassFracTag_, coMassFracTag_,
              co2_rhsTag_,    co_rhsTag_,      o2_rhsTag_,      h2o_rhsTag_,    h2_rhsTag_,      ch4_rhsTag_;

    // particle property tags
    const Tag pDiam0Tag_,       ashPorosityTag_,    ashThicknessTag_,  coreDiamTag_,     coreDensityTag_,
              thermAnnealTag_,  charConversionTag_, ashDensityTag_,    ashMassFracTag_,  charMassFracTag_,
              devolDensityTag_, pDiamModTag_,       devolDens0Tag_,    coreDens0Tag_,    pTempTag_,
              pDens0Tag_,       pDiamTag_,          pMassTag_,         pMass0Tag_,       volatilesTag_,
              devolAshMassFracTag_, devolCharMassFracTag_;

    const Coal::StringNames& sNames_;

    const bool initDevChar_;  ///< Initial char in volatile matter (Only with CPD Model)
    const CCKData cckData_;

    TagList massFracTags_, particleTags_, gasTags_, charDepletionTags_,
            char_rhsTags_,  logFreqDistTags_, logFreqDistRHSTags_;

    CCKInterface(); // no copying
    CCKInterface& operator=( const CCKInterface& );  // no assignment

    Coal::CoalEqVec logFrequencyEqns_;

    /**
     *  \brief parses ODEs solved for this char model
     */
    void parse_equations();

   /**
    *  \brief set tags that are not set in constructor initializer list
    */
   void set_tags();

    /**
     *  \brief registers all expressions relevant to evaluation this char model
     */
   void register_expressions();

  public:

    /**
     *  \param pDiamTag     particle diameter
     *  \param pTempTag       particle temperature
     *  \param gTempTag       gas phase temperature
     *  \param co2MassFracTag gas-phase CO2 mass fraction at the particle surface
     *  \param coMassFracTag  gas-phase CO mass fraction at the particle surface
     *  \param o2MassFracTag  gas-phase O2 mass fraction at the particle surface
     *  \param h2MassFracTag  gas-phase H2 mass fraction at the particle surface
     *  \param h2oMassFracTag gas-phase H2O mass fraction at the particle surface
     *  \param ch4MassFracTag gas-phase CH4 mass fraction at the particle surface
     *  \param mixMWTag       gas-phase mixture molecular weight at the particle surface
     *  \param pDensTag     gas-phase mixture mass density at the particle surface
     *  \param gasPressureTag gas phase pressure
     *  \param pMassTag       particle mass
     *  \param pMass0Tag     initial particle mass
     *  \param pDens0Tag       initial particle density
     *  \param pDiam0Tag      initial particle diameter
     *  \param volatilesTag   volatile mass within the coal
     *  \param coalType       the name of the coal
     *  \param devModel       devolatilization model
     *  \param chModel        char oxidation model
     */
    CCKInterface( WasatchCore::GraphCategories& gc,
                  const Tag& pDiamTag,
                  const Tag& pTempTag,
                  const Tag& gTempTag,
                  const Tag& co2MassFractag,
                  const Tag& coMassFractag,
                  const Tag& o2MassFracTag,
                  const Tag& h2MassFractag,
                  const Tag& h2oMassFractag,
                  const Tag& ch4MassFractag,
                  const Tag& mixMWTag,
                  const Tag& gasPressureTag,
                  const Tag& pMassTag,
                  const Tag& pMass0Tag,
                  const Tag& pDens0Tag,
                  const Tag& pDiam0Tag,
                  const Tag& volatilesTag,
                  const Coal::CoalType coalType,
                  const DEV::DevModel devModel );

    /**
     *  \brief obtain the Tag for the requested species
     */
    const Tag gas_species_src_tag( const CHAR::CharGasSpecies spec ) const;

  };

} // namespace CCK

#endif // CCKInterface_h
