#ifndef LangmuirInterface_h
#define LangmuirInterface_h

#include <expression/ExprLib.h>
#include <expression/Tag.h>
#include <expression/ExpressionID.h>

#include <CCA/Components/Wasatch/Coal/StringNames.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharBase.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharData.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>

namespace LH
{
  /**
   *  \ingroup CharOxidation
   *  \class   LangmuirInterface
   *  \author  Josh McConnell
   *  \date    December 2015
   *
   *  \brief Provides an interface for the Langmuir-Hinshelwood char oxidation model
   *
   */
  template< typename FieldT >
  class LangmuirInterface: public CHAR::CharBase
  {
    //particle and gas values
    const Tag pTempTag_,  gTempTag_, mixMWTag_, pDensTag_,
              gPressTag_, pDiamTag_, pMassTag_, pMass0Tag_;

    //mass fractions and species RHSs
    const Tag o2MassFracTag_, h2oMassFracTag_, co2MassFracTag_,
              o2_rhsTag_,     h2o_rhsTag_,      h2_rhsTag_,
              co2_rhsTag_,    co_rhsTag_;

    const bool initDevChar_;  ///< Initial char in volatile matter (Only with CPD Model)

    const CHAR::CharModel         charModel_;
    const CHAR::CharOxidationData charData_;

    WasatchCore::GraphCategories& gc_;

    const Coal::StringNames& sNames_;

    TagList co2CoTags_, char_co2coTags_, h2andh2o_rhsTags_;

    ExpressionID oxidationRHSID_, co2coRHSID_, o2RHSID_, gasifco2ID_, gasifh2oID_,
                 charRHSID_, h2Andh2oRHSID_;


    LangmuirInterface(); // no copying
    LangmuirInterface& operator=( const LangmuirInterface& );  // no assignment

    /**
     *  \brief parses ODEs solved for this char model
     */
    void parse_equations();

   /**
    *  \brief set tags that are not set in constructor initializer list
    */
   void set_tags();

    /**
     *  \brief registers all expressions relevant to evaluation of the
     *         specified char oxidation model
     */
    void register_expressions();

  public:

    /**
     *  \param pDiamTag       particle diameter
     *  \param pTempTag       particle temperature
     *  \param gTempTag       gas phase temperature
     *  \param co2MassFracTag gas-phase CO2 mass fraction at the particle surface
     *  \param o2MassFracTag  gas-phase O2 mass fraction at the particle surface
     *  \param h2oMassFracTag gas-phase H2O mass fraction at the particle surface
     *  \param mixMWTag       gas-phase mixture molecular weight at the particle surface
     *  \param pDensTag       particle density
     *  \param gPressTag      gas phase pressure
     *  \param pMassTag       particle mass
     *  \param pMass0Tag      initial particle mass
     *  \param coalType       the name of the coal
     *  \param devModel       devolatilization model
     *  \param chModel        char oxidation model
     */
    LangmuirInterface( WasatchCore::GraphCategories& gc,
                       const Tag& pDiamTag,
                       const Tag& pTempTag,
                       const Tag& gTempTag,
                       const Tag& co2MassFractag,
                       const Tag& o2MassFracTag,
                       const Tag& h2oMassFractag,
                       const Tag& mixMWTag,
                       const Tag& pDensTag,
                       const Tag& gPressTag,
                       const Tag& pMassTag,
                       const Tag& initPrtMassTag,
                       const Coal::CoalType coalType,
                       const DEV::DevModel dvmodel,
                       const CHAR::CharModel chmodel );

    /**
     *  \brief obtain the Tag for the requested species
     */
    const Tag gas_species_src_tag( const CHAR::CharGasSpecies spec ) const;

   /**
    * \brief Return the ID of Co2 and Co consumption in gas pahse
    */
    Expr::ExpressionID co2co_rhs_ID() const{ return co2coRHSID_; }

    /**
     * \brief Return the ID for Oxygen consumption is gas pahse
     */
    Expr::ExpressionID o2_rhs_ID() const{ return o2RHSID_; }

    /**
     * \brief Return the ID of H2 and H2O consumption of gas
     */
    Expr::ExpressionID h2andh2orhsID() const{ return h2Andh2oRHSID_; }

  };

} // namespace LH

#endif // LangmuirInterface_h
