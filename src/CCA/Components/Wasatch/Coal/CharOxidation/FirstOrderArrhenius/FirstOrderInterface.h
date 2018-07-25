#ifndef FirstOrderInterface_h
#define FirstOrderInterface_h

#include <expression/ExprLib.h>

#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/Coal/StringNames.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharBase.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharData.h>

#include "FirstOrderData.h"

namespace WasatchCore{ class TimeStepper; }

namespace FOA{

/**
 *  \ingroup CharOxidation
 *  \class   FirstOrderInterface
 *  \author  Josh McConnell
 *  \date    December 2015
 *
 *  \brief Provides an interface for the 1st-order Arrhenius gasification
 *         model using a Langmuir-Hinshelwood expression for char oxidation.
 *
 */
  template< typename FieldT >
  class FirstOrderInterface: public CHAR::CharBase
  {
    WasatchCore::GraphCategories& gc_;

    //particle and gas values
    const Tag pTempTag_,  gTempTag_,     mixMWTag_,    pDensTag_,
              gPressTag_, pDiamTag_,   pMassTag_,  pMass0Tag_;

    //mass fractions and species RHSs
    const Tag o2MassFracTag_,  h2oMassFracTag_, h2MassFracTag_, co2MassFracTag_, coMassFracTag_,
              ch4MassFracTag_, o2_rhsTag_,      h2_rhsTag_,     h2o_rhsTag_,     ch4_rhsTag_,
              co2_rhsTag_,     co_rhsTag_;

    const bool initDevChar_;  ///< Initial char in volatile matter (Only with CPD Model)
    const CHAR::CharModel         charModel_;
    const CHAR::CharOxidationData charData_;
    const FirstOrderData    firstOrderData_;

    const Coal::StringNames& sNames_;

    TagList co2CoTags_, char_co2coTags_, h2andh2o_rhsTags_, massFracTags_;

    ExpressionID oxidationRHSID_, co2coRHSID_, o2RHSID_,
                 charRHSID_, h2Andh2oRHSID_, gasifID_;

    FirstOrderInterface(); // no copying
    FirstOrderInterface& operator=( const FirstOrderInterface& );  // no assignment

    /**
     *  \brief parses ODEs solved for this char model
     */
    void parse_equations();

   /**
    *  \brief set tags that are not set in constructor initializer list
    */
   void set_tags();

    /**
     *  \brief registers all expressions relevant to evaluation of this
     *         char model
     */
    void register_expressions();

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
     *  \param totalMWTag     gas-phase mixture molecular weight at the particle surface
     *  \param pDensTag       gas-phase mixture mass density at the particle surface
     *  \param gPressTag      gas phase pressure
     *  \param pMassTag       particle mass
     *  \param pMass0Tag      initial particle mass
     *  \param coalType       name of the coal type
     *  \param devModel       devolatilization model
     */
    FirstOrderInterface( WasatchCore::GraphCategories& gc,
                         const Tag& pDiamTag,
                         const Tag& pTempTag,
                         const Tag& gTempTag,
                         const Tag& co2MassFractag,
                         const Tag& coMassFractag,
                         const Tag& o2MassFracTag,
                         const Tag& h2MassFractag,
                         const Tag& h2oMassFractag,
                         const Tag& ch4MassFractag,
                         const Tag& totalMWTag,
                         const Tag& pDensTag,
                         const Tag& gPressTag,
                         const Tag& pMassTag,
                         const Tag& pMass0Tag,
                         const Coal::CoalType coalType,
                         const DEV::DevModel devModel );

    /**
     *  \brief registers all expressions relevant to evaluation of the
     *         specified char oxidation model
     */
    void register_expressions( Expr::ExpressionFactory& );


    /**
     *  \brief add char oxidation equations onto the time integrator
     */
    void hook_up_time_integrator( WasatchCore::TimeStepper& ts, Expr::ExpressionFactory& factory );

    /**
     *  \brief set the initial conditions for char
     */
    void set_initial_conditions( Expr::FieldManagerList& );

    /**
     *  \brief obtain the Tag for the requested species
     */
    const Tag gas_species_src_tag( const CHAR::CharGasSpecies spec ) const;

   /**
    * \brief Returns the ID of Co2 and Co consumption in gas phase
    */
    Expr::ExpressionID co2co_rhs_ID() const{ return co2coRHSID_; }

    /**
     * \brief Return the ID for Oxygen consumption is gas phase
     */
    Expr::ExpressionID o2_rhs_ID() const{ return o2RHSID_; }

    /**
     * \brief Returns the ID of H2 and H2O consumption of gas
     */
    Expr::ExpressionID h2andh2orhsID() const{ return h2Andh2oRHSID_; }

  };

} // namespace FOA

#endif // FirstOrderInterface_h
