#ifndef CoalInterface_h
#define CoalInterface_h

#include <vector>
#include <string>
#include <expression/ExprLib.h>
#include <expression/Tag.h>

#include <CCA/Components/Wasatch/Coal/StringNames.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationInterface.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharInterface.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>

namespace EVAP{ template<typename T> class EvapInterface; }

/**
 *  \file CoalInterface.h
 *  \defgroup coal Coal Models
 */

namespace Coal{

using WasatchCore::GraphCategories;

  
  /**
   *  \class CoalInterface
   *  \ingroup coal
   *  \brief Provides a concise interface to the coal models
   *
   */
  template< typename FieldT >
  class CoalInterface
  {
    GraphCategories& gc_;
    const Expr::Tag pTempTag_,  gTempTag_, mixMWTag_, pDensTag_, gPressTag_, pMassTag_,
                    pDiamTag_,rePTag_,   scGTag_,   pCpTag_,   pTemp_rhsTag_,
                    heatReleasedToGasTag_;

    const SpeciesTagMap specTagMap_;

    const DEV::DevModel devModel_;

    DEV::DevolatilizationInterface<FieldT>* dev_;
    CHAR::CharInterface           <FieldT>* char_;
    EVAP::EvapInterface           <FieldT>* evap_;

    const Expr::Tag mvTag_, charTag_, moistureTag_;

    Expr::TagList gasSpeciesSourceTags_, productionRateTags_;


    /**
     *  \brief Obtain the Expr::Tag for the given species.
     */
    const Expr::Tag& get_species_tag( const GasSpeciesName ) const;

    /**
     *  \brief register necessary expressions
     */
    void register_expressions( );

    CoalInterface(); // no copying
    CoalInterface& operator=( const CoalInterface& ); // no assignment

  public:

    /**
     *  \param coalType          Describes the type of coal that is being considered
     *  \param pDiamTag          Particle Diameter
     *  \param pTempTag          Particle temperature
     *  \param gTempTag          Gas temperature
     *  \param mixMWTag          Gas-phase mixture molecular weight
     *  \param pDensTag          the tag for the particle density
     *  \param specMap           Connects each GasSpeciesName to the
     *                           corresponding Expr::Tag for that species. These
     *                           are required for coupling the coal models to
     *                           the gas phase.  Only a few of the species
     *                           actually get coupled, since the coal models
     *                           only need a few of the gas phase species.
     *  \param gPressTag    Interpolated gas pressure to the particle field
     *  \param pMassTag          Particle mass tag
     *  \param rePTag            Particle Reynolds number.
     *  \param scGTag            Schimdt Number of the gas !
     *  \param specTagMap        Map of all the Species that can peresent in gas pahse
     *  \param pMass0Tag         initial mass of each particle
     *  \param pDens0Tag         initial density of each particle
     *  \param pDiam0Tag         initial diameter of each particle
     */
     CoalInterface( GraphCategories& gc,
                    const CoalType coalType,
                    const DEV::DevModel devModel,
                    const CHAR::CharModel chModel,
                    const Expr::Tag& pDiamTag,
                    const Expr::Tag& pTempTag,
                    const Expr::Tag& gTempTag,
                    const Expr::Tag& mixMWTag,
                    const Expr::Tag& pDensTag,
                    const Expr::Tag& gPressTag,
                    const Expr::Tag& pMassTag,
                    const Expr::Tag& rePTag,
                    const Expr::Tag& scGTag,
                    const SpeciesTagMap& specTagMap,
                    const Expr::Tag& pMass0Tag,
                    const Expr::Tag& pDens0Tag,
                    const Expr::Tag& pDiam0Tag );

    ~CoalInterface();

    /**
     *  \brief retrieve the vector of equations for the selected coal models
     */
    Coal::CoalEqVec get_equations() const;

    /**
     *  \brief Set the initial conditions for the coal models
     */
    void set_initial_conditions( Expr::FieldManagerList& fml );

    /**
     *  \brief obtain the TagList for the gas phase species consumption rates (kg/s)
     */
    const Expr::TagList gas_species_source_terms() const{ return gasSpeciesSourceTags_; }

    /**
     *  \brief Obtain the TagList for the source terms for the requested gas phase species (kg/s)
     *
     *  \param spec the species name
     *  \param forceMatch if true and no match is found, an exception will be thrown.
     */

    
    Expr::TagList gas_species_source_term( const GasSpeciesName spec,
                                           const bool forceMatch = true ) const;
    
    /**
         *  \brief Obtain the Tag for the source terms for the requested gas phase tar (kg/s)
         *
         */
    
    Expr::Tag tar_source_term() const;

    /**
     *  \brief Obtain the Tag for particle production rate (kg/s)
     */
    const Expr::TagList particle_mass_rhs_taglist() const{ return productionRateTags_; }
    

    /**
     *  \brief Obtain the Tag for particle temperature rhs due to Char Oxidation and Evaporation
     */
    const Expr::Tag particle_temperature_rhs_tag() const {return pTemp_rhsTag_;}
    const Expr::Tag heat_released_to_gas_tag () const {return   heatReleasedToGasTag_;}
    const Expr::Tag coal_heat_capacity_tag() const {return pCpTag_;}
    const Expr::Tag mass_volatile_tag() const {return mvTag_;}
    const Expr::Tag mass_char_tag() const {return charTag_;}
    const Expr::Tag mass_moisture() const {return moistureTag_;}
  };

} // namespace coal

#endif // CoalInterface_h
