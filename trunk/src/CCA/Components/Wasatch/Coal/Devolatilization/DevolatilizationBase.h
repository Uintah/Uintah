#ifndef DEVOLATILIZATIONBase_h
#define DEVOLATILIZATIONBase_h

/**
 *  \ingroup Devolatilization 
 *  \brief   
 *           
 *
 *  \author  Babak Goshayeshi
 *  \date    Feb 2013 
 *
 *  
 *  \how 
 */

#include <expression/Tag.h>
#include <expression/ExpressionID.h>
#include <CCA/Components/Wasatch/Coal/CoalEquation.h>

namespace WasatchCore{ struct GraphHelper; }

namespace DEV{


  enum DEVSpecies {

        CO2  = 0,
        H2O  = 1,
        CO   = 2,
        HCN  = 3,
        NH3  = 4,
        C2H4 = 5,
        C2H6 = 6,
        C3H8 = 7,
        CH4  = 8,
        H2   = 9,
        INVALID_SPECIES = 99

  };


  enum DevModel{
    CPDM,
    KOBAYASHIM,
    SINGLERATE,
    DAE,
    INVALID_DEVMODEL
  };

  /**
   * @fn std::string dev_model_name( const DevModel model )
   * @param model the devolatilization model
   * @return the string name of the model
   */
  std::string dev_model_name( const DevModel model );

  /**
   * @fn DevModel devol_model( const std::string& modelName )
   * @param modelName the string name of the model
   * @return the corresponding enum value
   */
  DevModel devol_model( const std::string& modelName );

  /**
   * \class DevolatilizationBase
   * \brief Base class for devolatilization models
   */
  class DevolatilizationBase{
  public:

    Expr::TagList speciesSrcTags_;

    Expr::Tag charSrcTag_, tarSrcTag_,  volatilesTag_, volatilesSrcTag_;

    Expr::ExpressionID volatilesSrcID_;

    /**
     *  \brief obtain the TagList for the gas phase species production rates (kg/s)
     */
    const Expr::TagList& gas_species_src_tags() const{ return speciesSrcTags_; };
    
    
    /*
     *  \brief obtain the Tag for tar consuumption rate ( during devolatilization of coal )
     */
     const Expr::Tag& tar_production_rate_tag() const{ return tarSrcTag_; };

    /**
     *  \brief obtain the Tag for char production rate in CPD model ( during devolatilization of coal )
     */
    const Expr::Tag& char_production_rate_tag() const{ return charSrcTag_; };

    /**
     *  \brief obtain the Tag for volatile consumption rate rhs ( to add in rhs of particle mass change and volatile mass )
     */
    const Expr::Tag& volatile_consumption_rate_tag() const { return volatilesSrcTag_; };

    /**
     *  \brief obtain the Tag of volatile matter
     */
    const Expr::Tag& volatiles_tag() const { return volatilesTag_; };

    /**
     *  \brief retrieve the vector of equation objects for the selected devolatilization model
     */
    Coal::CoalEqVec get_equations() const{ return eqns_; }

    /**
     *  \brief obtain the Tag for production rate of specified species
     */
    virtual const Expr::Tag gas_species_src_tag( const DEV::DEVSpecies devspec ) const = 0;

    virtual ~DevolatilizationBase(){}

  protected:


    Coal::CoalEquation*           volatilesEqn_;
    Coal::CoalEqVec               eqns_;

    /**
     *  \brief Parses ODEs relevant to the selected devolatilization model.
     */
    virtual void parse_equations() = 0;

    /**
     *  \brief set tags that are not set in constructor initializer list
     */
    virtual void set_tags() = 0;

    /**
     *  \brief Registers expressions relevant to selected devolatilization model
     */
    virtual void register_expressions() = 0;

  };

} // namespace DEV

#endif // DEVOLATILIZATION_h
