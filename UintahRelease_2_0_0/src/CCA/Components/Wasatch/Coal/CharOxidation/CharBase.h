/*
 * CharBase.h
 *
 *  Created on: Dec 12, 2015
 *      Author: josh
 */

#ifndef CharBase_h
#define CharBase_h

/**
 *  \author  Josh McConnell
 *  \date    December 2015
 */

#include <expression/Tag.h>
#include <expression/ExpressionID.h>
#include <CCA/Components/Wasatch/Coal/CoalEquation.h>

using Expr::STATE_N;
using Expr::STATE_NONE;
using Expr::Tag;
using Expr::TagList;
using Expr::ExpressionID;

namespace CHAR{

enum CharGasSpecies{
  O2,
  CO2,
  CO,
  H2,
  H2O,
  CH4,
  INVALID_SPECIES = 99
};


  enum CharModel{
    LH,
    FRACTAL,
    FIRST_ORDER,
    CCK,
    INVALID_CHARMODEL
  };

  /**
   * @fn std::string char_model_name( const CharModel model )
   * @param model: char chemistry model
   * @return the string name of the model
   */
  std::string char_model_name( const CharModel model );

  /**
   * @fn CharModel char_model( const std::string& modelName )
   * @param modelName: the string name of the char model
   * @return the corresponding enum value
   */
  CharModel char_model( const std::string& modelName );

  /**
   * \class CharBase
   * \brief Base class for char chemistry models
   */
  class CharBase{

  public:


    /**
     *  \brief obtain the TagList for the gas phase species production rates (kg/s)
     */
    const TagList& gas_species_src_tags() const{ return speciesSrcTags_; };

    /**
     *  \brief obtain the Tag for char mass
     */
    const Tag& char_mass_tag() const{ return charMassTag_;};

    /**
     *  \brief obtain the Tag for char consumption rate
     */
    const Tag& char_consumption_rate_tag() const{ return charMass_rhsTag_;};


    /**
     *  \brief obtain the Tag for char consumption by oxygen
     */
    const Tag& oxidation_tag() const{ return oxidation_rhsTag_;};

    /**
     *  \brief obtain the Tag for char consumption by CO2 gasification
     */
    const Tag& char_gasification_co2_rate_tag() const{ return heteroCo2Tag_;};

    /**
     *  \brief obtain the Tag for char consumption by H2O gasification
     */
    const Tag& char_gasification_h2o_rate_tag() const{ return heteroH2oTag_;};

    /**
     *  \brief obtain the Tag for CO2-CO ratio
     */
    const Tag& co2coratio_tag() const{ return co2CoRatioTag_;};

    /**
     *  \brief obtain the Tag for production rate of specified species
     */
    virtual const Tag gas_species_src_tag( const CHAR::CharGasSpecies charSpec ) const = 0;

    /**
     *  \brief retrieve the vector of equation objects for the selected char model
     */
    Coal::CoalEqVec get_equations() const{ return eqns_; }

    virtual ~CharBase(){}

  protected:


    TagList speciesSrcTags_;

    Tag charMassTag_,  charMass_rhsTag_;
    Tag heteroCo2Tag_, heteroH2oTag_, oxidation_rhsTag_, co2CoRatioTag_;

    /**
     *  \brief Parse ODEs relevant to the selected char model.
     */
    virtual void parse_equations() = 0;

   /**
    *  \brief set tags that are not set in constructor initializer list
    */
    virtual void set_tags() = 0;

    /**
     *  \brief Register all expressions required to implement the CPD model
     */
    virtual void register_expressions() = 0;

    Coal::CoalEquation*           charEqn_;
    Coal::CoalEqVec               eqns_;
  };

} // namespace CHAR




#endif /* CharBase_h */
