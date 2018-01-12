#ifndef EvapInterface_h
#define EvapInterface_h


/**
 *  \file EvapInterface.h
 *  \
 *
 */
#include <expression/ExprLib.h>
#include <expression/Tag.h>
#include <expression/ExpressionID.h>

#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>
#include <CCA/Components/Wasatch/Coal/CoalEquation.h>

namespace WasatchCore{class GraphHelper;       }

namespace EVAP{

  enum EvapSpecies{
    H2O = 0,
    INVALID_SPECIES = 99
  };


  /**
   *  \ingroup Evaporation
   *  \class EvapInterface
   *
   */
  template< typename FieldT >
  class EvapInterface
  {
    WasatchCore::GraphCategories& gc_;

    const Coal::CoalComposition coalComp_;
    const Coal::CoalType ct_;
    const Expr::Tag tempGTag_, tempPTag_, diamPTag_, rePTag_, scGTag_, waterMasFracTag_,
                    totalMWTag_, gasPressureTag_, pMassTag_;

    Coal::CoalEquation* evapEqn_;
    Coal::CoalEqVec     eqns_;
    bool                eqnsParsed_;

    EvapInterface(); // no copying
    EvapInterface& operator=( const EvapInterface& );  // no assignment

    void register_expressions();
    void parse_equations();


  public:

    /**
     *  \param tempGTag gas phase temperature at the particle location (K)
     *  \param tempPTag particle temperature (K)
     *  \param diamPTag particle diameter (m)
     *  \param rePTag   particle Reynolds number
     *  \param scGTag   Schmidt number of Gas
     *  \param dgTag    Diffusivity of H2O in the gas phase -air, (m2/s)
     *  \param waterMasFracTag : Mass Fraction of Water in the gas phase
     *  \param totalMWTag : Total molecular weight of gas phase - Equation requires partial pressure of water which could be calculated by mass fraction and total molecular weight
     *  \param gasPressureTag : Total Pressure of Gas Phase
     */
     EvapInterface(WasatchCore::GraphCategories& gc,
                   const Expr::Tag tempGTag,
                   const Expr::Tag tempPTag,
                   const Expr::Tag diamPTag,
                   const Expr::Tag rePTag,
                   const Expr::Tag scGTag,
                   const Expr::Tag waterMasFracTag,
                   const Expr::Tag totalMWTag,
                   const Expr::Tag gasPressureTag,
                   const Expr::Tag prtmasstag,
                   const Coal::CoalType ct );

    /**
     *  \brief retrieve the moisture tag.
     */
    const Expr::Tag retrieve_moisture_mass_tag() const{ return evapEqn_->solution_variable_tag(); }
    /**
     *  \brief obtain the moisture production rate of particle
     */
    const Expr::Tag moisture_rhs_tag() const{ return evapEqn_->rhs_tag();}
    /**
     *  \brief obtain consumption rate of gas
     */
    Expr::Tag gas_species_src_tag( const EvapSpecies spec ) const;

    Coal::CoalEqVec get_equations() const{ return eqns_; }



  };

} // namespace EVAP

#endif // EvapInterface_h
