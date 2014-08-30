/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef EQUIL_RXN_MDL
#define EQUIL_RXN_MDL

#include <map>
#include <vector>

#include <tabprops/prepro/rxnmdl/ReactionModel.h>

class MixtureFraction;  // forward declaration; Mixture fraction tools.


struct EqStateVars{
  EqStateVars()
  {
    static const int NSMAX = 300;
    ys.resize(NSMAX,0.0);
  }
  EqStateVars( const double t, const double p, const std::vector<double> & y )
  {
    temp=t;
    press=p;
    ys = y;
  }
  std::vector<double> ys;
  double temp;
  double press;
};

//====================================================================
//====================================================================

//--------------------------------------------------------------------
/**
 *  @class AdiabaticEquilRxnMdl
 *  @brief Implements the adiabatic equilibrium reaction model.
 *
 *  @author  James C. Sutherland
 *  @date    February, 2005
 *
 *  Implement the Equilibrium reaction model without heat loss.
 *
 *  Possible revisions:
 *    \li Implement pressure variation.
 *    \li Non-uniform spacing in f-space
 */
class AdiabaticEquilRxnMdl : public ReactionModel
{
 public:

  /**
   *  @brief Construct an AdiabaticEquilRxnMdl object.
   *
   *  @param gas    : the Cantera gas object, of type Cantera_CXX::IdealGasMix.
   *  @param y_oxid : vector<double> oxidizer mass fractions.
   *  @param y_fuel : vector<double>   fuel   mass fractions.
   *  @param haveMassFrac : true if mass fractions are provided, false for mole fractions
   *  @param interpOrder : the order of interpolant (3 is default)
   *  @param nFpts  : specify the number of points in the mixture fraction
   *                  dimension.  A default value is provided.
   *  @param scaleFac : grid strech factor.  Grid will be clustered near stoichiometric.
   *                    Typical values range from 1 (no stretching) to 5 (lots of
   *                    stretching)
   */
  AdiabaticEquilRxnMdl( Cantera_CXX::IdealGasMix & gas,
                        const std::vector<double> & y_oxid,
                        const std::vector<double> & y_fuel,
                        const bool haveMassFrac,
                        const int interpOrder,
                        const int nFpts = 201,
                        const double scaleFac = 3.0 );

  /**
   *  @brief Construct an empty AdiabaticEquilRxnMdl object.
   *
   *  In general, this should not be used unless one is loading
   *  the reaction model from disk!
   */
  AdiabaticEquilRxnMdl();

  ~AdiabaticEquilRxnMdl();

  /**
   *  @brief Set the temperature of the fuel.
   *  @param T : Fuel stream temperature (Kelvin).
   */
  void set_fuel_temp( const double T ){ Tfuel_ = T; };

  /**
   *  @brief Set the temperature of the oxidizer.
   *  @param T : Oxidizer stream temperature (Kelvin).
   */
  void set_oxid_temp( const double T ){ Toxid_ = T; };

  /**
   *  @brief Set the pressure (assumed constant).  If not specified, atmospheric pressure
   *  is assumed (101325 Pa).
   *
   *  @param p : System pressure (Pa)
   */
  void set_pressure( const double p ){ pressure_ = p; };

  /** @brief Apply the equilibrium model and build the table */
  void implement();

  const std::vector<double>& get_f_mesh() const{ return fpts_; };

  const EqStateVars& get_state_vars( const int ifpt ) const{
    return stateVars_[ifpt];
  }

  MixtureFraction * const get_mixture_fraction(){ return mixfr_; }

  void no_output(){outputTable_=false;}

  //------------------------------------------------------------------
 private:

  AdiabaticEquilRxnMdl( const AdiabaticEquilRxnMdl& );  // no copying

  int nFpts_;

  std::vector<EqStateVars> stateVars_;
  std::vector<double> fpts_;

  double scaleFac_;

  const bool pressVary_;

  double Tfuel_, Toxid_, pressure_;

  MixtureFraction * const mixfr_;
  bool outputTable_;

  void set_f_mesh();
  const std::vector<std::string> & indep_var_names();
};

//====================================================================
//====================================================================

//--------------------------------------------------------------------
/**
 *  @class EquilRxnMdl
 *  @brief Implement the equilibrium reaction model with heat loss.
 *
 *  @author  James C. Sutherland
 *  @date    April, 2005
 *
 *  Implement the Equilibrium reaction model with heat loss.
 *  The amount of heat loss available is a function of the sensible enthalpy
 *  difference between the reactants and products at adiabatic conditions.
 *
 *  Possible revisions:
 *    \li Implement pressure variation.
 *    \li Non-uniform spacing in f-space
 */
class EquilRxnMdl : public ReactionModel
{

 public:

  /**
   * @param gas the Cantera object
   * @param y_oxid  oxidizer mass fractions
   * @param y_fuel  fuel     mass fractions
   * @param haveMassFrac if false, then the y_oxid and y_fuel inputs are taken as mole fractions
   * @param order   order for the interpolants
   * @param nFpts   number of points in mixture fraction space
   * @param nHLpts  number of points in heat loss space
   * @param scaleFac mesh compression factor in mixture fraction space
   */
  EquilRxnMdl( Cantera_CXX::IdealGasMix & gas,
               const std::vector<double> & y_oxid,
               const std::vector<double> & y_fuel,
               const bool haveMassFrac,
               const int order,
               const int nFpts = 201,
               const int nHLpts = 21,
               const double scaleFac = 4.0 );

  EquilRxnMdl();

  ~EquilRxnMdl();

  /**
   *  @brief Set the temperature of the fuel.
   *  @param T : Fuel stream temperature (Kelvin).
   */
  void set_fuel_temp( const double T ){ Tfuel_ = T; };

  /**
   *  @brief Set the temperature of the oxidizer.
   *  @param T : Oxidizer stream temperature (Kelvin).
   */
  void set_oxid_temp( const double T ){ Toxid_ = T; };

  /**
   *  @brief Set the pressure (assumed constant)
   *  @param p : System pressure (Pa)
   */
  void set_pressure( const double p ){ pressure_ = p; };

  /** @brief Apply the equilibrium model and build the table */
  void implement();

  MixtureFraction * const get_mixture_fraction(){ return mixfr_; }

 private:

  int nFpts_, nHLpts_;

  double scaleFac_;

  double Tfuel_, Toxid_, pressure_;
  std::vector<double> fpts_;

  MixtureFraction * const mixfr_;
  AdiabaticEquilRxnMdl* adiabaticEq_;

  std::map< double, std::map<double,EqStateVars> > fullTable_;

  void set_species_enthalpies( const double* ys,
                               const double  temp,
                               double* h );

  void set_f_mesh();

  bool do_equil( std::vector<double> & ys,
                 const double targetEnth,
                 const double Tguess );

  const std::vector<std::string> & indep_var_names();

  void output_table();


  class SensEnthEvaluator : public StateVarEvaluator
  {
  private:
    Cantera::ThermoPhase& thermo_;
    const AdiabaticEquilRxnMdl& adEq_;
    MixtureFraction& mixfrac_;
    const double hfuel_, hoxid_;
    std::vector<double> spwork_, ho_, hs_;
  public:
    SensEnthEvaluator( Cantera::ThermoPhase& thermo,
                       const AdiabaticEquilRxnMdl& adEq,
                       MixtureFraction& mixfrac,
                       const double fuelEnthalpy,
                       const double oxidEnthalpy );
    double evaluate( const double& t,
                     const double& p,
                     const std::vector<double>& ys );
    void set_adiab_reacted_state( const double f );
  };

};

//====================================================================
//====================================================================



#endif
