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

#ifndef FAST_CHEM_RXN_MDL_H
#define FAST_CHEM_RXN_MDL_H

#include <vector>

#include "ReactionModel.h"

// forward declarations

class MixtureFraction;

//====================================================================
//====================================================================

//--------------------------------------------------------------------
/**
 *  @class FastChemRxnMdl
 *  @brief Reaction model for infinitely fast, complete combustion.
 *
 *  @author James C. Sutherland
 *  @date   April, 2005
 *
 *  Note that this model is slightly different than the classical Burke-Schumann
 *  solution in that it does not assume constant and equal heat capacities for
 *  all species.  Thus, while enthalpy and composition are the classical
 *  piecewise-linear functions, the temperature is not a linear function of
 *  mixture fraction.
 *
 *  Since temperature is not a linear function of mixture fraction, a modest
 *  number of points should be used when constructing the model.
 */
class FastChemRxnMdl : public ReactionModel {

 public:

  /**
   *  @brief Construct an instance of an Adiabatic FastChemRxnMdl.
   *
   *  @param gas       : Cantera ideal gas object
   *  @param y_oxid    : composition of the oxidizer stream
   *  @param y_fuel    : composition of the fuel stream
   *  @param haveMassFrac : flag specifying whether mass or mole
   *                     fractions were provided.
   *  @param order       the order of interpolant to use
   *  @param nFpts     : number of points in mixture fraction space to
   *                     evaluate and tabulate the model at.
   *  @param fScaleFac : OPTIONAL grid compression factor for mixture fraction mesh.
   */
  FastChemRxnMdl( Cantera_CXX::IdealGasMix & gas,
                  const std::vector<double> & y_oxid,
                  const std::vector<double> & y_fuel,
                  const bool haveMassFrac,
                  const int order,
                  const int nFpts,
                  const double fScaleFac = 3.0 );


  /**
   *  @brief Construct an instance of an FastChemRxnMdl with heat release.
   *
   *  @param gas       : Cantera ideal gas object
   *  @param y_oxid    : composition of the oxidizer stream
   *  @param y_fuel    : composition of the fuel stream
   *  @param haveMassFrac : flag specifying whether mass or mole
   *                     fractions were provided.
   *  @param order       the order of interpolant to use
   *  @param nFpts     : number of points in mixture fraction space to
   *                     evaluate and tabulate the model at.
   *  @param nHLpts    : number of points in heat loss space to
   *                     evaluate and tabulate the model at.
   *  @param fScaleFac : OPTIONAL grid compression factor for mixture fraction mesh.
   */
  FastChemRxnMdl( Cantera_CXX::IdealGasMix & gas,
                  const std::vector<double> & y_oxid,
                  const std::vector<double> & y_fuel,
                  const bool haveMassFrac,
                  const int order,
                  const int nFpts,
                  const int nHLpts,
                  const double fScaleFac = 3.0 );

  ~FastChemRxnMdl();

    /** @brief Set the temperature for the fuel stream (K) */
  void set_fuel_temperature( const double temp );

  /** @brief Set the temperature for the oxidizer stream (K) */
  void set_oxid_temperature( const double temp );

  /** @brief Set the system pressure in Pa (assumed constant) */
  void set_pressure( const double press );

  /** @brief Implements the fast chemistry reaction model */
  void implement();

  std::vector<std::string> indep_var_names( const bool isAdiabatic );

  MixtureFraction * get_mixture_fraction() const{ return mixfr_; }


  static void set_species_enthalpies( Cantera::ThermoPhase& thermo,
                                      const double* const ys,
                                      const double  temp,
                                      const double  pressure,
                                      double* h );
 private:

  const bool isAdiabatic_;

  const int nFpts_, nHLpts_;

  double fuelTemp_, oxidTemp_, fuelEnth_, oxidEnth_, pressure_;
  double scaleFac_;

  MixtureFraction       * mixfr_;
  Cantera_CXX::IdealGasMix  & gasMix_;

  std::vector<double> fpts_, gammaPts_;

  std::vector<double> spwrk1, spwrk2;

  void set_up( const std::vector<double> & y_oxid,
               const std::vector<double> & y_fuel,
               const bool haveMassFrac );

  double product_temperature( const double gamma,
                              const double f,
                              const std::vector<double> & yo,
                              const std::vector<double> & yProd,
                              const double Tguess );

  void set_stoichiometry();


  void set_f_mesh();
  void adjust_f_mesh();


class SensEnthEvaluator : public StateVarEvaluator
{
private:
  Cantera::ThermoPhase& thermo_;
  MixtureFraction& mixfrac_;
  const double hfuel_, hoxid_;
  std::vector<double> yur_, ho_, hs_;
public:
  SensEnthEvaluator( Cantera::ThermoPhase& thermo,
                     MixtureFraction& mixfrac,
                     const double fuelEnthalpy,
                     const double oxidEnthalpy );
  double evaluate( const double& t,
                   const double& p,
                   const std::vector<double>& ys );
};

};


#endif
