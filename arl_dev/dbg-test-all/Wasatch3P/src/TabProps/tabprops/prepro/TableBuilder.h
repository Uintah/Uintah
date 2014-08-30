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

#ifndef TableBuilder_h
#define TableBuilder_h

#include <string>
#include <set>
#include <vector>
#include <map>

#include <sstream>  // only required for the StateVars extraction operator

#include <tabprops/TabProps.h>


//====================================================================

/**
 * \class  StateVarEvaluator
 * \author James C. Sutherland
 * \brief Evaluate various properties given the state (T,p,yi) of the system.
 */
class StateVarEvaluator{
 public:

  /**
   *  @enum StateVars
   *  @brief Enumerates the supported output variables
   *
   *  Supported state variables (dependent variables) for output in
   *  the table.  Density and viscosity will always be calculated, as
   *  they are required for any fluids simulation.  Other variables
   *  may be added as needed here.
   *
   *  Note: \f$ h = \sum_{i} \left( hc_i + hs_i \right)\f$ where
   *  \f$hc_i\f$ is the chemical enthalpy of species \f$i\f$ and
   *  \f$hs_i\f$ is the sensible enthalpy of species i.
   */
  enum StateVars{
    DENSITY       = 0,   ///< Mixture mass density (kg/m^3) - name "Density"
    VISCOSITY     = 1,   ///< Viscosity (kg/(m s)) - name "Viscosity"
    SPECIFIC_HEAT = 2,   ///< Mixture isobaric heat capacity, Cp (J/(kg K)) - name "SpecificHeat"
    CONDUCTIVITY  = 3,   ///< W/(m K) - name "Conductivity"

    ENTHALPY      = 5,   ///< Mixture enthalpy (J/kg) - name "Enthalpy"
    AD_ENTH       = 7,   ///< Mixture adiabatic enthalpy (J/kg)
    SENS_ENTH     = 8,   ///< Sensible enthalpy (J/kg)

    TEMPERATURE   = 20,  ///< Temperature (K) - name "Temperature"
    SPECIES       = 40,  ///< Species mass fractions - name "Species"
    SPECIES_RR    = 41,  ///< Species mass reaction rates (kg/m^3/s) - name "ReactionRate"
    SPECIES_ENTH  = 42,  ///< Species enthalpies (J/kg) - name "SpeciesEnthalpy"
    MOLEFRAC      = 43,  ///< Species mole fractions - name "MoleFrac"

    MIXTURE_MW    = 45,  ///< Mixture molecular weight (kg/kmol) - name "MolecularWeight"
  };


  StateVarEvaluator( const StateVars & id,
                     const std::string & name )
    : varID_( id ),
      evaluatorName_( name )
  {}

  virtual ~StateVarEvaluator(){};

  virtual double evaluate( const double & temperature,
                           const double & pressure,
                           const std::vector<double> & species ) = 0;

  const StateVars & get_id() const{ return varID_; }

  const std::string & name() const{ return evaluatorName_; }
  std::string get_name() const{ return evaluatorName_; }

protected:

  const StateVars varID_;
  const std::string evaluatorName_;

private:
  StateVarEvaluator( const StateVarEvaluator& );         // no copying
};



std::istringstream & operator >> ( std::istringstream & istr,
                                   StateVarEvaluator::StateVars& v );

StateVarEvaluator::StateVars get_state_var( const std::string& varname );

class StateVarLess : public std::binary_function<StateVarEvaluator* , StateVarEvaluator* , bool>
{
public:
  bool operator()(const StateVarEvaluator* const s , const StateVarEvaluator* const t ) const
  { return (s->name() < t->name() ); }
};

//====================================================================

class TemperatureEvaluator : public StateVarEvaluator
{
public:
  TemperatureEvaluator( )
    : StateVarEvaluator( TEMPERATURE, "Temperature" )
  {}

  virtual ~TemperatureEvaluator(){}

  /** compute the temperature, K */
  double evaluate( const double & temperature,
                   const double & pressure,
                   const std::vector<double> & species )
  {
    return temperature;
  }
};

//====================================================================

// forward declaration
namespace Cantera_CXX{ class IdealGasMix; }

/**
 *  @class  TableBuilder
 *  @author James C. Sutherland
 *  @date   January, 2006
 *
 *  Provides a common interface to build reaction tables for use in
 *  either a mixing model or a simulation.  Reaction model output
 *  should be crunched through this to get it in a Fuego-accepted
 *  format.
 *
 *  A property evaluation package is required.  Currently, Cantera is
 *  the only package supported.  It is freely available (see
 *  www.cantera.org)
 */
class TableBuilder{

 public:

  /**
   *  @param gas  Cantera object to evaluate thermodynamic properties of ideal gas mixtures.
   *  @param indepVarNames names of the independent variables
   *  @param order Interpolation order. Low order interpolation is less expensive.
   */
  TableBuilder( Cantera_CXX::IdealGasMix & gas,
                const std::vector<std::string> & indepVarNames,
                const int order );

  ~TableBuilder();

  void set_mesh( const int dimension,
                 const std::vector<double> & mesh );

  void set_mesh( const std::vector< std::vector<double> > & mesh );


  /**
   *  Insert a new entry in the table.  The entry will be copied into
   *  the table.  This must be done one point at a time.  Provide the
   *  independent variables at the point, along with T, P, Yi.  The
   *  requested output variables will then be calculated and
   *  automatically loaded in the table.
   *
   *  @param indepVar : the independent variables at the given point.  Fortran-style
   *  indexing, i.e. first dimension varies fastest.
   *
   *  @param temperature : temperature (K) corresponding to the given independent variables
   *
   *  @param pressure : pressure (Pa) corresponding to the given independent variables
   *
   *  @param species : vector of species mass fractions corresponding to the given
   *  independent variables
   */
  void insert_entry( const std::vector<double> & indepVar,
                     const double temperature,
                     const double pressure,
                     const std::vector<double> & species );

  /**
   *  Construct the entire table in one shot.  Given the interpolant
   *  entries and the independent variables, this constructs the
   *  table.
   *
   *  @param interpT interpolant for the temperature (K)
   *  @param interpP interpolant for the pressure (Pa)
   *  @param interpY interpolant for the species mass fractions.
   */
  void insert( const InterpT & interpT,
               const InterpT & interpP,
               const std::vector<const InterpT*> & interpY );

  /**
   *  The output file will always have a ".h5" suffix.  There is a
   *  default output filename, which can be reset by calling this
   *  function
   */
  void set_filename( const std::string & prefix ){ tablePrefix_ = prefix; }


  /** request that a specified variable be written to the table */
  void request_output( const StateVarEvaluator::StateVars stateVar );

  /** request that the species with the given name be output to the table */
  void request_output( const StateVarEvaluator::StateVars stateVar,
                       const std::string & speciesName );

  void request_output( StateVarEvaluator * const varEval );

  /**
   *  Generate the state table with the output variables that have
   *  been requested via the request_output methods.  The table will
   *  be written to the file specified by the set_filename method, or
   *  to the default file name.
   */
  void generate_table();

protected:

  Cantera_CXX::IdealGasMix & canteraProps_;
  const std::vector<std::string> indepVarNames_;
  const int nDim_;
  const int order_;

  std::string tablePrefix_;

  typedef std::set<StateVarEvaluator*, StateVarLess> OutputRequest;
  OutputRequest requestedOutputVars_;

  std::vector< std::vector<double> > mesh_;

  typedef std::map<std::string,std::vector<double> > PropertyEntries;
  PropertyEntries propEntries_;

  std::vector<int> npts_;

  int totEntries_;
  bool firstEntry_;

private:
  TableBuilder( const TableBuilder& ); // no copy constructor
};


#endif
