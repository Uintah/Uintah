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

#ifndef CanteraTableBuilder_h
#define CanteraTableBuilder_h

#include <tabprops/prepro/TableBuilder.h>
#include <tabprops/prepro/NonAdiabaticTableHelper.h>

#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>
#include <cantera/transport.h>

//====================================================================

/**
 *  @class DensityEvaluator
 *  @brief Evaluates the density, kg/m^3
 */
class DensityEvaluator : public StateVarEvaluator{
public:
  DensityEvaluator( Cantera_CXX::IdealGasMix & gas )
    : StateVarEvaluator( DENSITY, "Density" ),
      gas_( gas )
  {}

  /**
   *  @brief compute the density, g/cm^3
   *
   *  @param temperature : Units of Kelvin
   *  @param pressure    : Units of Pa
   *  @param species     : secies mass fractions
   */
  double evaluate( const double & temperature,
		   const double & pressure,
		   const std::vector<double> & species )
  {
    gas_.setState_TPY( temperature, pressure, &species[0] );
#   ifdef CGS_UNITS
    return gas_.density() * 1.0e-3;
#   else
    return gas_.density();
#   endif
  }

private:
  Cantera_CXX::IdealGasMix & gas_;
};

//====================================================================

/**
 *  @class ViscosityEvaluator
 *  @brief Evaluates the viscosity in Pa-s
 */
class ViscosityEvaluator : public StateVarEvaluator{
public:

  /**
   *  @param gas : Cantera object describing thermodynamic properties for an ideal gas
   *  mixture.
   *
   *  @param cTransport : Cantera transport evaluator. If no transport evaluator is
   *  provided, then mixture-averaged transport will be used.
   */
  ViscosityEvaluator( Cantera_CXX::IdealGasMix & gas,
		      Cantera::Transport * const cTransport = NULL )
    : StateVarEvaluator( VISCOSITY, "Viscosity" ),
      gas_( gas ),
      createdTransport_(  cTransport == NULL ),
      canteraTransport_( (cTransport != NULL) ? cTransport :
			 Cantera::TransportFactory::factory()->newTransport("Mix",&gas) )
  {}

  ~ViscosityEvaluator()
  {
    // delete the transport manager only if we created one
    // (i.e. none was provided in the constructor)
    if( createdTransport_ )
      delete canteraTransport_;
  }

  /**
   *  @brief compute the viscosity, g/(cm-s) - note that units for output are CGS while
   *  those for input are SI.
   *
   *  @param temperature : Kelvin
   *  @param pressure    : Pa  (N/m^2)
   *  @param species     : species mass fractions
   */
  double evaluate( const double & temperature,
		   const double & pressure,
		   const std::vector<double> & species )
  {
    gas_.setState_TPY( temperature, pressure, &species[0] );
#   ifdef CGS_UNITS
    return canteraTransport_->viscosity() * 10.0;
#   else
    return canteraTransport_->viscosity();
#   endif
  }

private:
  Cantera_CXX::IdealGasMix& gas_;
  const bool createdTransport_;
  Cantera::Transport * const canteraTransport_;
};

//====================================================================

/**
 *  @class EnthalpyEvaluator
 *  @brief Evaluates the mixture enthalpy (J/kg)
 */
class EnthalpyEvaluator : public StateVarEvaluator
{
public:
  EnthalpyEvaluator( Cantera::ThermoPhase & gas )
    : StateVarEvaluator( ENTHALPY, "Enthalpy" ),
      gas_( gas )
  {}

  double evaluate( const double & temperature,
		   const double & pressure,
		   const std::vector<double> & species )
  {
    gas_.setState_TPY( temperature, pressure, &species[0] );
#   ifdef CGS_UNITS
    return gas_.enthalpy_mass() * 1.0e4;  // convert J/kg to erg/g
#   else
    return gas_.enthalpy_mass();
#   endif
  }

private:
  Cantera::ThermoPhase & gas_;
};

//====================================================================

/**
 *  @class SpeciesEvaluator
 *  @brief Evaluates the species mass fractions (trivial)
 */
class SpeciesEvaluator : public StateVarEvaluator{
public:
  SpeciesEvaluator( Cantera_CXX::IdealGasMix & gas,
		    const std::string & specName )
    : StateVarEvaluator( SPECIES, specName ),
      specIx_( gas.speciesIndex( specName ) )
  {
    if( specIx_ < 0 ){
      std::ostringstream errmsg;
      errmsg << "No species named '" << specName << "' was found." << std::endl;
      throw std::runtime_error( errmsg.str() );
    }
  }

  virtual ~SpeciesEvaluator(){};

  /**
   *  @brief evaluate the species mass fractions.  This requires extraneous inputs because
   *  of the constraints imposed by the base class virtual method.
   */
  double evaluate( const double & temperature,
		   const double & pressure,
		   const std::vector<double> & species )
  {
    return species[specIx_];
  }

  bool operator == (const SpeciesEvaluator& s )
  {
    return (specIx_ == s.specIx_);
  }

 private:
  const int specIx_;
};

//====================================================================

/**
 *  @class MoleFracEvaluator
 *  @brief Evaluates the species mole fractions
 */
class MoleFracEvaluator : public StateVarEvaluator{
public:
  MoleFracEvaluator( Cantera_CXX::IdealGasMix & gas,
                     const std::string & specName )
    : StateVarEvaluator( MOLEFRAC, specName+"_molefrac" ),
      specIx_( gas.speciesIndex( specName ) ),
      gas_( gas )
  {
    if( specIx_ < 0 ){
      std::ostringstream errmsg;
      errmsg << "No species named '" << specName << "' was found." << std::endl;
      throw std::runtime_error( errmsg.str() );
    }
  }

  virtual ~MoleFracEvaluator(){};

  /**
   *  @brief evaluate the species mass fractions.  This requires extraneous inputs because
   *  of the constraints imposed by the base class virtual method.
   */
  double evaluate( const double & temperature,
		   const double & pressure,
		   const std::vector<double> & species )
  {
    gas_.setState_TPY( temperature, pressure, &species[0] );
    return gas_.moleFraction(specIx_);
  }

  bool operator == ( const MoleFracEvaluator& s )
  {
    return (specIx_ == s.specIx_);
  }

 private:
  const int specIx_;
  Cantera_CXX::IdealGasMix& gas_;
};

//====================================================================

/**
 *  @class SpecificHeatEvaluator
 *  @brief Evaluates the mixture specific heat at constant pressure, cp, J/(kg K)
 */
class SpecificHeatEvaluator : public StateVarEvaluator
{
public:
  SpecificHeatEvaluator( Cantera::ThermoPhase& thermo )
    : StateVarEvaluator( SPECIFIC_HEAT, "SpecificHeat" ),
      thermo_( thermo )
  {}

  double evaluate( const double & temperature,
		   const double & pressure,
		   const std::vector<double> & species )
  {
    thermo_.setState_TPY( temperature, pressure, &species[0] );
#   ifdef CGS_UNITS
    return thermo_.cp_mass() * 1.0e4;
#   else
    return thermo_.cp_mass();
#   endif
  }

private:
  Cantera::ThermoPhase& thermo_;
};

//====================================================================

/**
 *  @class ConductivityEvaluator
 *  @brief Evaluates the thermal conductivity of the mixture, W/(m K)
 */
class ConductivityEvaluator : public StateVarEvaluator
{
public:
  ConductivityEvaluator( Cantera_CXX::IdealGasMix & gas,
			 Cantera::Transport * const cTransport = NULL )
    : StateVarEvaluator( CONDUCTIVITY, "Conductivity" ),
      gas_( gas ),
      createdTransport_(  cTransport == NULL ),
      canteraTransport_( (cTransport != NULL) ? cTransport :
			 Cantera::TransportFactory::factory()->newTransport("Mix",&gas) )
  {}

  ~ConductivityEvaluator()
  {
    if( createdTransport_ ) delete canteraTransport_;
  }

  double evaluate( const double & temperature,
		   const double & pressure,
		   const std::vector<double> & species )
  {
    gas_.setState_TPY( temperature, pressure, &species[0] );
#   ifdef CGS_UNITS
    return canteraTransport_->thermalConductivity() * 1.0e5;
#   else
    return canteraTransport_->thermalConductivity();
#   endif
  }

private:
  Cantera_CXX::IdealGasMix & gas_;
  const bool createdTransport_;
  Cantera::Transport * const canteraTransport_;
};

//====================================================================

/**
 *  @class MolecularWeightEvaluator
 *  @brief Evaluates the mixture molecular weight, kg/kmol or g/mol
 */
class MolecularWeightEvaluator : public StateVarEvaluator
{
private:
  const Cantera::Constituents& mix_;
public:
  MolecularWeightEvaluator( const Cantera::Constituents& mix )
    : StateVarEvaluator( MIXTURE_MW, "MolecularWeight" ),
      mix_( mix )
  {}

  ~MolecularWeightEvaluator(){}

  double evaluate( const double& temperature,
                   const double& pressure,
                   const std::vector<double>& species )
  {
    double mixmw = 0;
    for( int i=0; i<species.size(); ++i ){
      mixmw += species[i] / mix_.molecularWeight(i);
    }
    return 1.0/mixmw;
  }
};

//====================================================================

/**
 *  @class ReactionRateEvaluator
 *  @brief Evaluates the reaction rate, kg/m^3/s
 */
class ReactionRateEvaluator : public StateVarEvaluator
{
private:
  Cantera::Kinetics& kin_;
  const int specIx_;
  std::vector<double> rr_;
public:
  ReactionRateEvaluator( Cantera::Kinetics& kin,
                         const std::string& specName )
    : StateVarEvaluator( SPECIES_RR, specName+"_rr" ),
      kin_( kin ),
      specIx_( kin_.thermo().speciesIndex( specName ) )
  {
    rr_.resize( kin_.thermo().nSpecies(), 0.0 );
  }

  ~ReactionRateEvaluator(){}

  double evaluate( const double& temperature,
                   const double& pressure,
                   const std::vector<double>& species )
  {
    kin_.thermo().setState_TPY( temperature, pressure, &species[0] );
    kin_.getNetProductionRates( &rr_[0] );

    const std::vector<double>& spmw = kin_.thermo().molecularWeights();

    // convert kmol/m^3/s to kg/m^3/s
    return rr_[specIx_] *= spmw[specIx_];
  }
};

//====================================================================

/**
 *  @class SpecEnthEvaluator
 *  @brief Evaluates the species enthalpies, J/kg
 */
class SpecEnthEvaluator : public StateVarEvaluator
{
private:
  Cantera::ThermoPhase& gas_;
  const int specIx_;
  std::vector<double> specEnth_;
public:
  SpecEnthEvaluator( Cantera::ThermoPhase& gas,
                     const std::string& specName )
    : StateVarEvaluator( SPECIES_ENTH, specName+"_enthalpy" ),
      gas_( gas ),
      specIx_( gas.speciesIndex( specName ) )
  {
    specEnth_.resize( gas.nSpecies(), 0.0 );
  }

  ~SpecEnthEvaluator(){}

  double evaluate( const double& temperature,
                   const double& pressure,
                   const std::vector<double>& species )
  {
    gas_.setState_TPY( temperature, pressure, &species[0] );

    const std::vector<double>& spmw = gas_.molecularWeights();
    gas_.getPartialMolarEnthalpies( &specEnth_[0] );

    // convert mole to mass
    return specEnth_[specIx_] /= spmw[specIx_];
  }
};

//====================================================================

#endif // CanteraTableBuilder_h
