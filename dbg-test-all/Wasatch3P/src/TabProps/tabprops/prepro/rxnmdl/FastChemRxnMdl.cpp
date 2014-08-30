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

//--------------------------------------------------------------------
// Cantera includes for thermochemistry
#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>    // defines class IdealGasMix
//--------------------------------------------------------------------
// SGS code includes:
#include <tabprops/prepro/rxnmdl/MixtureFraction.h>
#include <tabprops/prepro/rxnmdl/FastChemRxnMdl.h>
#include <tabprops/prepro/NonAdiabaticTableHelper.h>
//--------------------------------------------------------------------
// std includes:
#include <stdexcept>
#include <sstream>
#include <vector>
#include <string>
using std::vector;
using std::string;
#include <cassert>
//====================================================================
//====================================================================

FastChemRxnMdl::FastChemRxnMdl(  Cantera_CXX::IdealGasMix & gas,
                                 const vector<double> & y_oxid,
                                 const vector<double> & y_fuel,
                                 const bool haveMassFrac,
                                 const int order,
                                 const int nFpts,
                                 const double fScaleFac )
  : ReactionModel( gas,
                   indep_var_names(true),
                   order,
                   "Adiabatic Infinitely Fast Chemistry" ),

    isAdiabatic_( true ),

    nFpts_ ( nFpts  ),
    nHLpts_( 1 ),

    fuelTemp_( 300.0 ),           // default temperature (K)
    oxidTemp_( 300.0 ),           // default temperature (K)

    pressure_( 101325.0 ),        // default pressure (Pa)

    scaleFac_( fScaleFac ),

    mixfr_( NULL ),

    gasMix_( gas )
{
  tableBuilder_.set_filename( "AdiabaticFastChem" );

  set_up( y_oxid, y_fuel, haveMassFrac );
}
//--------------------------------------------------------------------
FastChemRxnMdl::FastChemRxnMdl( Cantera_CXX::IdealGasMix & gas,
                                const vector<double> & y_oxid,
                                const vector<double> & y_fuel,
                                const bool haveMassFrac,
                                const int order,
                                const int nFpts,
                                const int nHLpts,
                                const double fScaleFac )
  : ReactionModel( gas, indep_var_names(false), order, "Infinitely Fast Chemistry" ),

    isAdiabatic_( false ),

    nFpts_ ( nFpts  ),
    nHLpts_( nHLpts ),

    fuelTemp_( 300.0 ),           // default temperature (K)
    oxidTemp_( 300.0 ),           // default temperature (K)

    pressure_( 101325.0 ),        // default pressure (Pa)

    scaleFac_( fScaleFac ),

    mixfr_( NULL ),

    gasMix_( gas )
{
  tableBuilder_.set_filename( "FastChem" );

  set_up( y_oxid, y_fuel, haveMassFrac );

  // for the non-adiabatic case, we need the chemical and adiabatic enthalpies so that we
  // can get the heat loss parameter.
  tableBuilder_.request_output( new AdEnthEvaluator( *mixfr_, fuelEnth_, oxidEnth_ ) );
  tableBuilder_.request_output( new SensEnthEvaluator( gasMix_, *mixfr_, fuelEnth_, oxidEnth_ ) );
  tableBuilder_.request_output( StateVarEvaluator::ENTHALPY    );
  tableBuilder_.request_output( StateVarEvaluator::TEMPERATURE );
}
//--------------------------------------------------------------------
FastChemRxnMdl::~FastChemRxnMdl()
{
  delete mixfr_;
}
//--------------------------------------------------------------------
vector<string>
FastChemRxnMdl::indep_var_names( const bool isAdiabatic )
{
  vector<string> varNames;
  varNames.push_back( string("MixtureFraction") );
  if( !isAdiabatic )
    varNames.push_back( string("HeatLoss") );

  return varNames;
}
//--------------------------------------------------------------------
void
FastChemRxnMdl::set_up( const vector<double> & y_oxid,
                        const vector<double> & y_fuel,
                        const bool haveMassFrac )
{
  // consistency checking.
  if( !isAdiabatic_ ){
    if( nHLpts_ < 3 ){
      std::ostringstream errmsg;
      errmsg << "Must have at least three points in heat loss dimension" << std::endl;
      throw std::runtime_error(errmsg.str());
    }
  }

  if( nFpts_  < 3 ){
    std::ostringstream errmsg;
    errmsg << "Must have at least three points in mixture fraction dimension" << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  if( (nSpec_ != (int)y_oxid.size()) ||
      (nSpec_ != (int)y_fuel.size()) ){
    std::ostringstream errmsg;
    errmsg << "Inconsistent number of species in fuel and oxidizer streams!" << std::endl
           << "  Cantera: " << nSpec_ << ";  yox: " << y_oxid.size() << ";  yfuel: " << y_fuel.size()
           << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  spwrk1.assign(nSpec_,0.0);
  spwrk2.assign(nSpec_,0.0);

  // set up the mixture fraction
  mixfr_ = new MixtureFraction( gasMix_, y_oxid, y_fuel, haveMassFrac );

  set_f_mesh();
  adjust_f_mesh();

  // set up the heat-loss mesh
  if( isAdiabatic_ ){
    gammaPts_.push_back( 0.0 );
  }
  else{
    gammaPts_.resize( nHLpts_ );
    for( int i=0; i<nHLpts_; ++i ){
      const double gam = double(2*i)/(nHLpts_-1) -1.0;
      gammaPts_[i] = gam;
    }
  }

  // check to make sure that other things are ready.
  if( !( mixfr_->is_ready() ) ){
    std::ostringstream errmsg;
    errmsg << "Mixture fraction did not sucessfully initialize." << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  if( !(gasMix_.ready() ) ){
    std::ostringstream errmsg;
    errmsg << "Cantera object is invalid!" << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  //
  //  set fuel enthalpy based on default temperatures
  //
  gasMix_.setState_TPY( fuelTemp_, pressure_, mixfr_->fuel_massfr() );
  fuelEnth_ = gasMix_.enthalpy_mass();

  // consistency check:
  gasMix_.setState_HP( fuelEnth_, pressure_ );
  if( fabs(fuelTemp_ - gasMix_.temperature()) > 0.001 ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Problems in fuel stream specification." << std::endl
           << "       Temperature and enthalpy appear to be inconsistent" << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  //
  //  set oxidizer enthalpy based on default temperatures
  //
  gasMix_.setState_TPY( oxidTemp_, pressure_, mixfr_->oxid_massfr() );
  oxidEnth_ = gasMix_.enthalpy_mass();

  // consistency check:
  gasMix_.setState_HP( oxidEnth_, pressure_ );

  if( fabs(oxidTemp_ - gasMix_.temperature()) > 0.001 ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Problems in oxidizer stream specification." << std::endl
           << "       Temperature and enthalpy appear to be inconsistent" << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

}
//--------------------------------------------------------------------
void
FastChemRxnMdl::set_fuel_temperature( const double temp )
{
  fuelTemp_ = temp;
  gasMix_.setState_TPY( fuelTemp_, pressure_, mixfr_->fuel_massfr() );
  fuelEnth_ = gasMix_.enthalpy_mass();
}
//--------------------------------------------------------------------
void
FastChemRxnMdl::set_oxid_temperature( const double temp )
{
  oxidTemp_ = temp;
  gasMix_.setState_TPY( oxidTemp_, pressure_, mixfr_->oxid_massfr() );
  oxidEnth_ = gasMix_.enthalpy_mass();
}
//--------------------------------------------------------------------
void
FastChemRxnMdl::set_pressure( const double pres )
{
  pressure_ = pres;

  // reset fuel and oxidizer enthalpies to be consistent with this pressure.
  set_fuel_temperature( fuelTemp_ );
  set_oxid_temperature( oxidTemp_ );
}
//--------------------------------------------------------------------
void
FastChemRxnMdl::implement()
{
  double Tprod = oxidTemp_;
  vector<double> yo(nSpec_), yprod(nSpec_), ho_i(nSpec_);

  // for adiabatic, only one independent variable.
  vector<double> indepVar( isAdiabatic_ ? 1 : 2 );

  // set up the mesh for the table
  {
    vector< vector<double> > mesh( isAdiabatic_ ? 1 : 2 );
    mesh[0] = fpts_;
    if( !isAdiabatic_ ) mesh[1] = gammaPts_;
    tableBuilder_.set_mesh( mesh );
    mesh.clear();
  }

  for( int imf=0; imf<nFpts_; imf++ ){

    // set f this way rather than incrementing (i.e. f++)
    // because we might tweak f later on.
    const double f = fpts_[imf];
    indepVar[0] = f;

    //  set the reactant and product composition
    mixfr_->mixfrac_to_species( f, yo );
    mixfr_->estimate_product_comp( f, yprod, true );

    //
    //  Loop over heat loss
    //
    for( int ihl=0; ihl<nHLpts_; ihl++ ){

      // we may only have one entry for an adiabatic case:
      const double gamma = gammaPts_[ihl];

      if( !isAdiabatic_ ) indepVar[1]= gamma;

      //
      //  set the product temperature by an energy balance,
      //  using the previous temperature as a guess
      //
      Tprod = product_temperature( gamma, f, yo, yprod, Tprod );

//      std::cout << f << "  " << gamma << "  " << Tprod << std::endl;

      // add this entry to the table.
      tableBuilder_.insert_entry( indepVar, Tprod, pressure_, yprod );

    }
  }

  // we are done adding entries to the model.  Generate the table
  tableBuilder_.generate_table();
}
//--------------------------------------------------------------------
double
FastChemRxnMdl::product_temperature( const double gamma,
                                     const double f,
                                     const vector<double> & yo,
                                     const vector<double> & yProd,
                                     const double Tguess )
{
  // set the unreacted, adiabatic state.
  // Here we use a guess for the temperature, but the mixture enthalpy
  // (assuming an ideal mixture) is really used to set the state.
  const double ha     = f*fuelEnth_ + (1.0-f)*oxidEnth_;
  const double tguess = f*fuelTemp_ + (1.0-f)*oxidTemp_;
  gasMix_.setState_TPY( tguess, pressure_, &(yo[0]) );
  gasMix_.setState_HP( ha, pressure_ );

  // get the species enthalpies at this state (reference state)
  vector<double> & ho_i = spwrk1;
  const double Tmix = gasMix_.temperature();
  set_species_enthalpies( gasMix_, &(yo[0]), Tmix, pressure_, &(ho_i[0]) );

  //----------------------------------------------------------------
  // we can parameterize heat loss as:
  //
  //              ha - h      ha - h
  //     gamma = --------- = --------
  //              ha - hc      hs
  //
  // where ha  is the enthalpy at adiabatic conditions (unreacted mixture enthalpy),
  //       h   is the enthalpy of the system with the induced heat loss,
  //       hc  is the chemical enthalpy,
  //       hs is the sensible enthalpy.
  //
  // Thus, h is a function of mixture fraction for a chosen gamma.
  //----------------------------------------------------------------

  //
  //  enthalpy of the reacted mixture is the same as the reactants
  //  (assuming constant pressure), so now we must solve for the
  //  temperature given the reacted mixture compostion.
  //
  gasMix_.setState_TPY( Tguess, pressure_, &(yProd[0]) );
  gasMix_.setState_HP( ha, pressure_ );

  const double TAdiab = gasMix_.temperature();

  // compute the sensible enthalpy of the mixture at adiabatic conditions
  vector<double> & ha_i = spwrk2;
  set_species_enthalpies( gasMix_, &(yProd[0]), TAdiab, pressure_, &(ha_i[0]) );

  double hs = 0;
  for( int i=0; i<nSpec_; i++ )  hs += yProd[i]*( ha_i[i]-ho_i[i] );

  const double h = ha - gamma*hs;
  gasMix_.setState_HP( h, pressure_ );

  return( gasMix_.temperature() );
}
//--------------------------------------------------------------------
void
FastChemRxnMdl::set_species_enthalpies( Cantera::ThermoPhase& thermo,
                                        const double* const ys,
                                        const double  temp,
                                        const double  pressure,
                                        double* h )
{
  thermo.setState_TPY( temp, pressure, ys );
  thermo.getEnthalpy_RT( h );  // nondimensional molar enthalpies

  // scale the enthalpy to get it in mass units
  const double RT = Cantera::GasConstant * temp;
  const int nSpec = thermo.nSpecies();
  for( int i=0; i<nSpec; i++ ){
    h[i] *= RT / thermo.molecularWeight(i);
  }
}
//--------------------------------------------------------------------
void
FastChemRxnMdl::set_f_mesh()
{
  fpts_.assign(nFpts_,0.0);
  const double xcrit = mixfr_->stoich_mixfrac();

  assert( nFpts_ > 1 );

  // first create a uniform mesh
  const double dfTmp = 1.0/(nFpts_-1);
  for( int i=0; i<nFpts_; i++ ){
    fpts_[i] = double(i) * dfTmp;
  }
  const double xo = 1.0/(2.0*scaleFac_) * log( (1.+(exp( scaleFac_)-1.)*xcrit) /
                                               (1.+(exp(-scaleFac_)-1.)*xcrit) );

  const double A = sinh(scaleFac_*xo);
  fpts_[0] = 0.0;
  for( int i=1; i<nFpts_-1; i++ ){
    fpts_[i] = xcrit/A * (sinh(scaleFac_*(fpts_[i]-xo)) + A);
    assert( fpts_[i] >= 0.0  && fpts_[i] <= 1.0 );
  }
  fpts_[nFpts_-1] = 1.0;
}
//--------------------------------------------------------------------
void
FastChemRxnMdl::adjust_f_mesh()
{
  const double fst = mixfr_->stoich_mixfrac();
  int ixlo = 0;
  for( int i=0; i<nFpts_; i++ ){
    if( fpts_[i] <= fst )  ixlo=i;
  }
  int ix=ixlo;
  if( (fst-fpts_[ixlo]) > (fpts_[ixlo+1]-fst) )
    ix = ixlo+1;
  fpts_[ix] = fst;
}

//====================================================================

FastChemRxnMdl::SensEnthEvaluator::
SensEnthEvaluator( Cantera::ThermoPhase& thermo,
                   MixtureFraction & mixfrac,
                   const double fuelEnthalpy,
                   const double oxidEnthalpy )
  : StateVarEvaluator( SENS_ENTH, "SensibleEnthalpy" ),
    thermo_ ( thermo ),
    mixfrac_( mixfrac ),
    hfuel_( fuelEnthalpy ),
    hoxid_( oxidEnthalpy ),
    yur_( thermo.nSpecies(), 0.0 ),
    ho_ ( thermo.nSpecies(), 0.0 ),
    hs_ ( thermo.nSpecies(), 0.0 )
{}
//--------------------------------------------------------------------
double
FastChemRxnMdl::SensEnthEvaluator::
evaluate( const double& t,
          const double& p,
          const std::vector<double>& ys )
{
  // set the mixture fraction and the unreacted composition
  double f=0.0;
  mixfrac_.species_to_mixfrac( ys, f );
  mixfrac_.mixfrac_to_species( f, yur_ );

  // adiabatic enthalpy of the mixture
  const double ha = f * hfuel_ + (1.0-f)*hoxid_;

  // calculate the species enthalpies at the unreacted state (reference state)
  thermo_.setState_TPY( 300, p, &yur_[0] );
  thermo_.setState_HP( ha, p );
  thermo_.getPartialMolarEnthalpies( &ho_[0] );

  // calculate species enthalpies at the reacted state using the
  // adiabatic enthalpy
  thermo_.setState_TPY( t, p, &ys[0] );
  thermo_.setState_HP( ha, p );
  thermo_.getPartialMolarEnthalpies( &hs_[0] );

  // calculate the mixture enthalpy at adiabatic conditions
  double hsens = 0.0;
  for( size_t i=0; i<thermo_.nSpecies(); ++i ){
    hsens += ys[i] * ( hs_[i] - ho_[i] ) / thermo_.molecularWeight(i);
  }
#ifdef CGS_UNITS
  return hsens * 1.0e4;
#else
  return hsens;
#endif
}
