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
#include <tabprops/prepro/rxnmdl/StreamMixing.h>
//--------------------------------------------------------------------
// std includes:
#include <stdexcept>
#include <sstream>
#include <vector>
#include <string>
using std::vector;
using std::string;

//====================================================================

StreamMixing::StreamMixing( Cantera_CXX::IdealGasMix & gas,
                            const vector<double> & y_oxid,
                            const vector<double> & y_fuel,
                            const bool haveMassFrac,
                            const int order,
                            const int nFpts )
  : ReactionModel( gas, indep_var_names(), order, "NonReacting" ),
    gasMix_( gas ),
    nFpts_( nFpts )
{
  fuelTemp_ = oxidTemp_ = 300.0;
  pressure_ = 101325.0;

  // set up the mixture fraction grid
  const double df = 1.0/(nFpts_-1);
  for( int i=0; i<nFpts_; ++i ) fpts_.push_back( df*double(i) );


  // consistency checking
  if( (nSpec_ != (int)y_oxid.size()) ||
      (nSpec_ != (int)y_fuel.size()) ){
    std::ostringstream errmsg;
    errmsg << "Inconsistent number of species in fuel and oxidizer streams!" << std::endl
           << "  Cantera: " << nSpec_ << ";  yox: " << y_oxid.size() << ";  yfuel: " << y_fuel.size()
           << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  // set up the mixture fraction object
  mixfr_ = new MixtureFraction( gasMix_, y_oxid, y_fuel, haveMassFrac );

    // check to make sure that other things are ready.
  if( !( mixfr_->is_ready() ) ){
    std::ostringstream errmsg;
    errmsg << "Mixture fraction did not sucessfully initialize." << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  // be sure that the cantera object is okay
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


  // set the name for the output file
  tableBuilder_.set_filename( "NonReacting" );
}
//--------------------------------------------------------------------
StreamMixing::~StreamMixing()
{
  delete mixfr_;
}
//--------------------------------------------------------------------
void
StreamMixing::set_fuel_temperature( const double temp )
{
  fuelTemp_ = temp;
  gasMix_.setState_TPY( fuelTemp_, pressure_, mixfr_->fuel_massfr() );
  fuelEnth_ = gasMix_.enthalpy_mass();
}
//--------------------------------------------------------------------
void
StreamMixing::set_oxid_temperature( const double temp )
{
  oxidTemp_ = temp;
  gasMix_.setState_TPY( oxidTemp_, pressure_, mixfr_->oxid_massfr() );
  oxidEnth_ = gasMix_.enthalpy_mass();
}
//--------------------------------------------------------------------
void
StreamMixing::set_pressure( const double pres )
{
  pressure_ = pres;

  // reset fuel and oxidizer enthalpies to be consistent with this pressure.
  set_fuel_temperature( fuelTemp_ );
  set_oxid_temperature( oxidTemp_ );
}
//--------------------------------------------------------------------
void
StreamMixing::implement()
{
  tableBuilder_.set_mesh( 0, fpts_ );

  vector<double> ys(nSpec_,0.0);

  for( vector<double>::const_iterator imf=fpts_.begin(); imf!=fpts_.end(); ++imf ){

    const double f = *imf;

    // get the mixture composition
    mixfr_->mixfrac_to_species(f,ys);

    const double enth = f*fuelEnth_ + (1.0-f)*oxidEnth_;
    const double tguess = f*fuelTemp_ + (1.0-f)*oxidTemp_;

    // set the mixture properties
    gasMix_.setState_TPY( tguess, pressure_, &(ys[0]) );
    gasMix_.setState_HP( enth, pressure_ );

    // add this entry to the table
    vector<double> fentry(1,f);
    tableBuilder_.insert_entry( fentry, gasMix_.temperature(), pressure_, ys );
  }
  // generate the table
  tableBuilder_.generate_table();
}
//--------------------------------------------------------------------
const vector<string>&
StreamMixing::indep_var_names()
{
  static vector<string> names(1);
  names[0] = "MixtureFraction";
  return names;
}
//--------------------------------------------------------------------
