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
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <sstream>
#include <cassert>
#include <iostream>
using std::vector;
using std::map;
using std::string;
using std::cout;
using std::endl;
//--------------------------------------------------------------------
// Cantera includes for thermochemistry
#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>    // defines class IdealGasMix
#include <cantera/equilibrium.h>    // chemical equilibrium
//--------------------------------------------------------------------
#include <tabprops/TabProps.h>
#include <tabprops/prepro/rxnmdl/MixtureFraction.h>
#include <tabprops/prepro/rxnmdl/EquilRxnMdl.h>
#include <tabprops/prepro/NonAdiabaticTableHelper.h>
//--------------------------------------------------------------------

//====================================================================
//====================================================================

AdiabaticEquilRxnMdl::AdiabaticEquilRxnMdl( Cantera_CXX::IdealGasMix & gas,
                                            const vector<double> & y_oxid,
                                            const vector<double> & y_fuel,
                                            const bool haveMassFrac,
                                            const int interpOrder,
                                            const int nFpts,
                                            const double scaleFac )
  : ReactionModel( gas,                               // Cantera_CXX::IdealGasMix object
                   indep_var_names(),                 // names of independent variables
                   interpOrder,                       // order for interpolation
                   string("Adiabatic Equilibrium") ), // set model name

    nFpts_( nFpts),
    scaleFac_( scaleFac ),

    pressVary_( false ),          // do not include pressure dependence
    Tfuel_( 300.0 ),              // default fuel     temperature (K)
    Toxid_( 300.0 ),              // default oxidizer temperature (K)
    pressure_( 101325.0 ),        // default pressure (Pa)
    mixfr_( new MixtureFraction( gas, y_oxid, y_fuel, haveMassFrac ) )
{
  outputTable_ = true;  // by default, dump the state table.

  if( pressVary_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Cannot yet handle variable pressure in reaction models." << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  // consistency checking.
  if( nSpec_ != (int)y_oxid.size() ||
      nSpec_ != (int)y_fuel.size() ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Inconsistent number of species in fuel or oxidizer stream composition." << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  // set up the mixture fraction mesh
  set_f_mesh();

  // set the table name and establish the mesh for the table.
  tableBuilder_.set_filename( "AdiabaticEquil" );
}
//--------------------------------------------------------------------
AdiabaticEquilRxnMdl::~AdiabaticEquilRxnMdl()
{
  if( NULL != mixfr_ ) delete mixfr_;
}
//--------------------------------------------------------------------
const vector<string> &
AdiabaticEquilRxnMdl::indep_var_names(){
  static vector<string> names(1);
  names[0] = "MixtureFraction";
  return names;
}
//--------------------------------------------------------------------
void
AdiabaticEquilRxnMdl::set_f_mesh()
{
  fpts_.assign(nFpts_,0.0);
  const double xcrit = mixfr_->stoich_mixfrac();

  if( nFpts_ <= 1 ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Must have at least two points in mixture fraction space" << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

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
AdiabaticEquilRxnMdl::implement()
{
  // set enthalpy of the fuel and oxidizer streams
  gasProps_.setState_TPY( Tfuel_, pressure_, mixfr_->fuel_massfr() );
  const double hFuel = gasProps_.enthalpy_mass();

  gasProps_.setState_TPY( Toxid_, pressure_, mixfr_->oxid_massfr() );
  const double hOxid = gasProps_.enthalpy_mass();

  vector<double> ys(nSpec_);

  //
  // Loop over mixture fractions
  //
  double Tguess = Toxid_;

  vector<double>::iterator imixfr;
  for( imixfr=fpts_.begin(); imixfr!=fpts_.end(); imixfr++ ){

    const double f = *imixfr;

    // set the mixture composition for this mixture fraction
//    mixfr_->estimate_product_comp( f, ys, true );
    mixfr_->mixfrac_to_species( f, ys );

    // set mixture enthalpy
    const double hMix = f*hFuel + (1.0-f)*hOxid;

    // set the "gas" object composition and enthalpy
    gasProps_.setState_TPY( Tguess, pressure_, &(ys[0]) );
    gasProps_.setState_HP( hMix, pressure_ );

    // compute the equilibrium composition at constant pressure and enthalpy
    if( (f > 0.0)  && (f < 1.0) ){
      Cantera::ChemEquil eq;
      try{
        eq.options.relTolerance = 1.0e-6;
        eq.equilibrate( gasProps_, "HP" );

        // update guess for temperature
        Tguess = gasProps_.temperature();

        // save the equilibrium state
        gasProps_.getMassFractions( &(ys[0]) );
        stateVars_.push_back( EqStateVars( gasProps_.temperature(), pressure_, ys ) );

      }
      catch(Cantera::CanteraError&){
//      showErrors(cout);
        cout << "Error occured at a mixture fraction of " << f
             << ";  T=" << gasProps_.temperature()
             << ",  Tguess=" << Tguess << endl
             << "   This entry will be omitted." << endl;
        // remove this entry from the table
        imixfr = fpts_.erase(imixfr);
        --imixfr;
        --nFpts_;

        continue;
      }
    }
    else{
      stateVars_.push_back( EqStateVars(gasProps_.temperature(), pressure_, ys) );
    }


  } // loop over mixture fraction

  // if we want to dump the table, do so now.
  if( outputTable_ ){

    // load the information into the TableBuilder
    tableBuilder_.set_mesh( 0, fpts_ );

    if( stateVars_.size() != fpts_.size() ){
      std::ostringstream errmsg;
      errmsg << "ERROR: inconsistent size of vectors 'fpts_' (" << fpts_.size()
             << ") and 'stateVars_' (" << stateVars_.size() << ")" << std::endl;

      throw std::runtime_error( errmsg.str() );
    }

    vector<double> point(1);
    vector<double>::const_iterator imf;
    vector<EqStateVars>::const_iterator ist=stateVars_.begin();
    for( imf=fpts_.begin(); imf!=fpts_.end(); ++imf, ++ist ){
      point[0] = *imf;
      tableBuilder_.insert_entry( point, ist->temp, ist->press, ist->ys );
    }
    tableBuilder_.generate_table();
  }

}

//====================================================================
//====================================================================

EquilRxnMdl::EquilRxnMdl( Cantera_CXX::IdealGasMix & gas,
                          const std::vector<double> & y_oxid,
                          const std::vector<double> & y_fuel,
                          const bool haveMassFrac,
                          const int order,
                          const int nFpts,
                          const int nHLpts,
                          const double scaleFac )
  : ReactionModel( gas,                 // Cantera_CXX::IdealGasMix object
                   indep_var_names(),   // names of independent variables
                   order,               // order for interpolation
                   "Equilibruim" ),     // name of the model
    nFpts_( nFpts ),
    nHLpts_( nHLpts ),
    scaleFac_( scaleFac ),

    Tfuel_( 300.0 ),
    Toxid_( 300.0 ),
    pressure_( 101325.0 ),   // Pressure in Pa

    mixfr_( new MixtureFraction( gas, y_oxid, y_fuel, haveMassFrac ) ),
    adiabaticEq_( NULL )
{
  // consistency checking.
  assert( nSpec_ == (int)y_oxid.size() );
  assert( nSpec_ == (int)y_fuel.size() );

  // set up the mixture fraction mesh
  set_f_mesh();

  // make sure that other things are ready.
  assert( mixfr_->is_ready() );
  assert( gasProps_.ready() );

}
//--------------------------------------------------------------------
EquilRxnMdl::~EquilRxnMdl()
{
  if( NULL != mixfr_ ) delete   mixfr_;
  if( NULL != adiabaticEq_ ) delete adiabaticEq_;
}
//--------------------------------------------------------------------
const vector<string> &
EquilRxnMdl::indep_var_names()
{
  static vector<string> names(2);
  names[0] = "MixtureFraction";
  names[1] = "HeatLoss";
  return names;
}
//--------------------------------------------------------------------
void
EquilRxnMdl::implement()
{
  using namespace std;

  // set enthalpy of the fuel and oxidizer streams
  gasProps_.setState_TPY( Tfuel_, pressure_, mixfr_->fuel_massfr() );
  const double hFuel = gasProps_.enthalpy_mass();

  gasProps_.setState_TPY( Toxid_, pressure_, mixfr_->oxid_massfr() );
  const double hOxid = gasProps_.enthalpy_mass();

  // build the Adiabatic case
  adiabaticEq_ =
    new AdiabaticEquilRxnMdl( gasProps_,
                              mixfr_->oxid_massfr_vec(),
                              mixfr_->fuel_massfr_vec(),
                              true,
                              interpOrder_,
                              nFpts_,
                              scaleFac_ );
  adiabaticEq_->no_output(); // don't request an output database.

  adiabaticEq_->set_fuel_temp( Tfuel_ );
  adiabaticEq_->set_oxid_temp( Toxid_ );
  adiabaticEq_->set_pressure( pressure_ );
  adiabaticEq_->implement();

  fpts_.clear();
  fpts_   = adiabaticEq_->get_f_mesh();
  nFpts_  = fpts_.size();

  // create some workspace.
  vector<double> ysUnreacted(nSpec_), ys(nSpec_);
  vector<double> ha_i(nSpec_), ho_i(nSpec_);

  //
  // Loop over mixture fractions
  //
  for( int imf=0; imf<nFpts_; imf++ ){

    const double f = fpts_[imf];

    // set the mixture composition for this mixture fraction
    mixfr_->mixfrac_to_species( f, ysUnreacted );

    // set the mixture enthalpy with no heat loss (adiabatic)
    const double ha = f*hFuel + (1.0-f)*hOxid;

    //----------------------------------------------------------------
    //  compute the sensible enthalpy at adiabatic conditions.  This
    //  sets an upper limit on the amount of energy we can remove
    //  from the system, assuming that the surrounding temperature
    //  is given by the unreacted mixture temperature.
    //
    //  In the future, we may want to relax this to allow an arbitrary
    //  temperature for the surroundings.
    //----------------------------------------------------------------

    // set the species specific enthalpies (mass units) for the adiabatic,
    // unreacted mixture at the "surroundings" temperature.
    const double Tunreacted = f*Tfuel_ + (1.0-f)*Toxid_;
    gasProps_.setState_TPY( Tunreacted, pressure_, &(ysUnreacted[0]) );
    gasProps_.setState_HP( ha, pressure_ );

    set_species_enthalpies( &(ysUnreacted[0]), gasProps_.temperature(), &(ho_i[0]) );

    // get the species specific enthalpies (mass units) of the
    // reacted mixture at adiabatic conditions
    const EqStateVars & adState = adiabaticEq_->get_state_vars(imf);
    set_species_enthalpies( &(adState.ys[0]), adState.temp, &(ha_i[0]) );

    // compute the sensible enthalpy at adiabatic conditions
    //   h = \sum_i ( hc_i + hs_i )
    double hs=0;
    for( int isp=0; isp<nSpec_; ++isp ){
      // calculate species contribution to sensible enthalpy, in specific mass units.
      hs += adState.ys[isp] * ( ha_i[isp] - ho_i[isp] );
    }

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

    double Tguess = Tunreacted;
    ys = ysUnreacted;

    //
    // LOOP ON HEAT LOSS:
    //
    // gamma ranges from [ -1.0, 1.0 ]
    for( int ihl=0; ihl<nHLpts_; ++ihl ){

      const double gamma = 1.0 - double(2*ihl) / double(nHLpts_-1);
      const double h = ha - gamma*hs;

      // set the composition
      const double fac = gamma > 0.0 ? 1.0 : gamma+1.0;
      for( int isp=0; isp<nSpec_; ++isp ){
        ys[isp] = std::max( 0.0, fac*ysUnreacted[isp] + (1.0-fac)*adState.ys[isp] );
      }

      // compute the equilibrium composition at constant pressure and enthalpy
      bool converged = true;
      if( (f > 0.0) && (f < 1.0) ) converged = do_equil( ys, h, Tguess );

      // update the guess for the temperature
      Tguess = gasProps_.temperature();

      // if this point converged, then add it to the table
      if( converged ){
        gasProps_.getMassFractions( &(ys[0]) );
        for( int isp=0; isp<nSpec_; ++isp ){
          ys[isp] = std::max( 0.0, std::min(1.0, ys[isp]) );
        }
        EqStateVars state( gasProps_.temperature(), pressure_, ys );
        fullTable_[gamma][f] = state;
      }
      else{
        cout << " NOTE: point [f,gamma]=["<<f<<","<<gamma<<"] did not converge." << endl;
      }

    } // loop on heat loss

//     cout << endl << "-------------------------------------------------" << endl << endl;
    cout << "f=" << f << endl;

  } // loop on mixture fraction

  // generate the state table and dump it to disk
  output_table();
}
//--------------------------------------------------------------------
void
EquilRxnMdl::output_table()
{
  /*
   * We require a structured table for the TableBuilder.  We may not have a
   * structured table because some points may not converge.  Thus, we will
   * interpolate each gamma entry to a new mesh.  In the case where all points
   * converge, then we will be using the original mesh, and the interpolation
   * should recover the exact values anyway.
   */

  // output the chemical and adiabatic enthalpies to the table.
  gasProps_.setState_TPY( Tfuel_, pressure_, mixfr_->fuel_massfr() );
  const double hFuel = gasProps_.enthalpy_mass();
  gasProps_.setState_TPY( Toxid_, pressure_, mixfr_->oxid_massfr() );
  const double hOxid = gasProps_.enthalpy_mass();

  // for the equilibrium model with heat loss, output the adiabatic
  // enthalpy and the sensible enthalpy.  These are not normally
  // available.
  tableBuilder_.request_output( new AdEnthEvaluator( *mixfr_, hFuel, hOxid ) );
  tableBuilder_.request_output( new EquilRxnMdl::SensEnthEvaluator( gasProps_, *adiabaticEq_, *mixfr_, hFuel, hOxid ) );
  tableBuilder_.request_output( StateVarEvaluator::ENTHALPY    );
  tableBuilder_.request_output( StateVarEvaluator::TEMPERATURE );


  vector<Interp1D*> Tinterp, Pinterp;
  vector< vector<Interp1D*> >Yinterp;
  vector<double> gampts;

  map< double, map<double,EqStateVars> >::const_iterator igam;
  for( igam=fullTable_.begin(); igam!=fullTable_.end(); ++igam ){
    // interpolate each entry in the gamma dimension
    const map<double,EqStateVars> & fmap = igam->second;
    vector<double> f, t, p;
    vector< vector<double> > ys(nSpec_);
    map<double,EqStateVars>::const_iterator ientry;
    for( ientry=fmap.begin(); ientry!=fmap.end(); ++ientry ){
      f.push_back(ientry->first);
      t.push_back(ientry->second.temp);
      p.push_back(ientry->second.press);
      for( int i=0; i<nSpec_; ++i ){
        ys[i].push_back(ientry->second.ys[i]);
      }
    }
    Tinterp.push_back( new Interp1D( interpOrder_, f, t ) );
    Pinterp.push_back( new Interp1D( interpOrder_, f, p ) );
    vector<Interp1D*> ysp(nSpec_);
    vector<Interp1D*>::iterator iysp = ysp.begin();
    vector< vector<double> >::const_iterator iys;
    for( iys=ys.begin(); iys!=ys.end(); ++iys, ++iysp ){
      *iysp = new Interp1D( interpOrder_, f, *iys );
    }
    Yinterp.push_back( ysp );

    gampts.push_back( igam->first );
  }

  // wipe out the internally-held table, since we are now done with it.
  fullTable_.clear();

  // set the mesh for the output table.
  tableBuilder_.set_mesh( 0, fpts_ );
  tableBuilder_.set_mesh( 1, gampts );

  // set the file prefix for the output table
  tableBuilder_.set_filename( "Equilibrium" );

  // add each entry to the table - interpolating as necessary
  vector<double> ys( nSpec_ );
  vector<double> indepVars(2);
  vector<double>::const_iterator ifpt, igamma;
  vector<Interp1D*>::const_iterator
    its=Tinterp.begin(),
    ips=Pinterp.begin();
  vector< vector<Interp1D*> >::const_iterator iys = Yinterp.begin();
  for( igamma=gampts.begin(); igamma!=gampts.end(); ++igamma, ++its, ++ips, ++iys ){
    indepVars[1] = *igamma;
    for( ifpt=fpts_.begin(); ifpt!=fpts_.end(); ++ifpt ){
      indepVars[0] = *ifpt;
      const double T = (*its)->value( &indepVars[0] );
      const double P = (*ips)->value( &indepVars[0] );
      for( int i=0; i<nSpec_; ++i ){
        ys[i] = std::max( 0.0, std::min(1.0,(*iys)[i]->value( &indepVars[0] )) );
      }
      tableBuilder_.insert_entry( indepVars, T, P, ys );
    }
  }

  // generate the table and write it to disk.
  tableBuilder_.generate_table();

  // clean up
  its=Tinterp.begin();
  ips=Pinterp.begin();
  iys=Yinterp.begin();
  for( ; its!=Tinterp.end(); ++its, ++ips, ++iys ){
    delete *its;
    delete *ips;
    for( int i=0; i<nSpec_; ++i )  delete (*iys)[i];
  }

}
//--------------------------------------------------------------------
void
EquilRxnMdl::set_species_enthalpies( const double* ys,
                                     const double  temp,
                                     double* h )
{
  gasProps_.setState_TPY( temp, pressure_, ys );
  gasProps_.getEnthalpy_RT( h );  // nondimensional molar enthalpies

  // scale the enthalpy to get it in mass units
  const double RT = Cantera::GasConstant * temp;
  for( int i=0; i<nSpec_; i++ ){
    h[i] *= RT / gasProps_.molecularWeight(i);
  }
}
//--------------------------------------------------------------------
void
EquilRxnMdl::set_f_mesh()
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
bool
EquilRxnMdl::do_equil( vector<double> & ys,
                       const double targetEnth,
                       const double Tguess )
{
  const double Tmax = gasProps_.maxTemp();
  const double Tmin = gasProps_.minTemp();

  double Tnew = Tguess;
  double Told = Tnew;

  gasProps_.setState_TPY( Tnew, pressure_, &(ys[0]) );
  Cantera::ChemEquil eq;
  eq.options.relTolerance = 1.0e-6;

  const double AbsTTol = 0.5;       // temperature tolerance, K
  const double RelTTol = 0.005;     // relative error tolerance
  const int    MaxIt = 50;          // maximum number of iterations

  double dT = 0.5*Tguess;
  double dTold=dT;
  int i=0;
  for( i=0; i<MaxIt; i++ ){
    try{

      gasProps_.setState_TP( Tnew, pressure_ );

      eq.equilibrate( gasProps_, "TP" );

      dTold = dT;
      dT = 1.0/gasProps_.cp_mass() * ( targetEnth - gasProps_.enthalpy_mass() );

      // check for convergence on both absolute and relative tolerances
      bool converged = false;
      if( std::abs(dT) < AbsTTol ){
        if( std::abs(dT)/Tnew < RelTTol )
          converged = true;
      }

      const double sign = ( dT==std::abs(dT) ) ? 1.0 : -1.0;
      // make sure that dT is decreasing...
      if( std::abs(dTold) < std::abs(dT) ) dT = 0.9*sign*std::abs(dTold);

      // deal with situation where solution bounces around dT=0.
      if( dTold/dT < 0.0 ) dT = 0.8*dT;

      Told = Tnew;
      const double urf = 1.0;
      Tnew = std::max( Tmin, std::min( Told + urf*dT, Tmax ) );

      /*
        if( Tnew>Tmax ){
        Tnew = Tmax;
        if( Told>=Tmax ){
        cout << " Exceeded Cantera high temperature limit.  Returning state at T=" << Tmax << endl;
        gasProps_.setTemperature(Tnew);
        return false;
        }
        }
        if( Tnew<Tmin ){
        Tnew = Tmin;
        if( Told<=Tmin ){
        cout << " Exceeded Cantera low temperature limit.  Returning state at T=" << Tmin << endl;
        gasProps_.setTemperature(Tnew);
        return false;
        }
        }
      */
      gasProps_.setTemperature(Tnew);

      // check for convergence on both absolute and relative tolerances
      if( converged ) return true;

    }
    catch(Cantera::CanteraError&){
      Cantera::showErrors(cout);
      return false;
    }
  }
  return false;
}

//====================================================================

EquilRxnMdl::SensEnthEvaluator::
SensEnthEvaluator( Cantera::ThermoPhase& thermo,
                   const AdiabaticEquilRxnMdl& adEq,
                   MixtureFraction& mixfrac,
                   const double fuelEnthalpy,
                   const double oxidEnthalpy )
  : StateVarEvaluator( SENS_ENTH, "SensibleEnthalpy" ),
    thermo_ ( thermo ),
    adEq_( adEq ),
    mixfrac_( mixfrac ),
    hfuel_( fuelEnthalpy ),
    hoxid_( oxidEnthalpy ),
    spwork_( thermo.nSpecies(), 0.0 ),
    ho_    ( thermo.nSpecies(), 0.0 ),
    hs_    ( thermo.nSpecies(), 0.0 )
{}
//--------------------------------------------------------------------
double
EquilRxnMdl::SensEnthEvaluator::
evaluate( const double& t,
          const double& p,
          const std::vector<double>& ys )
{
  // the sensible enthalpy is only a function of composition (mixture
  // fraction) and the pure stream enthalpies.

  // set the mixture fraction and the unreacted composition
  double f=0.0;
  mixfrac_.species_to_mixfrac( ys, f );
  f = std::max( 0.0, std::min(1.0,f) );
  mixfrac_.mixfrac_to_species( f, spwork_ );

  // adiabatic enthalpy of the mixture
  const double ha = f * hfuel_ + (1.0-f)*hoxid_;

  // calculate the species enthalpies at the unreacted state (reference state)
  thermo_.setState_TPY( 300, p, &spwork_[0] );
  thermo_.setState_HP( ha, p );
  thermo_.getPartialMolarEnthalpies( &ho_[0] );

  set_adiab_reacted_state( f );  // sets thermo_ to the adiabatic equilibrium state
  thermo_.getMassFractions( &spwork_[0] );
  thermo_.getPartialMolarEnthalpies( &hs_[0] );

  // calculate the mixture enthalpy at adiabatic conditions
  double hsens = 0.0;
  for( size_t i=0; i<thermo_.nSpecies(); ++i ){
    hsens += spwork_[i] * ( hs_[i] - ho_[i] ) / thermo_.molecularWeight(i);
  }
#ifdef CGS_UNITS
  return hsens * 1.0e4;
#else
  return hsens;
#endif
}
//--------------------------------------------------------------------
void
EquilRxnMdl::SensEnthEvaluator::
set_adiab_reacted_state( const double f )
{
  const std::vector<double>& fpts = adEq_.get_f_mesh();
  typedef std::vector<double>::const_iterator FIter;

  // locate this mixture fraction in the adiabatic equilibrium object
  int ix = 0;
  if( f==0 ) {}
  else if( f==1.0 )
    ix = fpts.size();
  else{
    FIter iflo = fpts.begin();
    FIter ifhi = iflo+1;
    for( int i=0; ifhi!=fpts.end(); ++iflo, ++ifhi, ++i ){
      if( f<*ifhi ){
        if( *ifhi-f > f-*iflo )
          ix = i;
        else
          ix = i+1;
        break;
      }
    }

    const EqStateVars& sv = adEq_.get_state_vars( ix );
    thermo_.setState_TPY( sv.temp, sv.press, &(sv.ys[0]) );
  }
}
//--------------------------------------------------------------------
