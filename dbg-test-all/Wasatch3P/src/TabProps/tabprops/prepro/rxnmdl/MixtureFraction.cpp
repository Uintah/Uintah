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

#include "MixtureFraction.h"
#include "cantera/Cantera.h"
#include "cantera/kernel/Constituents.h"

#include <cassert>
#include <string>
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

using std::vector;
using std::string;

//====================================================================

static void mole_to_mass( const vector<double> & molWt,
			  vector<double> & moleFrac,
			  vector<double> & massFrac )
{
  const int ns = (int)moleFrac.size();

  assert( ns == (int)massFrac.size() );
  assert( ns == (int)molWt.size()    );

  double mixMW = 0.0;
  int n;
  for( n=0; n<ns; n++ )
    mixMW += moleFrac[n]*molWt[n];
  for( n=0; n<ns; n++ )
    massFrac[n] = moleFrac[n] * molWt[n] / mixMW;
}
//--------------------------------------------------------------------
static void mass_to_mole( const vector<double> & molWt,
			  vector<double> & massFrac,
			  vector<double> & moleFrac )
{
  const int ns = (int)moleFrac.size();

  assert( ns == (int)massFrac.size() );
  assert( ns == (int)molWt.size()    );

  double mixMW = 0.0;
  int n;
  for( n=0; n<ns; n++ ){
    mixMW += massFrac[n]/molWt[n];
  }
  mixMW = 1.0 / mixMW;
  for( n=0; n<ns; n++ ){
    moleFrac[n] = mixMW * massFrac[n] / molWt[n];
  }
}

//====================================================================
//====================================================================
//====================================================================

MixtureFraction::MixtureFraction( Cantera::Constituents & props,
				  const vector<double> & oxidFrac,
				  const vector<double> & fuelFrac,
				  const bool inputMassFrac )
  : specProps_( props ),
    nelem_( props.nElements() ),
    nspec_( props.nSpecies() ),

    stoichMixfrac( -1.0 ),

    beta0_( 0.0 ), beta1_( 0.0 ),

    ready_( false ),

    gamma_( nelem_ ),
    elemMassFr_( nelem_ ),
    fuelMassFrac_( fuelFrac ),
    oxidMassFrac_( oxidFrac )
{
  if( nelem_ <= 0 || nspec_ <= 0 ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << "MixtureFraction class must have valid Cantera object." << std::endl;
    throw std::runtime_error( msg.str() );
  }
  initialize( oxidFrac, fuelFrac, inputMassFrac );
}

//--------------------------------------------------------------------

MixtureFraction::MixtureFraction( Cantera::Constituents & props )
  : specProps_( props ),
    nelem_( props.nElements() ),
    nspec_( props.nSpecies() ),
    ready_( false ),
    gamma_( nelem_ )
{
  elemMassFr_.assign(nelem_,0.0);
  gamma_.assign(nelem_,0.0);
  fuelMassFrac_.assign(nspec_,0.0);
  oxidMassFrac_.assign(nspec_,0.0);
}

//--------------------------------------------------------------------

MixtureFraction::~MixtureFraction()
{}

//--------------------------------------------------------------------

void
MixtureFraction::initialize( const vector<double> & oxid,
			     const vector<double> & fuel,
			     const bool massFrac )
{

  // ensure that mass fractions sum to unity
  double oSum=0, fSum=0;
  for( int i=0; i<nspec_; i++ ){
    oSum += oxid[i];
    fSum += fuel[i];
  }
  for( int i=0; i<nspec_; i++ ){
    oxidMassFrac_[i] = oxid[i] / oSum;
    fuelMassFrac_[i] = fuel[i] / fSum;
  }

  // copy species and element MW into local storage.  Unfortunate to have to copy,
  // but required since Cantera won't use std::vector<>...

  specMolWt_.resize(nspec_);
  for( int i=0; i<nspec_; i++ )
    specMolWt_[i] = specProps_.molecularWeight(i);

  elemWt_.resize(nelem_);
  for( int i=0; i<nelem_; i++ )
    elemWt_[i] = specProps_.atomicWeight(i);

  // convert to mass fractions if we got mole fractions.
  if( !massFrac ){
    mole_to_mass( specMolWt_, oxidMassFrac_, oxidMassFrac_ );
    mole_to_mass( specMolWt_, fuelMassFrac_, fuelMassFrac_ );
  }

  // set the elemental weighting factors
  set_gammas();

  // set pure stream coupling functions
  beta0_ = compute_beta( oxidMassFrac_ );
  beta1_ = compute_beta( fuelMassFrac_ );

  assert( beta1_ != beta0_ );

  // set the stoichiometric mixture fraction
  stoichMixfrac = compute_stoich_mixfrac();

  if( (beta0_ != beta1_)     &&
      (stoichMixfrac >= 0.0-std::numeric_limits<double>::epsilon()) &&
      (stoichMixfrac <= 1.0+std::numeric_limits<double>::epsilon()) )
    ready_ = true;
  else{
    ready_ = false;

    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << "MixtureFraction class was not properly initialized!" << std::endl
        << "Stoichiometric mixture fraction is out of bounds" << std::endl;
    throw std::runtime_error( msg.str() );
  }

  // set stoichiometric coefficients for complete reaction
  set_stoichiometry();
}

//--------------------------------------------------------------------

void
MixtureFraction::species_to_mixfrac( const vector<double> & species,
				     double & mixFrac )
{

  assert( ready_ );
  /*
   *		   beta  - beta0	beta0 evaluated in air  stream
   *	mixfrac = ---------------	beta1 evaluated in fuel stream
   *		   beta1 - beta0
   *
   *	require pure stream compositions
   *    coupling functions (gammas)
   */
  const double beta = compute_beta( species );
  mixFrac = ( beta - beta0_ ) / ( beta1_ - beta0_ );
}

//--------------------------------------------------------------------

void
MixtureFraction::mixfrac_to_species( const double mixFrac,
				     vector<double> & species ) const
{
  assert( ready_ );
  assert( mixFrac >=    -std::numeric_limits<double>::epsilon() &&
          mixFrac <= 1.0+std::numeric_limits<double>::epsilon() );
  assert( nspec_ == (int)species.size() );

  vector<double>::iterator isp;
  vector<double>::const_iterator ifuel, ioxid;
  ifuel = fuelMassFrac_.begin();
  ioxid = oxidMassFrac_.begin();
  for( isp=species.begin(); isp!=species.end(); ++isp, ++ifuel, ++ioxid ){
    *isp = (mixFrac)*(*ifuel) + (1.0-mixFrac)*(*ioxid);
//    assert( *isp >= 0.0 && *isp <= 1.0 );
  }
}

//--------------------------------------------------------------------

double
MixtureFraction::compute_stoich_mixfrac() const
{
  return -beta0_/(beta1_-beta0_);
}

//--------------------------------------------------------------------

void
MixtureFraction::set_gammas()
{
  // set the element name vector
  const vector<string> elemName = specProps_.elementNames();

  // Bilger's mixture fraction
  for ( int i=0; i<nelem_; i++ ){
    const double elemWt = specProps_.atomicWeight(i);

    if( elemName[i] == "C" )
      gamma_[i] = 2.0 / elemWt;

    else if( elemName[i] == "H" )
      gamma_[i] = 1.0 / (2.0*elemWt);

    else if( elemName[i] == "O" )
      gamma_[i] = -1.0 / elemWt;

    else
      gamma_[i] = 0.0;
  }
}

//--------------------------------------------------------------------

double
MixtureFraction::compute_beta( const vector<double> & massFrac )
{
  compute_elem_mass_frac( massFrac, elemMassFr_ );
  double beta = 0.0;
  for ( int i=0; i<nelem_; i++ ){
    beta += gamma_[i] * elemMassFr_[i];
  }
  return beta;
}

//--------------------------------------------------------------------

void
MixtureFraction::compute_elem_mass_frac( const vector<double> & spec,
					 vector<double> & elem ) const
{
  assert( nspec_ == (int)spec.size() );
  assert( nelem_ == (int)elem.size() );

  for( size_t ielm=0; ielm<nelem_; ielm++ ){
    elem[ielm]=0.0;
    const double & eWt = elemWt_[ielm];
    for( size_t isp=0; isp<nspec_; isp++ ){
      elem[ielm] += specProps_.nAtoms(isp,ielm) * eWt * spec[isp] / specMolWt_[isp];
    }
    assert( elem[ielm] >= 0.0 );
  }
}

//--------------------------------------------------------------------

double
MixtureFraction::mixfrac_to_equiv_ratio( const double mixFrac ) const
{
  assert( ready_ );
  assert( mixFrac < 1.0 && mixFrac >= 0.0 );
  return mixFrac*(1.0-stoichMixfrac) / (stoichMixfrac*(1.0-mixFrac));
}

//--------------------------------------------------------------------

double
MixtureFraction::equiv_ratio_to_mixfrac( const double eqrat ) const
{
  assert( ready_ );
  assert( eqrat >= 0.0 );
  return (stoichMixfrac*eqrat) / (1.0+stoichMixfrac*(eqrat-1.0));
}

//--------------------------------------------------------------------

void
MixtureFraction::estimate_product_comp( const double mixFrac,
					vector<double> & massFrac,
					const bool calcMassFrac )
{
  if( mixFrac > stoichMixfrac ){  // fuel in excess
    const double fac = (mixFrac - stoichMixfrac) / (1.0-stoichMixfrac);
    for( size_t i=0; i<nspec_; i++ ){
      massFrac[i] = stoichProdMassFrac_[i]*(1.0-fac) + fuelMassFrac_[i]*fac;
    }
  }
  else if( mixFrac < stoichMixfrac ){ // oxidizer in excess
    const double fac = mixFrac / stoichMixfrac;
    for( size_t i=0; i<nspec_; i++ ){
      massFrac[i] = oxidMassFrac_[i]*(1.0-fac) + stoichProdMassFrac_[i]*fac;
    }
  }
  else{  // stoichiometric
    for( size_t i=0; i<nspec_; i++ )
      massFrac[i] = stoichProdMassFrac_[i];
  }

  // convert to mole fractions if requested
  if( !calcMassFrac )  mass_to_mole( specMolWt_, massFrac, massFrac );

  double invYsum = 1.0/accumulate( massFrac.begin(), massFrac.end(), 0.0 );
  for( size_t i=0; i<nspec_; i++ ) massFrac[i] *= invYsum;
  assert( invYsum < 1.001  && invYsum > 0.999 );
}
//--------------------------------------------------------------------

void
MixtureFraction::set_stoichiometry()
{
  // set stoichiometric coefficients assuming that the products are
  //    CO2  H2O  N2  AR
  // Reactants have negative coefficients while Products have positive coefficients

  std::vector<double> phi_reactant;   phi_reactant.assign( nspec_, 0.0 );
  std::vector<double> phi_product ;   phi_product.assign(  nspec_, 0.0 );

  vector<double> elemMoles_rx( nelem_, 0.0 );

  //
  // set the reactant composition (mole fractions) at stoichiometric conditions
  // this is also the stoichiometric coefficients for these species.
  //
  mixfrac_to_species( stoichMixfrac, phi_reactant );
  mass_to_mole( specMolWt_, phi_reactant, phi_reactant );

  //
  // get the elemental mole fractions for the reactants
  // this gives the stoichiometry for the reactants
  //
  for( size_t ielm=0; ielm<nelem_; ielm++ ){
    elemMoles_rx[ielm] = 0.0;
    for( size_t isp=0; isp<nspec_; isp++ ){
      elemMoles_rx[ielm] += specProps_.nAtoms(isp,ielm) * phi_reactant[isp];
    }
  }

  // now we can do the elemental balances by solving a system of equations:

  // Carbon balance to get phi[iCO2], assuming CO2 is the only product containing C
  const int iCO2 = specProps_.speciesIndex("CO2");
  const int iC   = specProps_.elementIndex("C");
  if( iCO2 >= 0 )
    phi_product[iCO2] = elemMoles_rx[iC] + phi_reactant[iCO2];

  // Hydrogen balance to get phi[iH2O], assuming H2O is the only product containing H
  const int iH2O = specProps_.speciesIndex("H2O");
  const int iH   = specProps_.elementIndex("H");
  if( iH2O >= 0 )  phi_product[iH2O] = 0.5*elemMoles_rx[iH] + phi_reactant[iH2O];

  // N2 balance
  const int iN2 = specProps_.speciesIndex("N2");
  const int iN  = specProps_.elementIndex("N");
  if( iN2 >= 0 ) phi_product[iN2] = 0.5*elemMoles_rx[iN];

  // Sulfur balanceot get phi[iSO2], assuming SO2 is the only product containing S.
  const int iSspecies = specProps_.speciesIndex("S");
  const int iSelem    = specProps_.elementIndex("S");
  const int iSO2      = specProps_.speciesIndex("SO2");
  if( iSO2 >= 0 ) phi_product[iSO2] = phi_reactant[iSspecies] + elemMoles_rx[iSelem];

  // deal with other elements
  const vector<string> & elementNames = specProps_.elementNames();
  for( int ielm=0; ielm<nelem_; ielm++ ){
    const string & nam = elementNames[ielm];
    if( nam != "C" && nam != "H" && nam != "O" && nam != "N" && nam != "S" ){
      // see what species this element is present in
      int n = 0;
      int ispec=-1;
      for( int isp=0; isp<nspec_; isp++ ){
	if( specProps_.nAtoms(isp,ielm) > 0 ){  ++n;  ispec=isp; }
      }
      // don't know what to do if it is in more than one species.
      assert( n <= 1 );
      if( n == 1 ){
	assert( ispec >= 0 );
	phi_product[ispec] = elemMoles_rx[ielm];
      }
    }
  }

  // normalize phi_product so that we have the product mole fractions
  // at stoichiometric conditions for complete reaction.
  stoichProdMassFrac_ = phi_product;
  const double invSum = 1.0 / accumulate( stoichProdMassFrac_.begin(), stoichProdMassFrac_.end(), 0.0 );
  for( vector<double>::iterator ispmf = stoichProdMassFrac_.begin();
       ispmf != stoichProdMassFrac_.end();
       ispmf++ )
  {
    (*ispmf) *= invSum;
  }

}
//--------------------------------------------------------------------




//==============================================================================
//------------------------------------------------------------------------------
//
//          Utilities to test the mixture fraction class.
//
//------------------------------------------------------------------------------
//==============================================================================



#include <cantera/Cantera.h>
#include <iostream>
using std::cout;
using std::endl;

bool test_mixfrac()
{
  cout << "  mixture fraction object...";
  try{

    Cantera::Constituents gas;

    // hard-wire some stuff for testing.  DO NOT CHANGE.
    gas.addElement("C",12.0112);
    gas.addElement("H",1.00797);
    gas.addElement("O",15.9994);
    gas.addElement("N",14.0067);
    gas.freezeElements();

    double ch4[] = {1,4,0,0};  gas.addSpecies("CH4", ch4 );
    double o2 [] = {0,0,2,0};  gas.addSpecies("O2",  o2  );
    double n2 [] = {0,0,0,2};  gas.addSpecies("N2",  n2  );
    double h2 [] = {0,2,0,0};  gas.addSpecies("H2",  h2  );
    double co2[] = {1,0,2,0};  gas.addSpecies("CO2", co2 );
    double h2o[] = {0,2,1,0};  gas.addSpecies("H2O", h2o );
    gas.freezeSpecies();

    // now we have set up the cantera things that are required.
    // so initialize a mixture of gases...
    const int nspec = gas.nSpecies();
    vector<double> oxid(nspec);
    vector<double> fuel(nspec);


    // mole fractions
    oxid[ gas.speciesIndex("O2") ] = 0.21;

    // mole fractions
    fuel[ gas.speciesIndex("CH4") ] = 0.221;

    MixtureFraction f0( gas, oxid, fuel, false );


    // mole fractions
    oxid[ gas.speciesIndex("O2") ] = 0.21;
    oxid[ gas.speciesIndex("N2") ] = 0.79;

    // mole fractions
    fuel[ gas.speciesIndex("CH4") ] = 0.221;
    fuel[ gas.speciesIndex("H2")  ] = 0.332;
    fuel[ gas.speciesIndex("N2")  ] = 0.447;

    MixtureFraction f( gas, oxid, fuel, false );

    //-----------------------------------------
    // don't change this!
    const double fst_true = 0.166925;

    bool okay = true;
    const double fst = f.stoich_mixfrac();
    if( fabs( fst - fst_true ) > 2.0e-6 )
      okay = false;

    double eqrat = f.mixfrac_to_equiv_ratio(fst);
    if( (1.0-eqrat) > 2.0e-6 )
      okay = false;

    double fst2 = f.equiv_ratio_to_mixfrac(eqrat);
    if( fabs(fst2-fst_true) > 2.0e-6 )
      okay = false;
    //-----------------------------------------

    //-----------------------------------------
    // check calculation of product composition
    // should work independently of changes above.
    vector<double> prodComp(nspec);
    double fprod;
    f.estimate_product_comp( 0.2, prodComp, true );
    f.species_to_mixfrac( prodComp, fprod );
    if( fabs(fprod-0.2) > 1.0e-6 )
      okay = false;

    f.estimate_product_comp( 0.6, prodComp, true );
    f.species_to_mixfrac( prodComp, fprod );
    if( fabs(fprod-0.6) > 1.0e-6 )
      okay = false;
    //-----------------------------------------

    if( okay )  cout << "PASS" << endl;
    else        cout << "FAIL!" << endl;

    return okay;

  }
  catch (Cantera::CanteraError&) {
    Cantera::showErrors(cout);
    return false;
  }
  // should not get here.
  return false;
}

bool perform_mixfrac_tests()
{
  bool okay = true;
  okay = (test_mixfrac() == true) ? okay : false;
  return okay;
}

