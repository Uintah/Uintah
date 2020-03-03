/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Models/MultiMatlExchange/ExchangeCoefficients.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace ExchangeModels;
using namespace std;

extern DebugStream dbgExch;
//______________________________________________________________________
//
ExchangeCoefficients::ExchangeCoefficients()
{
  d_heatExchCoeffModel = "constant"; // default
  d_convective = false;
  d_K_mom_V.clear();
  d_K_heat_V.clear();
}

ExchangeCoefficients::~ExchangeCoefficients()
{
}

//______________________________________________________________________
//
void ExchangeCoefficients::problemSetup(const ProblemSpecP  & matl_ps,
                                        const int numMatls,
                                        ProblemSpecP        & exch_ps )
{
  d_numMatls = numMatls;
  if (d_numMatls == 1 ){
    return;
  }

  //__________________________________
  // Pull out the constant Coeff exchange coefficients
  exch_ps = matl_ps->findBlock("exchange_properties");
  if (!exch_ps){
    throw ProblemSetupException("Cannot find exchange_properties tag", __FILE__, __LINE__);
  }

  //__________________________________
  // variable coefficient models
  exch_ps->get("heatExchangeCoeff",d_heatExchCoeffModel);

  if(d_heatExchCoeffModel !="constant" &&
     d_heatExchCoeffModel !="variable" &&
     d_heatExchCoeffModel !="Variable"){
     ostringstream warn;
      warn<<"ERROR\n Heat exchange coefficient model (" << d_heatExchCoeffModel
          <<") does not exist.\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  //  constant coefficient model
  ProblemSpecP exch_co_ps = exch_ps->findBlock("exchange_coefficients");

  // momentum
  exch_co_ps->require("momentum",d_K_mom_V);

  // Bullet Proofing
  for (int i = 0; i<(int)d_K_mom_V.size(); i++) {
    dbgExch << "K_mom = " << d_K_mom_V[i] << endl;
    
    if( d_K_mom_V[i] < 0.0 || d_K_mom_V[i] > 1e20 ) {
      ostringstream warn;
      warn<<"ERROR\n Momentum exchange coef. is either too big or negative\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }

  // heat
  if(d_heatExchCoeffModel == "constant"){
    exch_co_ps->require("heat", d_K_heat_V);

    // Bullet Proofing
    for (int i = 0; i<(int)d_K_heat_V.size(); i++) {
      dbgExch << "K_heat = " << d_K_heat_V[i] << endl;
      if( d_K_heat_V[i] < 0.0 || d_K_heat_V[i] > 1e15 ) {
        ostringstream warn;
        warn<<"ERROR\n Heat exchange coef. is either too big or negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
  }

  if (d_heatExchCoeffModel != "constant"){
    proc0cout << "------------------------------Using Variable heat exchange coefficients"<< endl;
  }
  
  //__________________________________
  //  convective heat transfer
  d_convective = false;
  exch_ps->get("do_convective_heat_transfer", d_convective);
  if(d_convective){
    exch_ps->require("convective_fluid",d_conv_fluid_matlindex);
    exch_ps->require("convective_solid",d_conv_solid_matlindex);
  }
}

//______________________________________________________________________
//
void ExchangeCoefficients::outputProblemSpec(ProblemSpecP& matl_ps,
                                             ProblemSpecP& exch_prop_ps)
{
  if (d_numMatls == 1 ){
    return;
  }
  // <exchange_properties>
  exch_prop_ps = matl_ps->appendChild("exchange_properties");
  exch_prop_ps->appendElement("heatExchangeCoeff",d_heatExchCoeffModel);

  // <exchange_coefficients>
  ProblemSpecP exch_coeff_ps = exch_prop_ps->appendChild("exchange_coefficients");
  exch_coeff_ps->appendElement("momentum", d_K_mom_V);
  exch_coeff_ps->appendElement("heat",     d_K_heat_V);

  if (d_convective) {
    exch_coeff_ps->appendElement("do_convective_heat_transfer",d_convective);
    exch_coeff_ps->appendElement("convective_fluid",           d_conv_fluid_matlindex);
    exch_coeff_ps->appendElement("convective_solid",           d_conv_solid_matlindex);
  }
}

//______________________________________________________________________
//
void ExchangeCoefficients::getConstantExchangeCoeff( FastMatrix& K,
                                                     FastMatrix& H  )
{

  // The vector of exchange coefficients only contains the upper triagonal
  // matrix

  // Check if the # of coefficients = # of upper triangular terms needed
  int num_coeff = ( d_numMatls * d_numMatls - d_numMatls)/2;

  vector<double>::iterator it_m = d_K_mom_V.begin();
  vector<double>::iterator it_h = d_K_heat_V.begin();

  //__________________________________
  // bulletproofing
  bool test = false;
  string desc;
  if (num_coeff != (int)d_K_mom_V.size()) {
    test = true;
    desc = "momentum";
  }

  if (num_coeff !=(int)d_K_heat_V.size() && d_heatExchCoeffModel == "constant") {
    test = true;
    desc = desc + " energy";
  }

  if(test) {
    ostringstream warn;
    warn << "\nThe number of exchange coefficients (" << desc << ") is incorrect.\n";
    warn << "Here is the correct specification:\n";
    for (int i = 0; i < d_numMatls; i++ ){
      for (int j = i+1; j < d_numMatls; j++){
        warn << i << "->" << j << ",\t";
      }
      warn << "\n";
      for (int k = 0; k <= i; k++ ){
        warn << "\t";
      }
    }
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  // Fill in the upper triangular matrix
  // momentum
  for (int i = 0; i < d_numMatls; i++ )  {
    K(i,i) = 0.0;
    for (int j = i + 1; j < d_numMatls; j++) {
      K(i,j) = K(j,i) = *it_m++;
    }
  }

  // heat
  if( d_heatExchCoeffModel == "constant" ) {
    for (int i = 0; i < d_numMatls; i++ )  {
      H(i,i) = 0.0;
      for (int j = i + 1; j < d_numMatls; j++) {
        H(i,j) = H(j,i) = *it_h++;
      }
    }
  }
}

//______________________________________________________________________
//
void ExchangeCoefficients::getVariableExchangeCoeff( FastMatrix& ,
                                                     FastMatrix& H,
                                                     IntVector & c,
                                                     std::vector<constCCVariable<double> >& mass_L  )
{

  //__________________________________
  // Momentum  (do nothing for now)

  //__________________________________
  // Heat coefficient
  for (int m = 0; m < d_numMatls; m++ )  {
    H(m,m) = 0.0;
    for (int n = m + 1; n < d_numMatls; n++) {
      double massRatioSqr = pow(mass_L[n][c]/mass_L[m][c], 2.0);

      // 1e5  is the lower limit clamp
      // 1e12 is the upper limit clamp
      if (massRatioSqr < 1e-12){
        H(n,m) = H(m,n) = 1e12;
      }
      else if (massRatioSqr >= 1e-12 && massRatioSqr < 1e-5){
        H(n,m) = H(m,n) = 1./massRatioSqr;
      }
      else if (massRatioSqr >= 1e-5 && massRatioSqr < 1e5){
        H(n,m) = H(m,n) = 1e5;
      }
      else if (massRatioSqr >= 1e5 && massRatioSqr < 1e12){
        H(n,m) = H(m,n) = massRatioSqr;
      }
      else if (massRatioSqr >= 1e12){
        H(n,m) = H(m,n) = 1e12;
      }

    }
  }
}


//______________________________________________________________________
//  utilities
bool ExchangeCoefficients::convective()
{
  return d_convective;
}

int ExchangeCoefficients::conv_fluid_matlindex()
{
  return d_conv_fluid_matlindex;
}

int ExchangeCoefficients::conv_solid_matlindex()
{
  return d_conv_solid_matlindex;
}
