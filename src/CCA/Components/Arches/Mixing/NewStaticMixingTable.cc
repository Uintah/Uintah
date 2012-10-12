/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- NewStaticMixingTable.cc --------------------------------------------------

#include <CCA/Components/Arches/Mixing/NewStaticMixingTable.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/Arches.h>

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Math/MiscMath.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <fcntl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

using namespace std;
using namespace Uintah;

//****************************************************************************
// Default constructor for NewStaticMixingTable
//****************************************************************************
NewStaticMixingTable::NewStaticMixingTable( bool calcReactingScalar,
                                            bool calcEnthalpy,
                                            bool calcVariance, 
                                                                                        const ProcessorGroup* myworld) :
  MixingModel(),
  d_calcReactingScalar(calcReactingScalar),
  d_calcEnthalpy(calcEnthalpy),
  d_calcVariance(calcVariance),
  d_myworld(myworld)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
NewStaticMixingTable::~NewStaticMixingTable()
{
}

//****************************************************************************
// Problem Setup for NewStaticMixingTable
//****************************************************************************
void 
NewStaticMixingTable::problemSetup(const ProblemSpecP& params)
{
  /*if (d_calcReactingScalar) 
    throw InvalidValue("Reacting scalar is unsupported parameter",
    __FILE__, __LINE__);*/
  string d_inputfile;
  ProblemSpecP db = params->findBlock("NewStaticMixingTable");

  db->getWithDefault("co_output",d_co_output,false);
  db->getWithDefault("sulfur_chem",d_sulfur_chem,false);
  db->getWithDefault("soot_precursors",d_soot_precursors,false);
  db->getWithDefault("tabulated_soot",d_tabulated_soot,false);
  db->getWithDefault("loud_heatloss_warning",d_loudHeatLossWarning, true);

  db->require("inputfile",d_inputfile);

  // Define mixing table, which includes call reaction model constructor
  readMixingTable(d_inputfile);
}

//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
NewStaticMixingTable::computeProps(const InletStream& inStream,
                                   Stream& outStream)
{
  double small=1.0e-10;
  // Extracting the independent variables from the in stream 
  double mixFrac = inStream.d_mixVars[0];
  // Scalar bounds are checked in ScalarSolver, so no need to do it here
  /*if(mixFrac > 1.0)
    mixFrac=1.0;
    else if (mixFrac < small)
    mixFrac=0.0;*/

  double mixFracVars = 0.0;
  if (d_calcVariance) {
    mixFracVars = inStream.d_mixVarVariance[0];
    // Variance bounds check and normalization is done at the model level,
    // so no need to do it here
    /*double var_limit = (mixFrac*(1.0-mixFrac));
      if(mixFracVars < small)
        mixFracVars=0.0;
      if(mixFracVars > var_limit)
        mixFracVars = var_limit;

      mixFracVars = mixFracVars/(var_limit+small);*/
  }
  // Heat loss for adiabatic case
  double current_heat_loss=0.0;
  double zero_heat_loss=0.0;
  //Absolute enthalpy
  double enthalpy=0.0;
  // Adiabatic enthalpy
  double adiab_enthalpy=0.0;
  // Sensible enthalpy
  double sensible_enthalpy=0.0;
  if(d_calcEnthalpy && !d_adiabGas_nonadiabPart){
    sensible_enthalpy=tableLookUp(mixFrac, mixFracVars, zero_heat_loss, Hs_index);
    enthalpy=inStream.d_enthalpy;
    adiab_enthalpy = d_H_fuel*mixFrac+d_H_air*(1.0-mixFrac);

    if ((inStream.d_initEnthalpy)||
        ((Abs(adiab_enthalpy-enthalpy)/Abs(adiab_enthalpy) < 1.0e-4)&&
         (mixFrac < 1.0e-4)))
      current_heat_loss = zero_heat_loss;
    else
      current_heat_loss=(adiab_enthalpy-enthalpy)/(sensible_enthalpy+small);

    if(((current_heat_loss < heatLoss[0]) && d_loudHeatLossWarning) || ((current_heat_loss > heatLoss[d_heatlosscount-1]) && d_loudHeatLossWarning) ){
      if (inStream.d_currentCell.x() == -2) {
        cout<< "Heat loss is exceeding the table bounds: "<<current_heat_loss 
            << " (at unknown cell) " << endl;
      } else {
        cout<< "Heat loss is exceeding the table bounds: "<<current_heat_loss 
            << " at cell " << inStream.d_currentCell << endl;
      }
      cout<< "Heat loss table bounds are: "<< heatLoss[0] 
          << " to "<< heatLoss[d_heatlosscount-1] << endl;
      cout<< "Absolute enthalpy is : "<< enthalpy << endl;
      cout<< "Adiabatic enthalpy is : "<< adiab_enthalpy << endl;
      cout<< "Sensible enthalpy is : "<< sensible_enthalpy << endl;
      cout<< "Mixture fraction is :  "<< mixFrac << endl;
      cout<< "Mixture fraction variance is :  "<< mixFracVars << endl;
    }
    if(fabs(current_heat_loss) < small)
      current_heat_loss = zero_heat_loss;
    if(current_heat_loss > heatLoss[d_heatlosscount-1])
      current_heat_loss = heatLoss[d_heatlosscount-1];
    else if (current_heat_loss < heatLoss[0])
      current_heat_loss = heatLoss[0];
  }
  // Looking for the properties corresponding to the heat loss
  
  outStream.d_temperature=tableLookUp(mixFrac, mixFracVars, current_heat_loss, T_index);  
  outStream.d_density=tableLookUp(mixFrac, mixFracVars, current_heat_loss, Rho_index);  
  outStream.d_cp=tableLookUp(mixFrac, mixFracVars, current_heat_loss, Cp_index);  
  outStream.d_enthalpy=adiab_enthalpy;
  outStream.d_co2=tableLookUp(mixFrac, mixFracVars, current_heat_loss, co2_index);  
  outStream.d_h2o=tableLookUp(mixFrac, mixFracVars, current_heat_loss, h2o_index);

  if (d_sulfur_chem) {
    outStream.d_h2s=tableLookUp(mixFrac, mixFracVars, current_heat_loss, h2s_index);
    outStream.d_so2=tableLookUp(mixFrac, mixFracVars, current_heat_loss, so2_index);
    outStream.d_so3=tableLookUp(mixFrac, mixFracVars, current_heat_loss, so3_index);
    outStream.d_sulfur=tableLookUp(mixFrac, mixFracVars, current_heat_loss, sulfur_index);

    outStream.d_s2=tableLookUp(mixFrac, mixFracVars, current_heat_loss, s2_index);
    outStream.d_sh=tableLookUp(mixFrac, mixFracVars, current_heat_loss, sh_index);
    outStream.d_so=tableLookUp(mixFrac, mixFracVars, current_heat_loss, so_index);
    outStream.d_hso2=tableLookUp(mixFrac, mixFracVars, current_heat_loss, hso2_index);

    outStream.d_hoso=tableLookUp(mixFrac, mixFracVars, current_heat_loss, hoso_index);
    outStream.d_hoso2=tableLookUp(mixFrac, mixFracVars, current_heat_loss, hoso2_index);
    outStream.d_sn=tableLookUp(mixFrac, mixFracVars, current_heat_loss, sn_index);
    outStream.d_cs=tableLookUp(mixFrac, mixFracVars, current_heat_loss, cs_index);

    outStream.d_ocs=tableLookUp(mixFrac, mixFracVars, current_heat_loss, ocs_index);
    outStream.d_hso=tableLookUp(mixFrac, mixFracVars, current_heat_loss, hso_index);
    outStream.d_hos=tableLookUp(mixFrac, mixFracVars, current_heat_loss, hos_index);
    outStream.d_hsoh=tableLookUp(mixFrac, mixFracVars, current_heat_loss, hsoh_index);

    outStream.d_h2so=tableLookUp(mixFrac, mixFracVars, current_heat_loss, h2so_index);
    outStream.d_hosho=tableLookUp(mixFrac, mixFracVars, current_heat_loss, hosho_index);
    outStream.d_hs2=tableLookUp(mixFrac, mixFracVars, current_heat_loss, hs2_index);
    outStream.d_h2s2=tableLookUp(mixFrac, mixFracVars, current_heat_loss, h2s2_index);
  }

  if (d_co_output)
    outStream.d_co=tableLookUp(mixFrac, mixFracVars, current_heat_loss, co_index);  
  if (d_soot_precursors) {
    outStream.d_c2h2=tableLookUp(mixFrac, mixFracVars, current_heat_loss, c2h2_index);
    outStream.d_ch4=tableLookUp(mixFrac, mixFracVars, current_heat_loss, ch4_index);
  }
  if (d_tabulated_soot)
    outStream.d_sootFV=tableLookUp(mixFrac, mixFracVars, current_heat_loss, soot_index);

  outStream.d_heatLoss = current_heat_loss;
  //need a better way to do this:
  if (co2rate_index != -1){
    outStream.d_co2rate=tableLookUp(mixFrac, mixFracVars, current_heat_loss, co2rate_index);
  }
  if (so2rate_index != -1){
     outStream.d_so2rate=tableLookUp(mixFrac, mixFracVars, current_heat_loss, so2rate_index);
  }
  if (mixmolweight_index != -1){
     outStream.d_mixmw=tableLookUp(mixFrac, mixFracVars, current_heat_loss, mixmolweight_index); 
  }
}

double
NewStaticMixingTable::tableLookUp(double mixfrac, double mixfracVars, double current_heat_loss, int var_index)
{
  double small = 1.0e-10;
  // compute index
  // nx - mixfrac, ny - mixfracVars, var_index - index of the variable
  // Enthalpy loss lookup
  double fmg=0.0, fpmg=0.0,s1=0.0,s2=0.0,var_value=0.0;
  int nhl_lo=0, nhl_hi=0;
  double dhl_lo=0.0, dhl_hi=0.0;
  if(current_heat_loss==0.0  && d_heatlosscount ==1){
    nhl_lo=0;
    nhl_hi=0;
    dhl_lo=0.0;
    dhl_hi=1.0;
  }
  else{
    for(int hl_index=0; hl_index < d_heatlosscount-1; hl_index++){
      dhl_lo = heatLoss[hl_index]-current_heat_loss;
      dhl_hi = heatLoss[hl_index+1]-current_heat_loss;
      if((dhl_lo*dhl_hi) == 0.0 && hl_index != 0){
        nhl_lo=hl_index+1;
        nhl_hi=nhl_lo;
        break;
      }
      else if((dhl_lo*dhl_hi) <= 0.0){
        nhl_lo=hl_index;
        nhl_hi=nhl_lo+1;
        break;
      }
    }
  }
  /*cout<< "nhl_lo is :" << nhl_lo<<endl;
    cout<< "nhl_hi is :" << nhl_hi<<endl;
    cout<< "dhl_lo is :" << dhl_lo<<endl;
    cout<< "dhl_hi is :" << dhl_hi<<endl;*/

  // Main loop
  for(int m_index = nhl_lo; m_index <= nhl_hi; m_index++){ 
    int nx_lo=0, nx_hi=0;

    //Non-unifrom mixture fraction 
    double df1=0.0,df2=0.0; 
    for(int index=0; index < d_mixfraccount-1; index++){
      df1 = meanMix[var_index][index]-mixfrac;
      df2 = meanMix[var_index][index+1]-mixfrac;
      if((df1*df2) == 0.0 && index != 0){
        nx_lo=index+1;
        nx_hi=nx_lo;
        break;
      }
      else if((df1*df2) <= 0.0){
        nx_lo=index;
        nx_hi=nx_lo+1;
        break;
      }
    }

    //cout<<"nx_lo index is  : "<<nx_lo<<endl;
    //cout<<"nx_hi index is  : "<<nx_hi<<endl;

  
    // Supports non-uniform normalized variance lookup  
    // Normalized variance
    double g=0.0;
    //Index for variances
    int k1=0,k2=0;
    //Weighing factors for variance
    double dk1=0.0,dk2=0.0;
    if(mixfracVars <= small){
      // Set the values to get the first entry
      g=0.0;
      k1=0;
      k2=0;
      dk1=0.0;
      dk2=1.0;
    }
    else{
      g=mixfracVars;
      // Finding the table entry
      for(int index=0; index < d_mixvarcount-1; index++){
        dk1 = variance[index]-g;
        dk2 = variance[index+1]-g;
        if((dk1*dk2) == 0.0 && index != 0){
          k1=index+1;
          k2=k1;
          break;
        }
        else if((dk1*dk2) <= 0.0){
          k1=index;
          k2=k1+1;
          break;
        }
      }
    }

    //cout<<" G value is :"<<g<<endl;
    //Interpolating the values
    fmg = (dk1*table[var_index][m_index*d_mixfraccount*d_mixvarcount+k2*d_mixfraccount+nx_lo]-dk2*table[var_index][m_index*d_mixfraccount*d_mixvarcount+k1*d_mixfraccount+nx_lo])/(dk1-dk2);
    //cout<< "FMG is :"<<fmg<<endl;
    fpmg = (dk1*table[var_index][m_index*d_mixfraccount*d_mixvarcount+k2*d_mixfraccount+nx_hi]-dk2*table[var_index][m_index*d_mixfraccount*d_mixvarcount+k1*d_mixfraccount+nx_hi])/(dk1-dk2);
    //cout<< "FPMG is : "<<fpmg<<endl;
    if(nhl_lo==nhl_hi){
      s1=(df1*fpmg-df2*fmg)/(df1-df2);
      s2=s1;
      break;
    }
    else if(m_index == nhl_lo)
      s1=(df1*fpmg-df2*fmg)/(df1-df2);
    else
      s2=(df1*fpmg-df2*fmg)/(df1-df2);
        
  }
  //cout<<"value of S1 is "<<s1<<endl;
  //cout<<"value of S2 is "<<s2<<endl;

  var_value = (dhl_lo*s2-dhl_hi*s1)/(dhl_lo-dhl_hi);
  return var_value; 
}

void
NewStaticMixingTable::readMixingTable( const string & inputfile )
{
  if( d_myworld->myrank() == 0 ) {
    cout << "Preparing to read the inputfile:   " << inputfile << "\n";
  }

  gzFile gzFp = gzopen( inputfile.c_str(), "r" );

  if( gzFp == NULL ) {
    // If errno is 0, then not enough memory to uncompress file.
    cout << "Error gz opening file " << inputfile << ".  Errno: " << errno << "\n";
    throw ProblemSetupException("Unable to open the given input file: " + inputfile, __FILE__, __LINE__);
  }

  d_f_stoich       = getDouble( gzFp );
  d_H_fuel         = getDouble( gzFp );
  d_H_air          = getDouble( gzFp );
  d_carbon_fuel    = getDouble( gzFp );
  d_carbon_air     = getDouble( gzFp );
  d_indepvarscount = getInt(    gzFp );

  if( d_myworld->myrank() == 0 ) {
    cout << "d_indepvars count: " << d_indepvarscount << "\n";
  }
  
  Hl_index = -1;
  F_index = -1;
  Fvar_index = -1;
  
  for (int ii = 0; ii < d_indepvarscount; ii++) {
    string varname = getString( gzFp );
    if( varname == "heat_loss" ) {
      Hl_index = ii;
    }
    else if( varname == "mixture_fraction" ) {
      F_index = ii;
    }
    else if( varname == "mixture_fraction_variance" ) {
      Fvar_index = ii;
    }
    if( d_myworld->myrank() == 0 ) {
      cout << varname << "\n";
    }
  }
  
  eachindepvarcount = vector<int>(d_indepvarscount);
  for (int ii = 0; ii < d_indepvarscount; ii++) {
    eachindepvarcount[ii] = getInt( gzFp );
  }

  d_heatlosscount = 1;
  d_mixfraccount = 1;
  d_mixvarcount = 1;
  if (!(Hl_index == -1))   d_heatlosscount = eachindepvarcount[Hl_index];
  if (!(F_index == -1))    d_mixfraccount  = eachindepvarcount[F_index];
  if (!(Fvar_index == -1)) d_mixvarcount   = eachindepvarcount[Fvar_index];

  if( d_myworld->myrank() == 0 ) {
    cout << d_heatlosscount << " " << d_mixfraccount << " " << d_mixvarcount << "\n";
  }

  // Total number of variables in the table: non-adaibatic table has sensibile enthalpy too
  d_varscount = getInt( gzFp );
  if( d_myworld->myrank() == 0 ) {
    cout<< "d_vars count: " << d_varscount << "\n";
  }
  vars_names= vector<string>(d_varscount);
  Rho_index = -1;
  T_index = -1;
  Cp_index = -1;
  Hs_index = -1;
  co2_index = -1;
  h2o_index = -1;
  
  h2s_index     = -1;
  so2_index     = -1;
  so3_index     = -1;
  sulfur_index  = -1;

  s2_index      = -1;
  sh_index      = -1;
  so_index      = -1;
  hso2_index    = -1;

  hoso_index    = -1;
  hoso2_index   = -1;
  sn_index      = -1;
  cs_index      = -1;

  ocs_index     = -1;
  hso_index     = -1;
  hos_index     = -1;
  hsoh_index    = -1;

  h2so_index    = -1;
  hosho_index   = -1;
  hs2_index     = -1;
  h2s2_index    = -1;
  
  co_index = -1;
  c2h2_index = -1;
  ch4_index = -1;
  soot_index = -1;

  co2rate_index = -1;
  so2rate_index = -1;

  mixmolweight_index = -1;  

  for (int ii = 0; ii < d_varscount; ii++) {
    vars_names[ii] = getString( gzFp );
    if(     vars_names[ii] == "density")
      Rho_index = ii;
    else if(vars_names[ii] == "temperature")
      T_index = ii;
    else if(vars_names[ii] == "heat_capacity")
      Cp_index = ii;
    else if(vars_names[ii] == "sensible_heat")
      Hs_index = ii;
    else if(vars_names[ii] == "CO2")
      co2_index = ii;
    else if(vars_names[ii] == "H2O")
      h2o_index = ii;

    else if(vars_names[ii] == "H2S")
      h2s_index = ii;
    else if(vars_names[ii] == "SO2")
      so2_index = ii;
    else if(vars_names[ii] == "SO3")
      so3_index = ii;
    else if(vars_names[ii] == "SULFUR")
      sulfur_index = ii;

    else if(vars_names[ii] == "S2")      s2_index         = ii;
    else if(vars_names[ii] == "SH")      sh_index         = ii;
    else if(vars_names[ii] == "SO")      so_index         = ii;
    else if(vars_names[ii] == "HSO2")    hso2_index       = ii;

    else if(vars_names[ii] == "HOSO")    hoso_index       = ii;
    else if(vars_names[ii] == "HOSO2")   hoso2_index      = ii;
    else if(vars_names[ii] == "SN")      sn_index         = ii;
    else if(vars_names[ii] == "CS")      cs_index         = ii;

    else if(vars_names[ii] == "OCS")     ocs_index        = ii;
    else if(vars_names[ii] == "HSO")     hso_index        = ii;
    else if(vars_names[ii] == "HOS")     hos_index        = ii;
    else if(vars_names[ii] == "HSOH")    hsoh_index       = ii;

    else if(vars_names[ii] == "H2SO")    h2so_index       = ii;
    else if(vars_names[ii] == "HOSHO")   hosho_index      = ii;
    else if(vars_names[ii] == "HS2")     hs2_index        = ii;
    else if(vars_names[ii] == "H2S2")    h2s2_index       = ii;

    else if(vars_names[ii] == "rate_CO2") co2rate_index   = ii;
    else if(vars_names[ii] == "rate_SO2") so2rate_index   = ii;

    else if(vars_names[ii] == "mixture_molecular_weight") mixmolweight_index = ii; 

    else if(vars_names[ii] == "CO")
      co_index = ii;
    else if(vars_names[ii] == "C2H2")
      c2h2_index = ii;
    else if(vars_names[ii] == "CH4")
      ch4_index = ii;
    else if(vars_names[ii] == "sootFV")
      soot_index = ii;

    if( d_myworld->myrank() == 0 ) {
      cout<<vars_names[ii]<<endl;
    }
  }
  
  if ((F_index == -1)||(Rho_index == -1)) 
    throw InvalidValue("No mixture fraction or density found in table "
                       + inputfile, __FILE__, __LINE__);

  if (((Hs_index == -1)||(Hl_index == -1))&&(d_calcEnthalpy)) 
    throw InvalidValue("No sensible heat or heat loss found in table "
                       + inputfile, __FILE__, __LINE__);

  if ((Fvar_index == -1)&&(d_calcVariance)) 
    throw InvalidValue("No variance found in table " + inputfile,
                       __FILE__, __LINE__);

  if ((T_index == -1)||(Cp_index == -1)||(co2_index == -1)||(h2o_index == -1)) 
    throw InvalidValue("No temperature or Cp or CO2 or H2O found in table "
                       + inputfile, __FILE__, __LINE__);
  if ((d_sulfur_chem)&&
      ( (h2s_index == -1)||
        (so2_index == -1)||
        (so3_index == -1)||
        (sulfur_index == -1)
        )) 
    throw InvalidValue("No H2S, SO2, S or SO3 for sulfur chemistry found in table "
                       + inputfile, __FILE__, __LINE__);

  if ((d_sulfur_chem)&&
      ( (s2_index == -1) || 
        (sh_index == -1) || 
        (so_index == -1) || 
        (hso2_index == -1)
        )) 
    throw InvalidValue("No s2, sh, so, or hso2 for sulfur chemistry found in table "
                       + inputfile, __FILE__, __LINE__);

  if ((d_sulfur_chem)&&
      ( (hoso_index == -1) || 
        (hoso2_index == -1) || 
        (sn_index == -1) || 
        (cs_index == -1)
        ))
    throw InvalidValue("No hoso, hoso2, sn, or cs for sulfur chemistry found in table "
                       + inputfile, __FILE__, __LINE__);
        

  if ((d_sulfur_chem)&&
      ( (ocs_index == -1) || 
        (hso_index == -1) || 
        (hos_index == -1) || 
        (hsoh_index == -1)
        ))
    throw InvalidValue("No ocs, hso, hos, or hsoh for sulfur chemistry found in table "
                       + inputfile, __FILE__, __LINE__);

  if ((d_sulfur_chem)&&
      ( (h2so_index == -1) || 
        (hosho_index == -1) || 
        (hs2_index == -1) || 
        (h2s2_index == -1)
        ))
    throw InvalidValue("No h2so, hosho, hs2, or h2s2 for sulfur chemistry found in table "
                       + inputfile, __FILE__, __LINE__);

  if ((d_co_output)&&(co_index == -1))
    throw InvalidValue("No CO found in table " + inputfile,
                       __FILE__, __LINE__);

  if ((d_soot_precursors)&&((c2h2_index == -1)||(ch4_index == -1)))
    throw InvalidValue("No C2H2 or CH4 found in table " + inputfile,
                       __FILE__, __LINE__);

  if ((d_tabulated_soot)&&(soot_index == -1))
    throw InvalidValue("No sootFV found in table " + inputfile,
                       __FILE__, __LINE__);

  if( d_myworld->myrank() == 0 ) {
    cout << "CO2 index is "  << co2_index  << endl;
    cout << "H2O index is "  << h2o_index  << endl;
  }
  
  if (d_sulfur_chem) {
    if( d_myworld->myrank() == 0 ) {
      cout  << "h2s index is "      << h2s_index    << endl; 
      cout  << "so2 index is "      << so2_index    << endl;
      cout  << "so3 index is "      << so3_index    << endl;
      cout  << "sulfur index is "   << sulfur_index << endl;
      cout  << "s2 index is "       << s2_index     << endl;
      cout  << "sh index is "       << sh_index     << endl;
      cout  << "so index is "       << so_index     << endl;
      cout  << "hso2 index is "     << hso2_index   << endl;
      cout  << "hoso index is "     << hoso_index   << endl;
      cout  << "hoso2 index is "    << hoso2_index  << endl;
      cout  << "sn index is "       << sn_index     << endl;
      cout  << "cs index is "       << cs_index     << endl;
      cout  << "ocs index is "      << ocs_index    << endl;
      cout  << "hso index is "      << hso_index    << endl;
      cout  << "hos index is "      << hos_index    << endl;
      cout  << "hsoh index is "     << hsoh_index   << endl;
      cout  << "h2so index is "     << h2so_index   << endl;
      cout  << "hosho index is "    << hosho_index  << endl;
      cout  << "hs2 index is "      << hs2_index    << endl;
      cout  << "h2s2 index is "     << h2s2_index   << endl;
    }
  }

  if( d_myworld->myrank() == 0 ) {
    cout << "CO index is "   << co_index   << endl;
    cout << "C2H2 index is " << c2h2_index << endl;
    cout << "CH4 index is "  << ch4_index  << endl;
    cout << "sootFV index is "  << soot_index  << endl;
    cout << "mixture molecular weight index is " << mixmolweight_index << endl;

    if (co2rate_index != -1) {
          cout << "CO2 rxn rate index is " << co2rate_index << endl;
    }
    if (so2rate_index != -1) {
      cout << "SO2 rxn rate index is " << so2rate_index << endl;
    }
  }

  // Not sure if we care about units in runtime, read them just in case
  vars_units= vector<string>(d_varscount);
  for (int ii = 0; ii < d_varscount; ii++) {
    vars_units[ii] = getString( gzFp );
  }
  
  // Enthalpy loss vector
  heatLoss=vector<double>(d_heatlosscount);
  // Reading heat loss values

  if (Hl_index != -1) {
    for (int mm=0; mm< d_heatlosscount; mm++){
      heatLoss[mm] = getDouble( gzFp );
      //if( d_myworld->myrank() == 0 ) {
      //  cout << heatLoss[mm] << endl;
      //}
    }
  }

  // Non-dimensionalized variance values normalized by maximum mixture fraction: f_mean*(1.0-f_mean)
  variance=vector<double>(d_mixvarcount);
  // Reading variance values
  // Jennifer's bug fix
  if (Fvar_index != -1) {
    for (int mm=0; mm< d_mixvarcount; mm++){
      variance[mm] = getDouble( gzFp );
    }
  }

  // Intitializing the mixture fraction table meanMix
  meanMix = vector < vector<double> >(d_varscount);
  for(int ii=0; ii<d_varscount; ii++)
    meanMix[ii]=vector<double>(d_mixfraccount);

  // Allocating the table space 
  int size = d_heatlosscount*d_mixfraccount*d_mixvarcount;
  table = vector <vector <double> > (d_varscount);
  for (int ii = 0; ii < d_varscount; ii++)
    table[ii] = vector <double> (size);

  //Reading the data
  // Variables loop 
  for (int kk=0; kk< d_varscount; kk++){
    // Reading mixture fraction values
    for (int ii=0; ii< d_mixfraccount; ii++){
      meanMix[kk][ii] = getDouble( gzFp );
      //cout << ii << ": " << meanMix[kk][ii] << "      ";
    }
    //cout << "\n";
    // Enthalpy loss loop
    for (int mm=0; mm< d_heatlosscount; mm++) {   // Variance loop
      for (int jj=0;jj<d_mixvarcount; jj++){      // Mixture fraction loop 
        for (int ii=0; ii< d_mixfraccount; ii++){
          table[kk][mm*(d_mixfraccount*d_mixvarcount)+jj*d_mixfraccount+ii] = getDouble( gzFp );
          //cout << table[kk][mm*(d_mixfraccount*d_mixvarcount)+jj*d_mixfraccount+ii]<< "  ";
        } // End of mixture fraction loop
        //cout << "\n";
      } // End of variance loop
    } // End of enthalpy loss loop
  } // End of Variables loop
  
  // Closing the file pointer
  gzclose( gzFp );
  if( d_myworld->myrank() == 0 ) {
    cout << "Table reading is successful" << endl;
  }
}
