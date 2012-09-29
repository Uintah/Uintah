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

//----- Stream.cc --------------------------------------------------

#include <CCA/Components/Arches/Mixing/Stream.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/InvalidValue.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace Uintah;
using namespace std;

Stream::Stream()
{
  d_CO2index = 0;
  d_H2Oindex = 0;
  d_speciesConcn = vector<double>(0); // initialize with 0
  d_rxnVarRates = vector<double>(1, 0.0); // initialize with 0
  d_pressure = 0.0;
  d_density = 0.0;
  d_temperature = 0.0;
  d_enthalpy = 0.0;
  d_sootFV = 0.0;
  d_sensibleEnthalpy = 0.0;
  d_moleWeight = 0.0;
  d_cp = 0.0;
  d_drhodf = 0.0;
  d_drhodh = 0.0;
  d_mole = false;
  d_numMixVars = 0;
  d_numRxnVars = 0;
}

Stream::Stream(int numSpecies,  int numElements)
{
  d_speciesConcn = vector<double>(numSpecies, 0.0); // initialize with 0
  d_pressure = 0.0;
  d_density = 0.0;
  d_temperature = 0.0;
  d_enthalpy = 0.0;
  d_sootFV = 0.0;
  d_sensibleEnthalpy = 0.0;
  d_moleWeight = 0.0;
  d_cp = 0.0;
  d_drhodf = 0.0;
  d_drhodh = 0.0;
  d_mole = false;
  d_numMixVars = 0;
  d_numRxnVars = 0;
  d_lsoot = false;
  //d_atomNumbers = vector<double>(numElements, 0.0); // initialize with 0
  // NUM_DEP_VARS corresponds to pressure, density, temp, enthalpy, sensh, 
  // cp, molwt; total number of dependent state space variables also 
  // includes mass fraction of each species
  d_depStateSpaceVars = NUM_DEP_VARS + numSpecies;
  d_CO2index = 0;
  d_H2Oindex = 0;
}


Stream::Stream(int numSpecies, int numElements, int numMixVars, 
               int numRxnVars, bool lsoot): 
  d_numMixVars(numMixVars), d_numRxnVars(numRxnVars), d_lsoot(lsoot)
{
  d_speciesConcn = vector<double>(numSpecies, 0.0); // initialize with 0
  d_pressure = 0.0;
  d_density = 0.0;
  d_temperature = 0.0;
  d_enthalpy = 0.0;
  d_sensibleEnthalpy = 0.0;
  d_moleWeight = 0.0;
  d_cp = 0.0;
  d_drhodf = 0.0;
  d_drhodh = 0.0;
  d_mole = false;
  //d_atomNumbers = vector<double>(numElements, 0.0); // initialize with 0
  if (d_numRxnVars > 0) {
    d_rxnVarRates = vector<double>(d_numRxnVars, 0.0); // initialize with 0
    d_rxnVarNorm = vector<double>(2*d_numRxnVars, 0.0); // min & max values
                                         // for normalizing rxn parameter

  }
  int sootTrue = 0;
  if (d_lsoot) {
    d_sootData = vector<double>(2, 0.0);
    sootTrue = 1;
  }
  // NUM_DEP_VARS corresponds to pressure, density, temp, enthalpy, sensh, 
  // cp, molwt; total number of dependent state space variables also 
  // includes mass fraction of each species, soot volume fraction and 
  // diameter, source terms (rxn rates) for each rxn variable, and min/max 
  // values of each rxn variable for normalization
  d_depStateSpaceVars = NUM_DEP_VARS + numSpecies + 2*sootTrue
    + 3*d_numRxnVars;
  d_CO2index = 0;
  d_H2Oindex = 0;
}


Stream::Stream(const Stream& strm) // copy constructor
{
  d_speciesConcn = strm.d_speciesConcn;
  d_pressure = strm.d_pressure;
  d_density = strm.d_density;
  d_temperature = strm.d_temperature;
  d_enthalpy = strm.d_enthalpy;
  d_sensibleEnthalpy = strm.d_sensibleEnthalpy;
  d_moleWeight = strm.d_moleWeight;
  d_cp = strm.d_cp;
  d_drhodf = strm.d_drhodf;
  d_drhodh = strm.d_drhodh;
  d_depStateSpaceVars = strm.d_depStateSpaceVars;
  d_mole = strm.d_mole;
  d_numMixVars = strm.d_numMixVars;
  d_numRxnVars = strm.d_numRxnVars;
  d_lsoot = strm.d_lsoot;
  //d_atomNumbers = strm.d_atomNumbers;
  if (strm.d_numRxnVars > 0) {
    d_rxnVarRates = strm.d_rxnVarRates;
    d_rxnVarNorm = strm.d_rxnVarNorm;
  }
  if (strm.d_lsoot)
    d_sootData = strm.d_sootData;
  d_CO2index = strm.d_CO2index;
  d_H2Oindex = strm.d_H2Oindex;
}

Stream&
Stream::operator=(const Stream &rhs)
{
  // Guard against self-assignment
  if (this != &rhs)
    {        
      d_speciesConcn = rhs.d_speciesConcn;
      d_pressure = rhs.d_pressure;
      d_density = rhs.d_density;
      d_temperature = rhs.d_temperature;
      d_enthalpy = rhs.d_enthalpy;
      d_sensibleEnthalpy = rhs.d_sensibleEnthalpy;
      d_moleWeight = rhs.d_moleWeight;
      d_cp = rhs.d_cp;
      d_depStateSpaceVars = rhs.d_depStateSpaceVars;
      d_mole = rhs.d_mole;
      d_drhodf = rhs.d_drhodf;
      d_drhodh = rhs.d_drhodh;
      d_numMixVars = rhs.d_numMixVars;
      d_numRxnVars = rhs.d_numRxnVars;
      d_lsoot = rhs.d_lsoot;
      //d_atomNumbers = rhs.d_atomNumbers;
      if (rhs.d_numRxnVars > 0) {
        d_rxnVarRates = rhs.d_rxnVarRates;
        d_rxnVarNorm = rhs.d_rxnVarNorm;
      }
      if (rhs.d_lsoot)
        d_sootData = rhs.d_sootData;
      d_CO2index = rhs.d_CO2index;
      d_H2Oindex = rhs.d_H2Oindex;
    }
  return *this;
  }

bool
Stream::operator==(const Stream &rhs)
{
  // Checks element by element for equality of two streams
  if (d_density != rhs.d_density)
    return(false);
  else if (d_pressure != rhs.d_pressure)
    return(false);
  else if (d_temperature != rhs.d_temperature)
    return(false);
  else if (d_enthalpy != rhs.d_enthalpy)
    return(false);
  else if (d_sensibleEnthalpy != rhs.d_sensibleEnthalpy)
    return(false);
  else if (d_moleWeight != rhs.d_moleWeight)
    return(false);
  else if (d_cp != rhs.d_cp)
    return(false);
  else if (d_drhodf != rhs.d_drhodf)
    return(false);
  else if (d_drhodh != rhs.d_drhodh)
    return(false);
  else if (d_speciesConcn != rhs.d_speciesConcn)
    return(false);
  else if (d_mole != rhs.d_mole)
    return(false);
  else if (d_lsoot != rhs.d_lsoot)
    return(false);
  else if( rhs.d_numRxnVars > 0 && 
         ( ( d_rxnVarRates != rhs.d_rxnVarRates ) || (d_rxnVarNorm != rhs.d_rxnVarNorm) ) ) {
    return false;
  }
  else if( rhs.d_lsoot && ( d_sootData != rhs.d_sootData ) ) {
    return false;
  }
  else if (d_CO2index != rhs.d_CO2index)
    return(false);
  else if (d_H2Oindex != rhs.d_H2Oindex)
    return(false);
  else
    return(true);
} 

Stream::~Stream()
{
}


double
Stream::getValue(int count) 
{
  int sumVars = NUM_DEP_VARS + d_speciesConcn.size();
  if ((d_numRxnVars > 0)&&(count >= sumVars))
    {
      if (count < (sumVars + d_numRxnVars))
        return d_rxnVarRates[count-NUM_DEP_VARS-d_speciesConcn.size()];
      else if (count < (sumVars + 3*d_numRxnVars))
        return d_rxnVarNorm[count-NUM_DEP_VARS-d_speciesConcn.size()- d_numRxnVars];
      else if ((count < d_depStateSpaceVars)&&d_lsoot)
        return d_sootData[count-NUM_DEP_VARS-d_speciesConcn.size()- 3*d_numRxnVars];
      else {
        cerr << "Invalid count value" << '\n';
        return 0;
      }
    }        
  if ((count >= NUM_DEP_VARS)&&(count < sumVars))
    return d_speciesConcn[count-NUM_DEP_VARS];
  else
    {
      switch (count) {
      case 0:
        //return d_density;
        return 1.0/d_density;
      case 1:
        return d_pressure;
      case 2:
        //return d_temperature;
        return d_temperature/d_density; //To satisfy argument that T measurements in 
        // flames are Reynolds-averaged while other measurements are favre-averaged
      case 3:
        return d_enthalpy;
      case 4:
        return d_sensibleEnthalpy;
      case 5:
        return d_moleWeight;
      case 6:
        return d_cp;
      case 7:
        return d_drhodf;
      case 8:
        return d_drhodh;
      default:
        cerr << "Invalid count value" << '\n';
        return 0;
      }
    }
}

void
Stream::normalizeStream() {
  double sum = 0.0;
  for (vector<double>::iterator iter = d_speciesConcn.begin();
       iter != d_speciesConcn.end(); ++iter) 
    sum += *iter;
  for (vector<double>::iterator iter = d_speciesConcn.begin();
       iter != d_speciesConcn.end(); ++iter) 
    *iter /= sum;
}  
        
void
Stream::convertVecToStream(const vector<double>& vec_stateSpace,
                           int numMixVars, int numRxnVars, bool lsoot) {
  d_depStateSpaceVars = vec_stateSpace.size();
  d_density = vec_stateSpace[0];
  d_pressure = vec_stateSpace[1];
  d_temperature = vec_stateSpace[2];
  d_enthalpy = vec_stateSpace[3];
  d_sensibleEnthalpy = vec_stateSpace[4];
  d_moleWeight = vec_stateSpace[5];
  d_cp = vec_stateSpace[6];
  d_drhodf = vec_stateSpace[7];
  d_drhodh = vec_stateSpace[8];
  d_numMixVars = numMixVars;
  d_numRxnVars = numRxnVars;
  d_lsoot = lsoot;
  d_mole = false; //???Is this true???
  int incSoot = 0;
  if (lsoot)
    incSoot = 2;
  d_speciesConcn = vector<double> (vec_stateSpace.begin()+NUM_DEP_VARS,
                                   vec_stateSpace.end()-3*d_numRxnVars-incSoot);
  if (d_numRxnVars > 0) {
    d_rxnVarRates = vector<double> (vec_stateSpace.end()-3*d_numRxnVars-incSoot, 
                                    vec_stateSpace.end()-2*d_numRxnVars-incSoot);
    d_rxnVarNorm = vector<double> (vec_stateSpace.end()-2*d_numRxnVars-incSoot, 
                                   vec_stateSpace.end()-incSoot);
  } 
  if (lsoot) {
    d_sootData = vector<double> (vec_stateSpace.end()-2, 
                                 vec_stateSpace.end());
  }
#if 0
  cout << "Density: "<< d_density << endl;
  cout << "Pressure: "<< d_pressure << endl;
  cout << "Temperature: "<< d_temperature << endl;
  cout << "Enthalpy: "<< d_enthalpy << endl;
  cout << "Sensible Enthalpy: "<< d_sensibleEnthalpy << endl;
  cout << "Molecular Weight: "<< d_moleWeight << endl;
  cout << "CP: "<< d_cp << endl;
  cout << "Species concentration in mass fraction: " << endl;
  for (int ii = 0; ii < d_speciesConcn.size(); ii++) {
    cout.width(10);
    cout << d_speciesConcn[ii] << " " ; 
    if (!(ii % 10)) cout << endl; 
    cout << endl;
  }
#endif
}
     

vector<double>
//Stream::convertStreamToVec(bool lsoot)
Stream::convertStreamToVec()
{
  vector<double> vec_stateSpace;
  vec_stateSpace.push_back(d_density);
  vec_stateSpace.push_back(d_pressure);
  vec_stateSpace.push_back(d_temperature);
  vec_stateSpace.push_back(d_enthalpy);
  vec_stateSpace.push_back(d_sensibleEnthalpy);
  vec_stateSpace.push_back(d_moleWeight);
  vec_stateSpace.push_back(d_cp);
  vec_stateSpace.push_back(d_drhodf);
  vec_stateSpace.push_back(d_drhodh);
  // copy d_speciesConcn to rest of the vector
  for (vector<double>::iterator iter = d_speciesConcn.begin(); 
       iter != d_speciesConcn.end(); ++iter) {
    vec_stateSpace.push_back(*iter);
    //cout << d_speciesConcn[jj++] <<  endl;
  }
  if (d_numRxnVars > 0) {
    for (vector<double>::iterator iter = d_rxnVarRates.begin(); 
         iter != d_rxnVarRates.end(); ++iter)
      vec_stateSpace.push_back(*iter);
    for (vector<double>::iterator iter = d_rxnVarNorm.begin(); 
         iter != d_rxnVarNorm.end(); ++iter)
      vec_stateSpace.push_back(*iter);
  }
  if (d_lsoot) {
    for (vector<double>::iterator iter = d_sootData.begin(); 
       iter != d_sootData.end(); ++iter)
    vec_stateSpace.push_back(*iter);
  }
  return vec_stateSpace;
}


Stream& Stream::linInterpolate(double upfactor, double lowfactor,
                               Stream& rightvalue) {
  d_pressure = upfactor*d_pressure+lowfactor*rightvalue.d_pressure;
  d_density = upfactor*d_density+lowfactor*rightvalue.d_density;
  d_temperature = upfactor*d_temperature+lowfactor*rightvalue.d_temperature;
  d_enthalpy = upfactor*d_enthalpy+lowfactor*rightvalue.d_enthalpy;
  d_sensibleEnthalpy = upfactor*d_sensibleEnthalpy+lowfactor*
                                           rightvalue.d_sensibleEnthalpy;
  d_moleWeight = upfactor*d_moleWeight+lowfactor*rightvalue.d_moleWeight;
  d_cp = upfactor*d_cp+lowfactor*rightvalue.d_cp;
  d_drhodf = upfactor*d_drhodf+lowfactor*rightvalue.d_drhodf;
  d_drhodh = upfactor*d_drhodh+lowfactor*rightvalue.d_drhodh;

  for (int i = 0; i < (int)d_speciesConcn.size(); i++)
    d_speciesConcn[i] = upfactor*d_speciesConcn[i] +
                   lowfactor*rightvalue.d_speciesConcn[i];
  if (d_numRxnVars > 0) {
    for (int i = 0; i < (int)d_rxnVarRates.size(); i++)
      d_rxnVarRates[i] = upfactor*d_rxnVarRates[i] +
        lowfactor*rightvalue.d_rxnVarRates[i];
    for (int i = 0; i < (int)d_rxnVarNorm.size(); i++)
      d_rxnVarNorm[i] = upfactor*d_rxnVarNorm[i] +
        lowfactor*rightvalue.d_rxnVarNorm[i];
  }
  if (d_lsoot) {
    for (int i = 0; i < (int)d_sootData.size(); i++)
      d_sootData[i] = upfactor*d_sootData[i] +
        lowfactor*rightvalue.d_sootData[i];
  }

  return *this;
}

void
Stream::print(std::ostream& out) const {
  out << "Integrated values"<< '\n';
  out << "Density: "<< d_density << endl;
  out << "Pressure: "<< d_pressure << endl;
  out << "Temperature: "<< d_temperature << endl;
  out << "Enthalpy: "<< d_enthalpy << endl;
  out << "Sensible Enthalpy: "<< d_sensibleEnthalpy << endl;
  out << "Molecular Weight: "<< d_moleWeight << endl;
  out << "CP: "<< d_cp << endl;
  out << "Drhodf: "<< d_drhodf << endl;
  out << "Drhodh: "<< d_drhodh << endl;
  out << "Species concentration in mass fraction: " << endl;
  for (int ii = 0; ii < (int)d_speciesConcn.size(); ii++) {
    out.width(10);
    out << d_speciesConcn[ii] << " " ; 
    if (!(ii % 10)) out << endl; 
    out << endl;
  }
  if (d_lsoot) {
    out << "Soot Data: " << endl;
    for (int ii = 0; ii < (int)d_sootData.size(); ii++) {
      out.width(10);
      out << d_sootData[ii] << " " ; 
      if (!(ii % 10)) out << endl; 
    }
  }
  out << endl;
}


void 
Stream::print_oneline(std::ofstream& out) {
  out << d_density << " " ;
  out << d_pressure << " " ;
  out << d_temperature << " " ;
  out << d_enthalpy << " " ;
  out << d_sensibleEnthalpy << " " ;
  out << d_moleWeight << " " ;
  out << d_cp << " " ;
  out << d_drhodf << " " ;
  out << d_drhodh << " " ;
  for (int ii = 0; ii < (int)d_speciesConcn.size(); ii++)
    out << d_speciesConcn[ii] << " " ;
  if (d_numRxnVars > 0) {
    for (int ii = 0; ii < (int)d_rxnVarRates.size(); ii++)
      out << d_rxnVarRates[ii] << " " ;
    for (int ii = 0; ii < (int)d_rxnVarNorm.size(); ii++)
      out << d_rxnVarNorm[ii] << " " ;
  }
  if (d_lsoot) {
    for (int ii = 0; ii < (int)d_sootData.size(); ii++)
      out << d_sootData[ii] << " " ;
  }
  out << endl;
}
