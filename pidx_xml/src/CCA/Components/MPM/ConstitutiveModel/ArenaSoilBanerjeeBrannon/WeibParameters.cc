/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 * Copyright (c) 2013-2014 Callaghan Innovation, New Zealand
 * Copyright (c) 2015-2017 Parresia Research Limited, New Zealand
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

#include <CCA/Components/MPM/ConstitutiveModel/ArenaSoilBanerjeeBrannon/WeibParameters.h>
#include <Core/Parallel/Parallel.h>  // fror proc0cout
#include <Core/Exceptions/InvalidValue.h>

using namespace Uintah;

WeibParameters::WeibParameters()
{
  d_Perturb = "False"; // 'True' for perturbed parameter
  d_WeibDist = "None"; // String for Distribution
  d_WeibSeed = 0;      // seed for random number generator
  d_WeibMed = 0.0;     // Median distrib. value OR const value depending on bool Perturb
  d_WeibMod = 0.0;     // Weibull modulus
  d_WeibRefVol = 0.0;  // Reference Volume
}

WeibParameters::WeibParameters(const std::string& weibDist)
{
  WeibullParser(weibDist);
}

WeibParameters::WeibParameters(const WeibParameters* weibull)
{
  d_Perturb = weibull->d_Perturb;
  d_WeibDist = weibull->d_WeibDist;
  d_WeibSeed = weibull->d_WeibSeed;
  d_WeibMed = weibull->d_WeibMed;
  d_WeibMod = weibull->d_WeibMod;
  d_WeibRefVol = weibull->d_WeibRefVol;
}

WeibParameters& 
WeibParameters::operator=(const WeibParameters& weibull)
{
  d_Perturb = weibull.d_Perturb;
  d_WeibDist = weibull.d_WeibDist;
  d_WeibSeed = weibull.d_WeibSeed;
  d_WeibMed = weibull.d_WeibMed;
  d_WeibMod = weibull.d_WeibMod;
  d_WeibRefVol = weibull.d_WeibRefVol;

  return *this;
}

WeibParameters::~WeibParameters()
{
}

void 
WeibParameters::WeibullParser(const std::string& weibDist)
{
  d_WeibDist = weibDist;

  // Remove all unneeded characters
  // only remaining are alphanumeric '.' and ','
  for ( int i = d_WeibDist.length()-1; i >= 0; i--) {
    d_WeibDist[i] = tolower(d_WeibDist[i]);
    if(!isalnum(d_WeibDist[i]) && d_WeibDist[i] != '.' && d_WeibDist[i] != ',' &&
        d_WeibDist[i] != '-' && d_WeibDist[i] != EOF) {
      d_WeibDist.erase(i,1);
    }
  } // End for

  if (d_WeibDist.substr(0,4) == "weib") {
    d_Perturb = true;
  } else {
    d_Perturb = false;
  }

  // ######
  // If perturbation is NOT desired
  // ######
  if( !d_Perturb ){
    bool escape = false;
    int num_of_e = 0;
    int num_of_periods = 0;
    for( unsigned int i = 0; i < d_WeibDist.length(); i++){
      if( d_WeibDist[i] != '.'
          && d_WeibDist[i] != 'e'
          && d_WeibDist[i] != '-'
          && !isdigit(d_WeibDist[i]))
        escape = true;
      if( d_WeibDist[i] == 'e' )
        num_of_e += 1;
      if( d_WeibDist[i] == '.' )
        num_of_periods += 1;
      if( num_of_e > 1 || num_of_periods > 1 || escape ){
        std::ostringstream out;
        out << "** ERROR: ** Input value cannot be parsed. Please"
            << " check your input values." << std::endl;
        throw InvalidValue(out.str(), __FILE__, __LINE__);
      }
    } // end for(int i = 0;....)

    try {
      d_WeibMed  = std::stod(d_WeibDist);
    } catch (std::invalid_argument) {
      std::ostringstream out;
      out << "** ERROR: ** Input value " << d_WeibDist << " cannot be parsed. Please"
          << " check your input values." << std::endl;
      throw InvalidValue(out.str(), __FILE__, __LINE__);
    }
  }

  // ######
  // If perturbation IS desired
  // ######
  if( d_Perturb ){

    int weibValues[4];
    int weibValuesCounter = 0;
    for( unsigned int r = 0; r < d_WeibDist.length(); r++){
      if( d_WeibDist[r] == ',' ){
        weibValues[weibValuesCounter] = r;
        weibValuesCounter += 1;
      } // end if(d_WeibDist[r] == ',')
    } // end for(int r = 0; ...... )

    if (weibValuesCounter != 4) {
      std::ostringstream out;
      out << "** ERROR: ** Weibull perturbed input string must contain "
          << "exactly 4 commas. Verify that your input string is "
          << "of the form 'weibull, 45e6, 4, 0.001, 1'." << std::endl;
      throw InvalidValue(out.str(), __FILE__, __LINE__);
    } // end if(weibValuesCounter != 4)

    std::string weibMedian;
    std::string weibModulus;
    std::string weibRefVol;
    std::string weibSeed;
    weibMedian  = d_WeibDist.substr(weibValues[0]+1,weibValues[1]-weibValues[0]-1);
    weibModulus = d_WeibDist.substr(weibValues[1]+1,weibValues[2]-weibValues[1]-1);
    weibRefVol  = d_WeibDist.substr(weibValues[2]+1,weibValues[3]-weibValues[2]-1);
    weibSeed    = d_WeibDist.substr(weibValues[3]+1);
    d_WeibMed    = std::stod(weibMedian);
    d_WeibMod    = std::stod(weibModulus);
    d_WeibRefVol = std::stod(weibRefVol);
    d_WeibSeed   = std::stoi(weibSeed);
  } // End if (d_Perturb)
}

void 
WeibParameters::assignWeibullVariability(const Patch* patch,
                                         ParticleSubset* pset,
                                         constParticleVariable<double>& pVolume,
                                         ParticleVariable<double>& pvar)
{
  if (d_Perturb) { 

    proc0cout << "Perturbing parameters." << std::endl;
    // Make the seed differ for each patch, otherwise each patch gets the
    // same set of random #s.
    int patchID = patch->getID();
    int patch_div_32 = patchID/32;
    patchID = patchID%32;

    unsigned int unique_seed = ((d_WeibSeed+patch_div_32+1) << patchID);
    Uintah::Weibull weibGen(d_WeibMed, d_WeibMod, d_WeibRefVol,
                            unique_seed, d_WeibMod);

    for (auto iter = pset->begin(); iter != pset->end(); iter++) {

      //set value with variability and scale effects
      pvar[*iter] = weibGen.rand(pVolume[*iter]);

      //set value with ONLY scale effects
      if (d_WeibSeed == 0) {
        pvar[*iter]= pow(d_WeibRefVol/pVolume[*iter], 1.0/d_WeibMod)*d_WeibMed;
      }
    }
  }
}
