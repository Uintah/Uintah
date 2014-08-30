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

#include "AbsCoeffGas.h"
#include "TestHelper.h"

#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace std;

int main( int argc, char* argv[] )
{
  TestHelper status(false);

  std::vector<RadiativeSpecies> gasSpeciesEnum(5);
  gasSpeciesEnum[0]= H2O;
  gasSpeciesEnum[1]= CO2;
  gasSpeciesEnum[2]= CO;
  gasSpeciesEnum[3]= NO;
  gasSpeciesEnum[4]= OH;


  try{
    const FSK fsk1(gasSpeciesEnum);
    const FSK fsk2("Gs.txt");
    std::vector<RadiativeSpecies> gasSpeciesCheck;
    gasSpeciesCheck = fsk2.species();
    vector<string> gasSpecies;
    for (int r = 0; r<5; r++) {
      gasSpecies.push_back(species_name( fsk2.species()[r] ));
      cout << "Order = " << gasSpecies[r] << endl;
    }

    std::vector<double> myMoleFracs (5);
    myMoleFracs[0]=0.1;
    myMoleFracs[1]=0.4;
    myMoleFracs[2]=0.15;
    myMoleFracs[3]=0.3;
    myMoleFracs[4]=0.05;

    std::vector<double> aF1, aF2;
    fsk1.a_function(aF1,myMoleFracs,1030,1080);
    fsk2.a_function(aF2,myMoleFracs,1030,1080);

    //Test a Func
    for( size_t r = 0; r<aF1.size(); ++r ){
      const double err = std::abs( aF1[r] -aF2[r] )/aF1[r];
      status( err < 1e-9 );
      if( err>1e-9 ){
        cout << "PROBLEMS!  " << r << " of " << aF1.size() << " : "
            << aF1[r] << " : " << aF2[r] << " : " <<  std::abs( aF1[r] -aF2[r] )/aF1[r]
            << endl;
      }
    }

    // jcs need to add some more temperatures
    status( std::abs( fsk1.mixture_abs_coeff( myMoleFracs,  1050.0, 0.1 ) -
                      fsk2.mixture_abs_coeff( myMoleFracs,  1050.0, 0.1 ))/fsk1.mixture_abs_coeff( myMoleFracs,  1050.0, 0.1 ) < 1e-5 );

    if( status.ok() ){
      cout << "PASS" << endl;
      return 0;
    }
  }
  catch( std::exception& err ){
    cout << err.what() << endl;
  }

  cout << "FAIL" << endl;
  return -1;
}
