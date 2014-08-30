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
#include <iomanip>

using namespace std;


int main( int argc, char* argv[] )
{
  TestHelper status(true);

  std::vector<RadiativeSpecies> gasSpeciesEnum(5);
  gasSpeciesEnum[0]= H2O;
  gasSpeciesEnum[1]= CO2;
  gasSpeciesEnum[2]= CO;
  gasSpeciesEnum[3]= NO;
  gasSpeciesEnum[4]= OH;

  try{

    const string fname = "GreyGasProperties.txt";
    const GreyGas g1(gasSpeciesEnum,0.01,fname);
    const GreyGas g2(fname);

    std::vector<RadiativeSpecies> gasSpeciesCheck;
    gasSpeciesCheck = g2.species();
    vector<string> gasSpecies;
    for( int r = 0; r<5; r++ ){
      gasSpecies.push_back( species_name( g2.species()[r] ) );
    }

    std::vector<double> myMoleFracs (5);
    myMoleFracs[H2O]=0.1;
    myMoleFracs[CO2]=0.4;
    myMoleFracs[OH ]=0.05;
    myMoleFracs[CO ]=0.15;
    myMoleFracs[NO ]=0.3;

    double planckG1;
    double rossG1;
    double effG1;
    g1.mixture_coeffs(planckG1, rossG1, effG1, myMoleFracs,  1050.0);
    double planckG2;
    double rossG2;
    double effG2;
    g2.mixture_coeffs(planckG2, rossG2, effG2, myMoleFracs,  1050.0);

    double plankErr = std::abs(planckG1 - planckG2);
    double rossErr  = std::abs(rossG1   - rossG2  );
    double effErr   = std::abs(effG1    - effG2   );

    status( plankErr < 1.7e-7  && plankErr/planckG1 < 1.8e-6, "Planck    abs coef I/O" );
    status( rossErr  < 1.0e-13 && rossErr /rossG1   < 9.2e-7, "Rosseland abs coef I/O" );
    status( effErr   < 1.1e-8  && effErr  /effG1    < 1.3e-7, "effective abs coef I/O" );

    // expected values:
    const double plank = 0.0941019  ;
    const double ross  = 1.09386e-07;
    const double eff   = 0.0869219  ;

    plankErr = std::abs(planckG1 - plank);
    rossErr  = std::abs(rossG1   - ross );
    effErr   = std::abs(effG1    - eff  );

//    std::cout << planckG1 << "\t" << plankErr << "\t" << plankErr/plank << endl
//              << rossG1   << "\t" << rossErr  << "\t" << rossErr /ross  << endl
//              << effG1    << "\t" << effErr   << "\t" << effErr  /eff   << endl;

    status( plankErr < 3.5e-8  && plankErr/plank < 3.7e-7, "Planck    abs coef value" );
    status( rossErr  < 2.7e-13 && rossErr /ross  < 2.5e-6, "Rosseland abs coef value" );
    status( effErr   < 4.8e-9  && effErr  /eff   < 5.5e-8, "effective abs coef value" );

//    cout << "               Original        Reloaded        Expected\n"
//         << "----------+----------------------------------------------\n"
//         << "Planck    |" << setw(15) << planckG1 << setw(15) << planckG2 << setw(15) << plank<< endl
//         << "Rosseland |" << setw(15) << rossG1   << setw(15) << rossG2   << setw(15) << ross << endl
//         << "Effective |" << setw(15) << effG1    << setw(15) << effG2    << setw(15) << eff  << endl;

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
