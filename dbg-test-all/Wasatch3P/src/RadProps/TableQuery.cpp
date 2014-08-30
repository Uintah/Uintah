/**
 *  \file   TableQuery.cpp
 *  \date   Dec 9, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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
 *
 */



#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/foreach.hpp>

#include <iostream>
#include <vector>
#include <string>

#include "RadiativeSpecies.h"
#include "AbsCoeffGas.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::string;

int main( int iarg, char* carg[] )
{
  string fname;
  bool doGreyGas=true;
  // parse the command line options input describing the problem
  po::options_description desc("Supported Options");
  try{
    desc.add_options()
           ( "help", "print help message" )
           ;

    po::options_description hidden("Hidden options");
    hidden.add_options()
      ("input-file", po::value<string>(&fname), "input file");
    po::positional_options_description p;
    p.add("input-file", -1);
    po::options_description cmdline_options;
    cmdline_options.add(desc).add(hidden);

    po::variables_map args;
    po::store( po::command_line_parser(iarg,carg).
               options(cmdline_options).positional(p).run(), args );
    po::notify(args);

    if (args.count("help")) {
      cout << desc << "\n";
      return 1;
    }
  }
  catch( std::exception& err ){
    cout << "ERROR while parsing command line arguments!" << endl << endl
         << err.what() << endl << endl
         << desc << "\n";
    return -1;
  }

  if( doGreyGas ){
    cout << "Processing " << fname << endl;
    GreyGas greyGas( fname );

    const std::vector<RadiativeSpecies>& species = greyGas.species();
    while(true){
      vector<double> ivar;
      cout << endl;
      BOOST_FOREACH( RadiativeSpecies sp, species ){
        cout << "Enter the mole fraction for " << species_name(sp) << ": ";
        double tmp;
        cin >> tmp;
        ivar.push_back(tmp);
      }
      double temperature;
      cout << "Enter the temperature (K): ";
      cin >> temperature;

      double planck, rosseland, effective;
      greyGas.mixture_coeffs( planck, rosseland, effective, ivar, temperature );

      cout << "\nAbsorbtion coefficients:\n"
          << "\tPlanck   : " << planck << endl
          << "\tRosseland: " << rosseland << endl
          << "\tEffective: " << effective << endl << endl;
    }
  }

  return 0;
}
