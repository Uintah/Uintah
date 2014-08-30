/*
 * PreProcessor.cpp
 *
 *  Created on: Nov 30, 2012
 *      Author: "James C. Sutherland"
 *
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

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/foreach.hpp>

#include <iostream>
#include <vector>
#include <string>

#include "RadiativeSpecies.h"
#include "AbsCoeffGas.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;

int main( int iarg, char* carg[] )
{
  bool greyGas = false, fsk = false;
  double greyGasOPL;
  vector<string> specNameList;
  vector<RadiativeSpecies> specList;
  string greyGasOutputFileName, fskOutputFileName, dataPath;

  // parse the command line options input describing the problem
  try{
    po::options_description desc("Supported Options");
    desc.add_options()
           ( "help", "print help message" )
           ( "grey-gas", "preprocess grey gas tables" )
           ( "grey-gas-output-name", po::value<string>(&greyGasOutputFileName)->default_value("GreyGasProperties.txt"),"name for grey gas table output file" )
           ( "grey-gas-opl", po::value<double>(&greyGasOPL)->default_value(0.01),"Optical path length (cm) for calculating the mean absorption coefficient.")
           ( "fsk", "preprocess FSK tables" )
           ( "fsk-output-name", po::value<string>(&fskOutputFileName)->default_value("FSKProperties.txt"),"name for FSK table output file" )
           ( "species",po::value< vector<string> >(&specNameList)->multitoken(), "Which species to include.  Example: --species OH CO2 H2O")
           ( "data-dir", po::value<string>(&dataPath)->default_value("."), "Directory to search for the AbsCoef*.txt data files" )
           ;

    po::variables_map args;
    po::store( po::parse_command_line(iarg,carg,desc), args );
    po::notify(args);

    if( args.count("grey-gas") ) greyGas=true;
    if( args.count("fsk")      ) fsk = true;

    if (args.count("help")) {
      cout << endl
          << "The radprops_prepro utility provides preprocessing for grey gas and FSK" << endl
          << "radiation models to generate tables that are faster for use in property" << endl
          << "evaluation schemes." << endl << endl
          << desc << endl << endl;
      return 1;
    }

    BOOST_FOREACH( string& spec, specNameList ){
      specList.push_back( species_enum(spec) );
    }

    if( specNameList.size() == 0 ){
      cout << endl << endl
           << "You must specify the species to be included for processing." << endl
           << "They should be associated with files that have the form:" << endl
           << "   AbsCoeff<speciesname>T<temperature>" << endl << endl
           << desc << endl;
      return -1;
    }
    cout << endl << "Processing requested for the following species:" << endl;
    BOOST_FOREACH( RadiativeSpecies spec, specList ){
      cout << "  -> " << species_name(spec) << endl;
    }
    cout << endl;
  }
  catch( std::exception& err ){
    cout << "ERROR while parsing command line arguments!" << endl << endl
         << err.what() << endl << endl;
    return -1;
  }

  if( greyGas ){
    cout << endl << "Preprocessing data for grey gas properties..." << endl;
    try{
      GreyGas grey( specList, greyGasOPL, greyGasOutputFileName, dataPath );
    }
    catch( std::exception& err ){
      cout << "Error while processing grey gas properties.  Details follow: " << endl
          << err.what() << endl << endl;
    }
  }

  if( fsk ){
    cout << endl << "Preprocessing data for FSK properties..." << endl;
    try{
      FSK fsk( specList, fskOutputFileName, "." );
    }
    catch( std::exception& err ){
      cout << "Error while processing grey gas properties.  Details follow: " << endl
          << err.what() << endl << endl;
    }
  }
}
