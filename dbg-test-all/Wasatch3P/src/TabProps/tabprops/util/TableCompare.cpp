/*
 * The MIT License
 *
 * Copyright (c) 2010-2014 The University of Utah
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
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iomanip>

#include <tabprops/StateTable.h>
#include <tabprops/Archive.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// jcs for some reason the serialization doesn't work without this:
Interp1D i1d;
Interp2D i2d;
Interp3D i3d;
Interp4D i4d;
Interp5D i5d;

using namespace std;

int main( int argc, const char* argv[] )
{
  string tname1, tname2;

  // parse the command line options
  try{
    po::options_description desc("Supported Options");
    desc.add_options()
          ( "help", "print help message" );

    po::options_description hidden("Hidden options");
    hidden.add_options()
            ("file1", po::value<string>(&tname1), "file1")
            ("file2", po::value<string>(&tname2), "file2");

    po::positional_options_description p;
    p.add("file1", 1);
    p.add("file2", 1);

    po::options_description cmdline_options;
    cmdline_options.add(desc).add(hidden);

    po::variables_map args;
    po::store( po::command_line_parser(argc,argv).
               options(cmdline_options).positional(p).run(), args );
    po::notify(args);

    if( args.count("help") ){
      cout << "Usage:  tablecompare <file1> <file2>\n"
          << desc << "\n";
      return -1;
    }

    if( tname1.empty() || tname2.empty() ){
      cout << "Must provide names of input files for comparison!" << endl << desc << endl;
      return -1;
    }
  }
  catch( std::exception& err ){
    cout << err.what() << "\n\nUsage:  tablecompare <file1> <file2>\n";
    return -1;
  }

  cout << "Comparing:\n\t" << tname1 << "\n\t" << tname2 << "\n\n";

  StateTable tbl1, tbl2;

  try{
    tbl1.read_table( tname1 );
    tbl2.read_table( tname2 );
    if( tbl1 == tbl2 ){
      cout << endl << "+++ tables are identical +++" << endl << endl;
      return 0;
    }
    cout << "----------------------------------------------------------------------------------------------------" << endl
         << left << setw(25) << "Table Name" << setw(35) << "Code base date" << setw(40) << "Code base hash" << endl
         << "----------------------------------------------------------------------------------------------------" << endl
         << setw(25) << tname1 << setw(35) << tbl1.repo_date() << setw(40) << tbl1.repo_hash() << endl
         << setw(25) << tname2 << setw(35) << tbl2.repo_date() << setw(40) << tbl2.repo_hash() << endl
         << "----------------------------------------------------------------------------------------------------" << endl
         << endl;
  }

  catch( exception& e ){
    cout << endl << "FAILURE : " << e.what() << endl
         << " This likely occurred because one of the tables you tried" << endl
         << " loading could not be imported (unserialized)." << endl;
    return -1;
  }
  catch( ... ){
    cout << "An unknown error occurred" << endl;
    return -1;
  }

  bool isFailed = false;

  //___________________________
  // Check table dimension
  if( tbl1.get_ndim() != tbl2.get_ndim() ){
    cout << "Number of independent variables are different!" << endl
         << " There are " << tbl1.get_ndim() << " independent variables in " << tname1 << endl
         << " There are " << tbl2.get_ndim() << " independent variables in " << tname2 << endl
         << endl;
    isFailed = true;
  }

  //_________________________________
  // check independent variable names
  if( !isFailed ){
    {
      const vector<string>& ivarNames = tbl1.get_indepvar_names();
      for( vector<string>::const_iterator ivn=ivarNames.begin(); ivn!=ivarNames.end(); ++ivn ){
        if( !tbl2.has_indepvar( *ivn ) ){
          isFailed = true;
          cout << "Independent variable: '" << *ivn << "' is in " << tname1 << " but not in " << tname2 << endl;
        }
      }
    }
    {
      const vector<string>& ivarNames = tbl2.get_indepvar_names();
      for( vector<string>::const_iterator ivn=ivarNames.begin(); ivn!=ivarNames.end(); ++ivn ){
        if( !tbl1.has_indepvar( *ivn ) ){
          isFailed = true;
          cout << "Independent variable: '" << *ivn << "' is in " << tname2 << " but not in " << tname1 << endl;
        }
      }
    }
  }

  //_______________________________
  // check dependent variables
  if( !isFailed ){
    const vector<string>& dvarNames = tbl1.get_depvar_names();
    for( vector<string>::const_iterator idvn=dvarNames.begin(); idvn!=dvarNames.end(); ++idvn ){
      const InterpT* bs1 = tbl1.find_entry( *idvn );
      const InterpT* bs2 = tbl2.find_entry( *idvn );
      if( !bs2 ){
        isFailed = true;
        cout << "Entry for '" << *idvn  << "' in " << tname1 << " has no corresponding entry in " << tname2 << endl;
      }
      else if( *bs1 != *bs2 ){
        isFailed = true;
        cout << "Interpolant for entry '" << *idvn << "' in " << tname1 << " differs from that in " << tname2 << endl;
      }
    }
  }

  if( isFailed ){
    cout << endl << "--- TABLES ARE DIFFERENT ---" << endl << endl;
    return -1;
  }

  cout << endl << "+++ tables are identical +++" << endl << endl;
  return 0;
}
