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

#include "PhilCoalTableImporter.h"

#include <tabprops/StateTable.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ios>
#include <stdexcept>
#include <cmath>
#include <algorithm>

//--------------------------------------------------------------------

PhilCoalTableReader::PhilCoalTableReader( const std::string inputFilename,
                                          const std::string outputFileName,
                                          const bool subSampleFirstDim )
  : order_( 1 ),
    fileName_( inputFilename ),
    subSampleFirstDim_( subSampleFirstDim )
{
  try{
    read_table();
  }
  catch( std::exception& err ){
    std::ostringstream msg;
    msg << err.what() << std::endl
        << "Error reading 'phil' formatted table from file: '" << inputFilename << "'" << std::endl;
    throw std::runtime_error( msg.str() );
  }
  try{
    build_table(outputFileName);
  }
  catch( std::exception& err ){
    std::ostringstream msg;
    msg << err.what() << std::endl
        << "Error writing coal table to '" << outputFileName << std::endl;
    throw std::runtime_error( msg.str() );
  }
}

//--------------------------------------------------------------------

PhilCoalTableReader::~PhilCoalTableReader()
{
  for( size_t i=0; i<nIndepVars_; ++i )  delete ivarGrid_[i];
  for( size_t i=0; i<nDepVars_;   ++i )  delete dvars_[i];
}

//--------------------------------------------------------------------

void
PhilCoalTableReader::read_table()
{
  int nskip = extract_header();
  nskip += extract_metadata( nskip );
  nskip += set_mesh( nskip );

  std::ifstream inFile( fileName_.c_str(),std::ios::in );
  std::string line;

  // skip the header stuff
  for( size_t i=0; i<nskip; ++i )  std::getline(inFile,line);

  int nptstot = 1;
  for( size_t i=0; i<nIndepVars_; ++i ) nptstot *= nptsIvar_[i];

  std::vector<double> varmesh0(n0_);

  // load the dependent variables
  for( size_t idvar=0; idvar<nDepVars_; ++idvar ){

    std::cout << "Loading '" << depVarNames_[idvar] << "' from disk" << std::endl;

    // for the third independent variable, the grid spacing varies.
    // We interpolate each entry onto a fixed mesh.
    if( nIndepVars_ == 3 ){
      int ipt = 0;
      std::vector<double>* const dvar = new std::vector<double>(nptstot,0.0);
      for( size_t iblock=0; iblock<nptsIvar_[2]; ++iblock ){

        for( size_t k=0; k<n0_; ++k ){
          inFile >> varmesh0[k];
        }

        std::vector<double> tmp(n0_,0.0);
        for( size_t i1=0; i1<nptsIvar_[1]; ++i1 ){
          for( size_t i0=0; i0<n0_; ++i0 ){
            inFile >> tmp[i0];
          }

          // interpolate this "row" in the table (tmp as a function of
          // varmesh0) and then interpolate it to the new mesh that we
          // are using (ivarGrid_)
          Interp1D sp( order_, varmesh0, tmp, true );

          for( size_t i0=0; i0<nptsIvar_[0]; ++i0 ){
            const double& xx = (*ivarGrid_[0])[i0];
            (*dvar)[ipt++] = sp.value(xx);
          }
        }
      }
      dvars_.push_back( dvar );

    }
    else{
      if( idvar>0 ){
        double tmp;
        for( size_t j=0; j<nptsIvar_[0]; ++j ){
          inFile >> tmp;
          std::cout << tmp << ", ";
        }
        std::cout << std::endl;
      }
      std::vector<double>* const dvar = new std::vector<double>(nptstot);
      std::vector<double>& dv = *dvar;
      int ipt = 0;
      for( size_t ipt=0; ipt<nptstot; ++ipt ){
        inFile >> dv[ipt];
      }
      dvars_.push_back( dvar );
    }
  }
}

//--------------------------------------------------------------------

int
PhilCoalTableReader::extract_header() const
{
  std::ifstream inFile( fileName_.c_str(), std::ios::in );

  if( inFile.bad() ){
    std::ostringstream msg;
    msg << "Error reading file '" << fileName_ << "'" << std::endl
        << "File: " << __FILE__ << " : " << __LINE__  << std::endl;
    throw std::runtime_error( msg.str() );
  }

  // header is delineated by "#"
  std::string line;
  std::getline(inFile,line);
  int nskip=0;
  while(line.find("#",0) != line.npos){
    ++nskip;
    std::cout << line << std::endl;
    std::getline(inFile,line);
  }
  return nskip;
}

//------------------------------------------------------------------

int
PhilCoalTableReader::extract_metadata( const int nskip )
{
  std::ifstream inFile( fileName_.c_str(), std::ios::in );

  metaData_.clear();

  std::string line;

  // skip the header
  for( size_t i=0; i<nskip; ++i )  std::getline(inFile,line);

  // look for "metadata:" keywords
  bool more = true;
  while( more ){
    std::getline( inFile, line );
    // remove white space
    std::string::size_type whiteLoc = line.find(" ");
    while( whiteLoc != std::string::npos ){
      line.replace( whiteLoc, 1, "" );
      whiteLoc = line.find(" ");
    }
    std::string::size_type mdpos = line.find("metadata:");
    std::string::size_type eqpos = line.find("=");
    if( mdpos != std::string::npos  && eqpos != std::string::npos ){
      // read the name
      std::string name = line.substr( 9, eqpos-9 );
      std::stringstream val( std::string( line.begin()+eqpos+1, line.end() ) );
      double value=0.0;
      val >> value;
      metaData_[ name ] = value;
      std::cout << "inserted metadata:  " << name << " = " << value << std::endl;
    }
    else{
      more = false;
    }
  }
  return metaData_.size();
}

//------------------------------------------------------------------

int
PhilCoalTableReader::set_mesh( const int nskip )
{
  int nlines = 0;

  std::ifstream inFile( fileName_.c_str(), std::ios::in );

  // skip the header stuff
  std::string line;
  for( size_t i=0; i<nskip; ++i )  std::getline(inFile,line);

  nIndepVars_ = -1;
  inFile >> nIndepVars_;  ++nlines;

  if( nIndepVars_ < 1 ){
    std::ostringstream msg;
    msg << "ERROR : invalid number of independent variables (" << nIndepVars_ << ")" << std::endl;
    throw std::runtime_error( msg.str() );
  }
  if( nIndepVars_ == 3 ){
    if( subSampleFirstDim_ ){
      std::cout << "NOTE: The first dimension in the table will be subsampled at a uniform" << std::endl
                << "      spacing with twice the number of points in the original table." << std::endl
                << "      This may result in inaccurate interpolation." << std::endl
                << std::endl;
    }
    else{
      std::cout << "NOTE: The first specification of the mesh in the first independent variable will" << std::endl
                << "      be used everywhere in the output table.  If this mesh varies through the" << std::endl
                << "      the original table, then some interpolation errors could result." << std::endl
                << "      We will interpolate the data from the original table onto the mesh and then fit that." << std::endl
                << std::endl;
    }
  }

  indepVarNames_.resize(nIndepVars_,"");
  nptsIvar_.resize(nIndepVars_,0);

  for( size_t i=0; i<nIndepVars_; ++i )  inFile >> indepVarNames_[i];
  for( size_t i=0; i<nIndepVars_; ++i )  inFile >> nptsIvar_[i];
  nlines += 2;

  std::cout << "Indep. Variable    # Points" << std::endl;
  for( size_t i=0; i<nIndepVars_; ++i ){
    std::cout << std::setw(25) << std::left << indepVarNames_[i]
              << nptsIvar_[i] << std::endl;
  }

  inFile >> nDepVars_;  ++nlines;
  depVarNames_.resize(nDepVars_,"");
  for( size_t i=0; i<nDepVars_; ++i )  inFile >> depVarNames_[i];

  ++nlines; getline(inFile,line);  // the rest of the previous line
  ++nlines; getline(inFile,line);  // units of depvars
  ++nlines; getline(inFile,line);  // blank line

  ivarGrid_.resize( nIndepVars_, NULL );

  const int nlo = (nIndepVars_==3) ? 1 : 0;
  for( int i=nIndepVars_-1; i>=nlo; --i ){
    std::vector<double>* const ivarMesh = new std::vector<double>( nptsIvar_[i], 0.0 );
    std::vector<double>& varmesh = *ivarMesh;
    std::cout << indepVarNames_[i] << ": ";
    for( size_t j=0; j<nptsIvar_[i]; ++j ){
      inFile >> varmesh[j];
      std::cout << varmesh[j] << ", ";
    }
    ++nlines;
    std::cout << std::endl;
    ivarGrid_[i] = ivarMesh;
  }

  if( nIndepVars_ == 3 ){
    n0_ = nptsIvar_[0];
    std::vector<double> varmesh0(nptsIvar_[0]);
    for( size_t k=0; k<nptsIvar_[0]; ++k ){
      inFile >> varmesh0[k];
    }

    // read the first grid for the first independent variable and use
    // this to set the mesh in the first dim
    if( subSampleFirstDim_ ){
      // find the smallest spacing and create a mesh based on that.
      // All entries will be interpolated onto this fine mesh.
      double dx = 1e99;
      for( size_t i=1; i<varmesh0.size(); ++i )  dx = std::min( dx, varmesh0[i]-varmesh0[i-1] );
      if( dx <= 0.0 ){
        std::ostringstream msg;
        msg << "invalid spacing in file " << fileName_ << "  (" << dx << ")"
            << std::endl
            << "  " << __FILE__ << " : " << __LINE__
            << std::endl;
        throw std::runtime_error( msg.str() );
      }
      const int npts = 1 + ( *std::max_element( varmesh0.begin(), varmesh0.end() ) -
                             *std::min_element( varmesh0.begin(), varmesh0.end() ) ) / dx;
      nptsIvar_[0] = npts;
      std::vector<double>* varmesh = new std::vector<double>( nptsIvar_[0], 0.0 );
      for( size_t i=0; i<nptsIvar_[0]; ++i ){
        (*varmesh)[i] = dx*i;
      }
      ivarGrid_[0] = varmesh;
    }
    else{
      // use the first grid specification found in the file for all entries.
      // subsequent grids will be interpolated back to this one.
      std::vector<double>* varmesh = new std::vector<double>( nptsIvar_[0], 0.0 );
      std::copy( varmesh0.begin(), varmesh0.end(), varmesh->begin() );
      ivarGrid_[0] = varmesh;
    }
  }

  return nlines;
}

//------------------------------------------------------------------

void
PhilCoalTableReader::build_table( const std::string fname )
{
  const bool clip = true;
  const int order = 3;  // third order polynomials...

  StateTable table( nDepVars_ );

  // insert the metadata
  for( MetaData::const_iterator ii=metaData_.begin(); ii!=metaData_.end(); ++ii ){
    table.add_metadata( ii->first, ii->second );
  }

  for( size_t idv=0; idv<nDepVars_; ++idv ){
    std::cout << "Adding '" << depVarNames_[idv] << "' to the table" << std::endl;
    InterpT* interp = NULL;
    switch (nIndepVars_){
    case 1: interp = new Interp1D( order, *ivarGrid_[0],                                              *dvars_[idv] ); break;
    case 2: interp = new Interp2D( order, *ivarGrid_[0], *ivarGrid_[1],                               *dvars_[idv] ); break;
    case 3: interp = new Interp3D( order, *ivarGrid_[0], *ivarGrid_[1], *ivarGrid_[2],                *dvars_[idv] ); break;
    case 4: interp = new Interp4D( order, *ivarGrid_[0], *ivarGrid_[1], *ivarGrid_[2], *ivarGrid_[3], *dvars_[idv] ); break;
    default:{
      std::ostringstream errmsg;
      errmsg << "ERROR: unsupported dimension for interpolant creation!"
             << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( errmsg.str() );
    }
    } // switch
    table.add_entry( depVarNames_[idv], interp, indepVarNames_ );
  }

  table.write_table( fname+".tbl" );
  verify_table( table );
}

//------------------------------------------------------------------

void
PhilCoalTableReader::verify_table( const StateTable& table )
{
  using std::cout;
  using std::endl;

  if( nIndepVars_ != 3 ) return;

  for( size_t idvar=0; idvar<nDepVars_; ++idvar ){

    const std::string& name = depVarNames_[idvar];
    const InterpT* const interp = table.find_entry(name);

    const std::vector<double>& dvar = *dvars_[idvar];

    std::vector<double> ivarPt( nIndepVars_, 0.0 );

    // cycle through independent variables
    size_t ipt=0;
    for( size_t k=0; k<nptsIvar_[2]; ++k ){
      ivarPt[2] = (*ivarGrid_[2])[k];
      for( size_t j=0; j<nptsIvar_[1]; ++j ){
        ivarPt[1] = (*ivarGrid_[1])[j];
        for( size_t i=0; i<nptsIvar_[0]; ++i ){
          ivarPt[0] = (*ivarGrid_[0])[i];

          const double val = interp->value( &ivarPt[0] );
          const double valtbl = dvar[ipt];
          const double tol = 1e-10;
          const double err = std::abs( valtbl - val );
          const double relerr = err/(tol+valtbl);
          if( relerr>tol && err>1e-9 ){
            cout << name << " : xpt: [" << ivarPt[0];
            for( size_t itmp=1; itmp<nIndepVars_; ++itmp )
              cout << "," << ivarPt[itmp];
            cout << "]" << std::flush;
              cout << " = " << val << "  (" << valtbl << ")  (relerr=" << relerr << ")  abserr=" << err << endl;
          }
          ++ipt;
        }
      }

    }

  }

}

//--------------------------------------------------------------------
