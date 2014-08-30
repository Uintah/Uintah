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

#include <PhilCoalTableImporter.h>
#include <tabprops/StateTable.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ios>
#include <stdexcept>

//--------------------------------------------------------------------

PhilCoalTableReader::PhilCoalTableReader( const std::string fileName )
  : fileName_( fileName )
{
  read_table();
  build_table();
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

  std::ifstream inFile( fileName_.c_str(),std::ios::in );
  std::string line;

  // skip the header
  for( size_t i=0; i<nskip; ++i )  std::getline(inFile,line);

  nIndepVars_ = -1;
  inFile >> nIndepVars_;

  if( nIndepVars_ < 1 ){
    std::ostringstream msg;
    msg << "ERROR : invalid number of independent variables (" << nIndepVars_ << ")" << std::endl;
    throw std::runtime_error( msg.str() );
  }
  indepVarNames_.resize(nIndepVars_,"");
  nptsIvar_.resize(nIndepVars_,0);

  for( size_t i=0; i<nIndepVars_; ++i )  inFile >> indepVarNames_[i];
  for( size_t i=0; i<nIndepVars_; ++i )  inFile >> nptsIvar_[i];

  std::cout << "Indep. Variable    # Points" << std::endl;
  for( size_t i=0; i<nIndepVars_; ++i ){
    std::cout << std::setw(25) << std::left << indepVarNames_[i]
              << nptsIvar_[i] << std::endl;
  }

  // pure stream enthalpies - discard for now.
  //   getline(inFile,line);
  //   getline(inFile,line);

  inFile >> nDepVars_;
  depVarNames_.resize(nDepVars_,"");
  for( size_t i=0; i<nDepVars_; ++i )  inFile >> depVarNames_[i];

  getline(inFile,line);  // the rest of the previous line
  getline(inFile,line);  // units of depvars
  getline(inFile,line);  // blank line

  ivarGrid_.resize( nIndepVars_, NULL );

  for( size_t i=nIndepVars_; i>0; --i ){
    std::vector<double>* const ivarMesh = new std::vector<double>( nptsIvar_[i-1], 0.0 );
    std::vector<double>& varmesh = *ivarMesh;
    for( size_t j=0; j<nptsIvar_[i-1]; ++j ){
      inFile >> varmesh[j];
    }
    ivarGrid_[i-1] = ivarMesh;
  }

  int nptstot = 1;
  for( size_t i=0; i<nIndepVars_; ++i ) nptstot *= nptsIvar_[i];

  // load the dependent variables
  for( size_t j=0; j<nDepVars_; ++j ){
    if( j>0 ){
      double tmp;
      for( size_t j=0; j<nptsIvar_[0]; ++j ) inFile >> tmp;
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

void
PhilCoalTableReader::build_table( const std::string fname )
{
  const bool clip = true;
  const int order = 1;

  StateTable table( nDepVars_ );
  table.name( fname );

  // insert the metadata
  for( MetaData::const_iterator ii=metaData_.begin(); ii!=metaData_.end(); ++ii ){
    table.add_metadata( ii->first, ii->second );
  }

  for( size_t idv=0; idv<nDepVars_; ++idv ){
    std::cout << "Processing entry for '" << depVarNames_[idv] << "'" << std::endl;
    InterpT* interp = NULL;
    switch (nIndepVars_){
    case 1:
      interp = new Interp1D( order, *ivarGrid_[0], *dvars_[idv], clip );
      break;
    case 2:
      interp = new Interp2D( order, *ivarGrid_[0], *ivarGrid_[1], *dvars_[idv], clip );
      break;
    case 3:
      interp = new Interp3D( order, *ivarGrid_[0], *ivarGrid_[1], *ivarGrid_[2], *dvars_[idv], clip );
      break;
    case 4:
      interp = new Interp4D( order, *ivarGrid_[0], *ivarGrid_[1], *ivarGrid_[2], *ivarGrid_[3], *dvars_[idv], clip );
      break;
    case 5:
      interp = new Interp5D( order, *ivarGrid_[0], *ivarGrid_[1], *ivarGrid_[2], *ivarGrid_[3], *ivarGrid_[4], *dvars_[idv], clip );
      break;
    }
    table.add_entry( depVarNames_[idv], interp, indepVarNames_ );
  }

  table.write_table(fname);
}

//------------------------------------------------------------------
