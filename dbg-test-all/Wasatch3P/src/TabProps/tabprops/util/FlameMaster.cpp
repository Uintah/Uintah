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
#include "FlameMaster.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#include <boost/algorithm/string.hpp> // case-insensitive comparison
#include <boost/algorithm/string/trim_all.hpp> // trim trailing and leading whitespace
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>

using boost::algorithm::iequals;  // case insensitive string comparison
using std::cout;
using std::endl;

// uncomment this to enable verbose output of file parsing
//#define PARSE_DIAGNOSTICS

//============================================================================

enum Block {
  NONE    = 0,
  HEADER  = 1,
  BODY    = 2,
  TRAILER = 3
};

Block get_block( const std::string& name )
{
  // Blocks in the FlameMaster file are tagged with a unique header.
  // Search for this header, indicating a block switch.  If found, set the
  // new block value and gobble up the line so that the block-specific
  // parsing code won't have to worry about this one special-case header
  // line.
  Block block( NONE );
  if     ( name == "header" ) block = HEADER;
  else if( name == "body"   ) block = BODY;
  else if( name == "trailer") block = TRAILER;
  return block;
}

//============================================================================

FMFlamelet::FMFlamelet( const std::string fileName, const int order )
: fileName_   ( fileName ),
  interpOrder_( order    )
{
  npts_ = 0;
  load_file( fileName );
  reorder(); // ensure that everything is ordered by increasing mixture fraction
}

//-------------------------------------------------------------------

void
FMFlamelet::reorder()
{
  // reorder the arrays to ensure that the mixture fraction is in ascending order
  VarEntry::const_iterator imf = extract_entry("Z");
  if( imf->second.back() - imf->second.front() > 0 ) return;  // already ascending

  BOOST_FOREACH( VarEntry::value_type& vt, varEntries_ ){
    std::vector<double>& data = vt.second;
    std::reverse( data.begin(), data.end() );
  }
}

//-------------------------------------------------------------------

void
FMFlamelet::load_file( const std::string& fileName )
{
  npts_ = 0;
  // open the file
  std::ifstream file( fileName.c_str(), std::ios::in );
  if( !file ){
    std::ostringstream msg;
    msg << "\nUnable to open FlameMaster file named:\n '" << fileName << "'\n\n";
    throw std::invalid_argument(msg.str());
  }

  Block currentBlock = NONE;

  typedef boost::tokenizer< boost::char_separator<char> > Token;
  boost::char_separator<char> eqSep("=");
  boost::char_separator<char> spaceSep(" ");
  boost::char_separator<char> tabSpaceSep(" \t");

  //-----------------------------
  // keep track of how many points we have parsed in each field to tell
  // when we are done.  This is used when parsing the "body" of the file
  int nptsParsed = 0;

  // since we parse a line at a time, this tells us if we are in the
  // middle of an entry or not
  bool expectingMore = false;

  std::string entryKey = "";
  //--------------------------------

# ifdef PARSE_DIAGNOSTICS
  cout << endl << endl << "Parsing file: " << fileName << endl;
# endif

  std::string line;
  while( std::getline(file,line) ){

    boost::algorithm::trim( line );

//#   ifdef PARSE_DIAGNOSTICS
//    std::cout << "Parsed line: <" << line << ">\n";
//#   endif

    const Block thisBlock = get_block(line);
    if( thisBlock != NONE ){
      currentBlock = thisBlock;
      continue;
    }

    switch( currentBlock ){

      case HEADER:{ // the appetizer.

        if( line.empty() ) continue;

        const Token token(line,eqSep); // break into LHS and RHS of '=' sign

        // determine how many pieces we got from this - we should only get two.
        size_t nEntries=0;
        for( Token::const_iterator i=token.begin(); i!=token.end(); ++i, ++nEntries ){}

        if( nEntries == 1 ) continue; // don't know what to do here...

        if( nEntries > 2 ){
          std::ostringstream msg;
          msg << __FILE__ << " : " << __LINE__ << "Expected one '=' sign but found " << nEntries-1 << std::endl;
          for( Token::const_iterator i=token.begin(); i!=token.end(); ++i ){
            msg << "\t(" << *i << ")\n";
          }
          throw std::runtime_error( msg.str() );
        }

        Token::const_iterator itok = token.begin();
        const std::string key = boost::algorithm::trim_all_copy( boost::lexical_cast<std::string>(*itok) );
        metaData_[key] = *(++itok);

        if( key == "gridPoints" ){
          npts_ = boost::lexical_cast<int>(  boost::algorithm::trim_all_copy(*itok) );
#         ifdef PARSE_DIAGNOSTICS
          std::cout << "found " << npts_ << " grid points\n";
#         endif
        }

        break;
      }

      case BODY:{ // the main course.

        assert( npts_ > 0 );

        if( !expectingMore ){ // new entry
          entryKey = boost::algorithm::trim_all_copy( line );
#         ifdef PARSE_DIAGNOSTICS
          std::cout << "Reading information for grid variable: '" << entryKey << "'\n";
#         endif
          expectingMore = true;
          continue; // all done here.
        }

        const Token token(line,tabSpaceSep); // break line into pieces separated by tabs

        std::vector<double>& vals = varEntries_[entryKey];
        for( Token::const_iterator itok = token.begin(); itok != token.end(); ++itok, ++nptsParsed ){
#         ifdef PARSE_DIAGNOSTICS
          std::cout << "\t(" << *itok << ")\n";
#         endif
          vals.push_back( boost::lexical_cast<double>(  boost::algorithm::trim_all_copy(*itok) ) );
        }

        expectingMore = ( nptsParsed < npts_ );

        if( !expectingMore ){
#         ifdef PARSE_DIAGNOSTICS
          std::cout << "Finished reading " << nptsParsed << " points for " << entryKey << std::endl;
#         endif
          nptsParsed = 0;
        }

        break;
      } // case BODY

      case TRAILER:{
        // we aren't parsing the trailer at the moment - we are done here.
        break;
      }
      case NONE:{
        throw std::runtime_error("Invalid parsing state while reading flamelet file");
      }
    }

  }
# ifdef PARSE_DIAGNOSTICS
  dump_info( cout );
# endif
}

//-------------------------------------------------------------------

void FMFlamelet::dump_info( std::ostream& os ) const
{
  cout << endl
       << "___________________________________________________" << endl
       << "SUMMARY OF INFORMATION PARSED FROM: " << fileName_ << endl << endl
       << "Usng interpolation order: " << interpOrder_ << endl
       << "Grid has " << npts_ << " points" << endl << endl
       << "Grid variables follow\n";

  BOOST_FOREACH( const VarEntry::value_type& vt, varEntries_ ){
    cout << "\t" << vt.first << endl;
  }

  cout << "\nMeta-Data follows\n";
  BOOST_FOREACH( const MetaData::value_type& vt, metaData_ ){
    cout << "\t" << std::setw(27) << std::left << vt.first << " = " << vt.second << endl;
  }
  cout << "___________________________________________________" << endl << endl;
}

//-------------------------------------------------------------------

FMFlamelet::VarEntry::const_iterator
FMFlamelet::extract_entry( const std::string& var ) const
{
  VarEntry::const_iterator ivar = varEntries_.find( var );
  if( ivar == varEntries_.end() ){
    std::ostringstream msg;
    msg << endl << "ERROR from " << __FILE__ << " : " << __LINE__ << endl << endl
        << "Could not find requested variable:\n\t'" << var << "'" << endl
        << "in flamelet file:\n\t" << fileName_ << endl << endl;
    throw std::invalid_argument( msg.str() );
  }
  return ivar;
}

//-------------------------------------------------------------------

Interp1D
FMFlamelet::interpolant( const std::string& entry,
                         const std::string indepVarName,
                         const bool allowClipping ) const
{
  const VarEntry::const_iterator ivar = extract_entry( indepVarName );
  const VarEntry::const_iterator dvar = extract_entry( entry        );
  Interp1D interp( interpOrder_, ivar->second, dvar->second, allowClipping );
  return interp;
}

//-------------------------------------------------------------------

std::vector<std::string>
FMFlamelet::table_entries() const
{
  std::vector<std::string> entries;
  BOOST_FOREACH( const VarEntry::value_type& vt, varEntries_ ){
    entries.push_back( vt.first );
  }
  return entries;
}

//-------------------------------------------------------------------

const std::vector<double>&
FMFlamelet::variable_values( const std::string varName ) const
{
  return extract_entry(varName)->second;
}

//-------------------------------------------------------------------

template<typename T>
T
FMFlamelet::extract_metadata_value( const std::string key ) const
{
  MetaData::const_iterator imd = metaData_.find( key );
  if( imd == metaData_.end() ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl
        << "Metadata entry for " << key << " was not found\n";
    throw std::invalid_argument(msg.str());
  }

  // metadata frequently incorporates units. In that case, space-separate them
  // and return the first entry, assuming that it will be the value we want.
  const boost::char_separator<char> spaceSep(" ");
  const boost::tokenizer< boost::char_separator<char> > token(imd->second,spaceSep);
  return boost::lexical_cast<T>( *token.begin() );
}

template<>
std::string
FMFlamelet::extract_metadata_value( const std::string key ) const
{
  MetaData::const_iterator imd = metaData_.find( key );
  if( imd == metaData_.end() ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl
        << "Metadata entry for " << key << " was not found\n";
    throw std::invalid_argument(msg.str());
  }
  return imd->second;
}

// explicit instantiation of this function template for supported types
#define DECLARE_EMV(T) \
template T FMFlamelet::extract_metadata_value<T>(const std::string key) const;
DECLARE_EMV(double)
DECLARE_EMV(float)
DECLARE_EMV(int)
DECLARE_EMV(short)
DECLARE_EMV(size_t)
DECLARE_EMV(std::string)

//===================================================================

FlameMasterLibrary::FlameMasterLibrary( const std::string path,
                                        const std::string filePattern,
                                        const int order,
                                        const bool allowClipping )
: interpOrder_( order ),
  allowClipping_( allowClipping ),
  mixfrac_( "Z" ),
  dissipRate_( "stoichScalarDissRateBilger" ),
  table_( 2 )  // (z,chi) are indep. vars.
{
  hasGeneratedTable_ = false;
  load_files( path, filePattern );
}

//-------------------------------------------------------------------

FlameMasterLibrary::~FlameMasterLibrary()
{
  BOOST_FOREACH( FMFlamelet* f, library_ ) delete f;
}

//-------------------------------------------------------------------

void
FlameMasterLibrary::load_files( const std::string& path,
                                const std::string& pattern )
{
  const boost::regex filter( pattern+".*" );  // the ".*" is a wildcard matching operation.

  try{
    boost::filesystem::directory_iterator end;
    for( boost::filesystem::directory_iterator i(path); i!=end; ++i ){

      if( !boost::filesystem::is_regular_file( i->status() ) ) continue; // only look at regular files (not dirs)

      const std::string fname = i->path().filename().string();

      // Skip if no match
      boost::smatch what;
      if( !boost::regex_match( fname, what, filter ) ) continue;

      // File matches. parse it
      cout << "Loading FlameMaster file: " << fname << endl;
      library_.push_back( new FMFlamelet( path+"/"+fname, interpOrder_ ) );  // jcs need to use i->string() but cannot get that to compile.
    }
  }
  catch( std::exception& err ){
    std::ostringstream msg;
    msg << "ERROR loading flamelet library.  Details follow:\n" << err.what();
    throw std::runtime_error( msg.str() );
  }
}

//-------------------------------------------------------------------

StateTable&
FlameMasterLibrary::generate_table( const std::string& filePrefix )
{
  if( hasGeneratedTable_ ) return table_;

  typedef std::vector<double> Vec;
  typedef std::vector<std::string> SVec;

  // get the Z-grid off of the first flamelet
  const Vec& zVals = library_[0]->variable_values( mixfrac_ );

  Vec chiVals;
  double chiLast=-1e99;
  BOOST_FOREACH( const FMFlamelet* const flm, library_ ){

    const double chi = flm->extract_metadata_value<double>(dissipRate_);
    chiVals.push_back( chi );

    // currently we will assume that the dissipation rate is reflected in the
    // file name such that it is sorted.  Otherwise we need to sort the flamelets
    // by dissipation rate.
    // \todo implement a sort.
    if( chi < chiLast ){ // not sorted.
      std::ostringstream msg;
      msg << endl << endl
          << __FILE__ << " : " << __LINE__ << endl << endl
          << "ERROR: dissipation rate is not sorted in ascending order." << endl << endl;
      throw std::runtime_error(msg.str());
    }
    chiLast = chi;

  }

  SVec indepVarNames;
  indepVarNames.push_back(mixfrac_);
  indepVarNames.push_back(dissipRate_);

  const SVec depVarNames = library_[0]->table_entries();

  BOOST_FOREACH( const std::string& varName, depVarNames ){

    Vec varValues; // store values into this buffer to build the final interpolant

    // extract values for each flamelet into this buffer
    BOOST_FOREACH( const FMFlamelet* const flm, library_ ){
      const Interp1D interp = flm->interpolant( varName, mixfrac_, allowClipping_ );

      BOOST_FOREACH( const double x, zVals ){
        varValues.push_back( interp.value(x) );
      }

    } // loop over flamelets (chi space)

    InterpT* interp2 = new Interp2D( interpOrder_, zVals, chiVals, varValues, allowClipping_ );
    table_.add_entry( varName, interp2, indepVarNames, false );
  }
  hasGeneratedTable_ = true;
  return table_;
}

//============================================================================

