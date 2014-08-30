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

#include <vector>
#include <algorithm>
#include <cmath>

#include <sstream>
#include <iomanip>
#include <stdexcept>

//-- TabProps Includes --//
#include <tabprops/StateTable.h>
#include <tabprops/Archive.h>


//-- Boost serialization tools --//
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

using std::vector;
using std::string;

// these are for temporary diagnostics...
#include <iostream>
#include <fstream>
using std::cout; using std::endl;

//====================================================================

//--------------------------------------------------------------------

StateTable::StateTable( const int numDim )
  : gitRepoDate_( TabPropsVersionDate ),
    gitRepoHash_( TabPropsVersionHash )
{
  numDim_ = numDim;
}

//--------------------------------------------------------------------

StateTable::StateTable()
  : gitRepoDate_( TabPropsVersionDate ),
    gitRepoHash_( TabPropsVersionHash )
{}

//--------------------------------------------------------------------

StateTable::~StateTable()
{
  Table::const_iterator itbl;
  for( itbl=table_.begin(); itbl!=table_.end(); itbl++ )
    delete itbl->second;
}

//--------------------------------------------------------------------

void
StateTable::read_table( const std::string fileName )
{
  std::ifstream inFile( fileName.c_str(), std::ios_base::in );
  if( !inFile ){
    std::ostringstream msg;
    msg << "Cannot open StateTable: " << fileName << std::endl
        << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( msg.str() );
  }
  InputArchive ia(inFile);
  ia >> BOOST_SERIALIZATION_NVP( *this );
}

//--------------------------------------------------------------------

void
StateTable::write_table( const std::string fileName ) const
{
  std::ofstream outFile( fileName.c_str(), std::ios_base::out|std::ios_base::trunc );
  OutputArchive oa(outFile);
  oa << BOOST_SERIALIZATION_NVP( *this );
}

//--------------------------------------------------------------------

void
StateTable::add_entry( const string & name,
                       InterpT * const interp,
                       const vector<string> & indepVarNames,
                       const bool createCopy )
{
  // Is this the first entry?  If so, save off some information about the independent
  // variables.  We require all interpolants in a table to be functions of the same independent
  // variables.
  if( table_.size() == 0 ){
    indepVarNames_ = indepVarNames;
    numDim_ = indepVarNames.size();
  }

  // check that this interpolant is consistent with others in the table
  if( interp->get_dimension() != numDim_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Inconsistent dimensionality detected for entry '" << name << "'" << std::endl
           << "       Previous entries had " << numDim_ << " dimensions" << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  // disallow duplicates.
  const InterpT * const entry = find_entry( name );
  if( entry == NULL ){
    //cout << "adding entry: '" << name << "' to table." << endl;
    if( createCopy )  table_[name] = interp->clone();
    else              table_[name] = interp;
  }
  else{
    std::ostringstream errmsg;
    errmsg << "ERROR: Attempted to add a duplicate entry to the table." << std::endl
           << "       An entry with the name: '" << name << "' already exists." << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  // ensure that this interpolant's independent variable names match the others in the table.
  if( indepVarNames != indepVarNames_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: One or more independent variable names in interpolant '" << name << "'" << std::endl
           << "       do not match those already in the table." << std::endl
           << "       Inconsistent pairs follow: " << endl;
    vector<string>::const_iterator istr =indepVarNames_.begin();
    vector<string>::const_iterator istr2=indepVarNames.begin();
    for( ; istr!=indepVarNames_.end(); ++istr, ++istr2 ) {
      if( *istr != *istr2 )
        errmsg << "      ('" << *istr << "','" << *istr2 << "')" << std::endl;
    }
    errmsg << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
  // populate list of dependent variable names
  depVarNames_.push_back( name );
}
//--------------------------------------------------------------------
const InterpT *
StateTable::find_entry( const string & name ) const
{
  Table::const_iterator itbl = table_.begin();
  for( ; itbl!=table_.end(); itbl++ ){
    if( boost::algorithm::iequals(itbl->first, name) )
      return itbl->second;
  }
  return NULL;
}
//--------------------------------------------------------------------
InterpT *
StateTable::find_entry( const string & name )
{
  Table::iterator itbl = table_.begin();
  for( ; itbl!=table_.end(); ++itbl ){
    if( boost::algorithm::iequals( itbl->first, name ) )
      return itbl->second;
  }
  return NULL;
}
//--------------------------------------------------------------------
bool
StateTable::has_indepvar( const string & name ) const
{
  vector<string>::const_iterator istr;
  BOOST_FOREACH( const std::string& str, indepVarNames_ ){
    if( name == str ) return true;
  }
  return false;
}

//--------------------------------------------------------------------

double
StateTable::query( const InterpT * const interp,
                   const double * indepVars ) const
{
  if( interp == NULL )
    throw std::runtime_error("ERROR: NULL pointer to interpolant in StateTable::query()");
  return interp->value( indepVars );
}

//--------------------------------------------------------------------

double
StateTable::query( const std::string & name,
                   const double* indepVars ) const
{
  return query( find_entry(name), indepVars );
}

//--------------------------------------------------------------------

template<typename Archive>
void
StateTable::serialize( Archive& ar, const unsigned int version )
{
  ar & boost::serialization::make_nvp( "gitRepoDate_", const_cast<std::string&>(gitRepoDate_) );
  ar & boost::serialization::make_nvp( "gitRepoHash_", const_cast<std::string&>(gitRepoHash_) );
  ar & BOOST_SERIALIZATION_NVP(tableName_    );
  ar & BOOST_SERIALIZATION_NVP(numDim_       );
  ar & BOOST_SERIALIZATION_NVP(table_        );
  ar & BOOST_SERIALIZATION_NVP(indepVarNames_);
  ar & BOOST_SERIALIZATION_NVP(depVarNames_  );
  ar & BOOST_SERIALIZATION_NVP(metaData_     );
}

// explicit instantiation
template void StateTable::serialize<InputArchive >( InputArchive&,  const unsigned int );
template void StateTable::serialize<OutputArchive>( OutputArchive&, const unsigned int );

BOOST_CLASS_VERSION( StateTable, 1 )

//--------------------------------------------------------------------
void
StateTable::write_tecplot( const std::vector<int> & npts,
                           const std::vector<double> & upbounds,
                           const std::vector<double> & lobounds,
                           const std::string & fileNamePrefix )
{
  using std::endl;

  if( npts.size() > 3 ){
    cout << "WARNING: cannot write tecplot files with more than three independent variables." << endl
         << "         File: '" << fileNamePrefix << "' will not be written." << endl;
    return;
  }

  const string fnam = fileNamePrefix+".dat";
  std::ofstream fout( fnam.c_str() );
  fout << "title = \"" << fileNamePrefix << "\"" << endl
       << "variables = ";

  // write independent variable labels
  BOOST_FOREACH( const std::string& nam, indepVarNames_ ){
    fout << "  \"" << nam << "\"";
  }

  // write dependent variable labels
  Table::const_iterator itbl;
  for( itbl=table_.begin(); itbl!=table_.end(); ++itbl ){
    const std::string& nam = itbl->first;
    fout << "  \"" << nam << "\"";
    BOOST_FOREACH( const std::string& ivarNam, indepVarNames_ ){
      fout << " \"d"   << nam << "/d" << ivarNam << "\""
           << " \"d^2" << nam << "/d" << ivarNam << "^2\"";
    }
  }

  fout << endl;

  // write information about number of points - data will be written in block format
  fout << "ZONE ";
  string label[] = {"I=", "J=", "K="};
  for(unsigned int i=0; i<npts.size(); ++i )
    fout << label[i] << npts[i] << ", ";
  fout << "F=BLOCK";
  fout << endl;

  const int ndim = npts.size();

  // dump out the mesh  - independent variables
  switch ( ndim ){
    case 3:
    {
      fout << endl << "# " << indepVarNames_[0] << endl;
      double spacing = (upbounds[0]-lobounds[0])/double(npts[0]-1);
      double shift = lobounds[0];
      for( int k=0; k<npts[2]; ++k )
        for( int j=0; j<npts[1]; ++j )
          for( int i=0; i<npts[0]; ++i )
            fout << " " << i*spacing+shift << endl;
      fout << endl;

      fout << endl << "# " << indepVarNames_[1] << endl;
      spacing = (upbounds[1]-lobounds[1])/double(npts[1]-1);
      shift = lobounds[1];
      for( int k=0; k<npts[2]; ++k )
        for( int j=0; j<npts[1]; ++j )
          for( int i=0; i<npts[0]; ++i )
            fout << " " << j*spacing+shift << endl;
      fout << endl;

      fout << endl << "# " << indepVarNames_[2] << endl;
      spacing = (upbounds[2]-lobounds[2])/double(npts[2]-1);
      shift = lobounds[2];
      for( int k=0; k<npts[2]; ++k )
        for( int j=0; j<npts[1]; ++j )
          for( int i=0; i<npts[0]; ++i )
            fout << " " << k*spacing+shift << endl;
      fout << endl;
      break;
    }
    case 2:
    {
      fout << endl << "# " << indepVarNames_[0] << endl;
      double spacing = (upbounds[0]-lobounds[0])/double(npts[0]-1);
      double shift = lobounds[0];
      for( int j=0; j<npts[1]; ++j )
        for( int i=0; i<npts[0]; ++i )
          fout << " " << i*spacing+shift << endl;
      fout << endl;

      fout << endl << "# " << indepVarNames_[1] << endl;
      spacing = (upbounds[1]-lobounds[1])/double(npts[1]-1);
      shift = lobounds[1];
      for( int j=0; j<npts[1]; ++j )
        for( int i=0; i<npts[0]; ++i )
          fout << " " << j*spacing+shift << endl;
      fout << endl;
      break;
    }
    case 1:
    {
      fout << endl << "# " << indepVarNames_[0] << endl;
      double spacing = (upbounds[0]-lobounds[0])/double(npts[0]-1);
      double shift = lobounds[0];
      for( int i=0; i<npts[0]; ++i )
        fout << " " << i*spacing+shift << endl;
      fout << endl;
      break;
    }

  }

  double query[3];
  for( itbl=table_.begin(); itbl!=table_.end(); ++itbl ){

    const std::string& nam = itbl->first;

    // loop over derivative entries and the original variable
    //  ivar=0         -> original variable
    //  ivar=1,3,5 ... -> first derivative w.r.t. each indep var
    //  ivar=2,4,6 ... -> second derivative w.r.t. each indep var
    for( short ivar=0; ivar<1+2*ndim; ++ivar ){

      const InterpT& interp = *itbl->second;

      fout << endl << "# ";
      if     ( ivar>0          ) fout << itbl->first;
      else if( (ivar-1)%2 == 0 ) fout << "d^2" << itbl->first;
      else                       fout << "d" << itbl->first;
      fout << endl;

      switch ( ndim ){
        case 3:
        {
          for( int k=0; k<npts[2]; ++k ){
            double spacing = (upbounds[2]-lobounds[2])/double(npts[2]-1);
            double shift = lobounds[2];
            query[2] = k*spacing+shift;
            for( int j=0; j<npts[1]; ++j ){
              spacing = (upbounds[1]-lobounds[1])/double(npts[1]-1);
              shift = lobounds[1];
              query[1] = j*spacing+shift;
              spacing = (upbounds[0]-lobounds[0])/double(npts[0]-1);
              shift = lobounds[0];
              for( int i=0; i<npts[0]; ++i ){
                query[0] = i*spacing+shift;
                if     ( ivar==0 ) fout << " " << interp.            value(query                      ) << endl;
                else if( ivar%2  ) fout << " " << interp.       derivative(query,(ivar-1)/2           ) << endl;
                else               fout << " " << interp.second_derivative(query,(ivar-1)/2,(ivar-1)/2) << endl;
              }
            }
          }
          break;
        }
        case 2:
        {
          for( int j=0; j<npts[1]; ++j ){
            double spacing = (upbounds[1]-lobounds[1])/double(npts[1]-1);
            double shift = lobounds[1];
            query[1] = j*spacing+shift;
            spacing = (upbounds[0]-lobounds[0])/double(npts[0]-1);
            shift = lobounds[0];
            for( int i=0; i<npts[0]; ++i ){
              query[0] = i*spacing+shift;
              if     ( ivar==0 ) fout << " " << interp.            value(query                      ) << endl;
              else if( ivar%2  ) fout << " " << interp.       derivative(query,(ivar-1)/2           ) << endl;
              else               fout << " " << interp.second_derivative(query,(ivar-1)/2,(ivar-1)/2) << endl;
            }
          }
          break;
        }
        case 1:
        {
          double spacing = (upbounds[0]-lobounds[0])/double(npts[0]-1);
          double shift = lobounds[0];
          for( int i=0; i<npts[0]; ++i ){
            query[0] = i*spacing+shift;
            if     ( ivar==0 ) fout << " " << interp.            value(query                      ) << endl;
            else if( ivar%2  ) fout << " " << interp.       derivative(query,(ivar-1)/2           ) << endl;
            else               fout << " " << interp.second_derivative(query,(ivar-1)/2,(ivar-1)/2) << endl;
          }
          break;
        }
        default:
        {
          break;
        }
      }
      fout << endl;
    }
  }

}

//--------------------------------------------------------------------

void
StateTable::write_matlab( const std::vector<int>& npts,
                          const std::vector<double>& upbounds,
                          const std::vector<double> & lobounds,
                          const std::vector<bool>& logScale,
                          const std::string & fileNamePrefix )
{
  using namespace std;

  const int nivar = npts.size();

  //
  // independent variables:
  //
  string fnam = fileNamePrefix + ".m";
  ofstream fout( fnam.c_str() );
  fout << "% THIS IS AN AUTOMATICALLY GENERATED FILE." << endl << endl;

  fout << "nivar=" << nivar << ";  % number of independent variables" << endl << endl;

  // names
  fout << "ivarNames = {" << endl;
  for( int ivar=0; ivar<nivar; ++ivar ){
    fout << "  '" << indepVarNames_[ivar] << "'" << endl;
  }
  fout << "};" << endl << endl;

  fout << "% number of points in each independent variable dimension (helpful for reshape)"
       << endl
       << "npts = [ ";
  int ntot=1;
  for( std::vector<int>::const_iterator ii=npts.begin(); ii!=npts.end(); ++ii ){
    ntot *= *ii;
    fout << *ii << " ";
  }
  fout << "];" << endl << endl;

  // values
  fout << "ivar = cell(" << nivar << ",1);" << endl;
  for( int ivar=0; ivar<nivar; ++ivar ){
    fout << "ivar{" << ivar+1 << "} = [ ..." << endl;
    const double spacing = (upbounds[ivar]-lobounds[ivar])/double(npts[ivar]-1);
    const double shift = lobounds[ivar];
    for( int i=0; i<ntot; ++i ){
      const int ipt = getIndex( i, npts, ivar );
      double value = ipt*spacing+shift;
      if( logScale[ivar] ) value=std::pow(10,value);
      fout << "  " << value << " ..." << endl;
    }
    fout << "];" << endl << endl;
  }


  //
  // dependent variables
  //
  const int ndvar = 3*table_.size();
  fout << "ndvar=" << ndvar  << ";  % number of dependent variables" << endl << endl;

  fout << "dvarNames = {" << endl;
  Table::const_iterator itbl;
  for( itbl=table_.begin(); itbl!=table_.end(); ++itbl ){
    fout << "  '" << itbl->first << "'" << endl;
    BOOST_FOREACH( const std::string& ivarNam, indepVarNames_ ){
      fout << "  'd" << itbl->first << "/d" << ivarNam << "'" << endl;
      fout << "  'd^2" << itbl->first << "/d" << ivarNam << "^2'" << endl;
    }
  }
  fout << "};" << endl << endl;

  fout << "dvar = cell(" << ndvar << ",1);" << endl;
  int iii=1;
  std::vector<double> query(nivar,0);
  for( itbl=table_.begin(); itbl!=table_.end(); ++itbl ){

    for( short kvar=0; kvar<=2*nivar; ++kvar, ++iii ){ // loop over derivatives & original var

      fout << "dvar{" << iii << "} = [ ..." << endl;
      for( int i=0; i<ntot; ++i ){

        // load the independent variable vector for this point.
        for( int ivar=0; ivar!=nivar; ++ivar ){
          const double spacing = (upbounds[ivar]-lobounds[ivar])/double(npts[ivar]-1);
          const double shift = lobounds[ivar];
          const int ipt = getIndex( i, npts, ivar );
          double value = ipt*spacing+shift;
          if( logScale[ivar] ) value = std::pow(10,value);
          query[ivar] = value;
        }

        if     ( kvar==0 ) fout << "  " << itbl->second->            value(query                      ) << " ... " << endl;
        else if( kvar%2  ) fout << "  " << itbl->second->       derivative(query,(kvar-1)/2           ) << " ... " << endl;
        else               fout << "  " << itbl->second->second_derivative(query,(kvar-1)/2,(kvar-1)/2) << " ... " << endl;
      } // loop over points in the table.

      fout << "];" << endl << endl;
    }

  } // loop over dependent variables
}

//--------------------------------------------------------------------

void
StateTable::output_table_info( std::ostream& out ) const
{
  using namespace std;
  out << endl
      << "------------------------------------------------------" << endl
      << "                 Table information:" << endl
      << "------------------------------------------------------" << endl
      << endl
      << "This table was generated using TabProps from source revision: " << endl
      << "  date    : " << repo_date() << endl
      << "  hash tag: " << repo_hash() << endl
      << endl
      << "Independent variables:" << endl;

  const std::vector<std::pair<double,double> > bounds = begin()->second->get_bounds();
  size_t ivar=0;
  BOOST_FOREACH( const string& nam, indepVarNames_ ){
    out << setw(35) << left << nam << " [ " << bounds[ivar].first << ", " << bounds[ivar].second << " ]" << endl;
    ++ivar;
  }
  out << endl << "Table entry information:" << endl;
  for( Table::const_iterator i=begin(); i!=end(); ++i ){
    const InterpT* const interp = i->second;
    out << "  " << setw(30) << left << i->first
        << " :  order=" << interp->get_order()
        << "  with " << interp->get_dimension() << " independent variables,";
    if( !interp->clipping() ) out << " no";
    out << " clipping"
        << endl;
  }
  out << endl
      << "------------------------------------------------------" << endl
      << endl;
}

//--------------------------------------------------------------------

bool
StateTable::operator==( const StateTable& st ) const
{
  if( numDim_ != st.numDim_ ) return false;

  if( indepVarNames_.size() != st.indepVarNames_.size() ) return false;
  for( size_t i=0; i<indepVarNames_.size(); ++i )
    if( indepVarNames_[i] != st.indepVarNames_[i] ) return false;

  if( depVarNames_.size() != st.depVarNames_.size() ) return false;
  for( size_t i=0; i<depVarNames_.size(); ++i )
    if( depVarNames_[i] != st.depVarNames_[i] ) return false;


  if( table_.size() != st.table_.size() ) return false;

  Table::const_iterator iother = st.table_.begin();
  for( Table::const_iterator ientry=table_.begin(); ientry!=table_.end(); ++ientry, ++iother ){
    if( *ientry == *iother ){}
    else{ return false; }
  }
  return true;
}

//--------------------------------------------------------------------

int
StateTable::getIndex( const int i,
                      const std::vector<int>& npts,
                      const int ivar )
{
  int ipt = -1;

  const int nvar = npts.size();

  switch (nvar) {
  case 1:{
    ipt = i;
    break;
  }
  case 2:{
    switch (ivar) {
    case 0:
      ipt =  i%npts[0];
      break;
    case 1:
      ipt =  i/npts[0];
      break;
    }
    break;
  }
  case 3:
    switch (ivar) {
    case 0:
      ipt =  i%npts[0];
      break;
    case 1:
      ipt =  i/npts[0] % npts[1];
      break;
    case 2:
      ipt =  i/(npts[0]*npts[1]);
      break;
    }
    break;
  default:
    std::ostringstream errmsg;
    errmsg << "Unsupported number of independent variables" << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
  return ipt;
}

//--------------------------------------------------------------------

//====================================================================
