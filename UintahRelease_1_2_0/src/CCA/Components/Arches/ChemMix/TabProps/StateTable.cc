/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/Arches/ChemMix/TabProps/StateTable.h>

#include <Core/Exceptions/InternalError.h>

#include <sci_defs/hdf5_defs.h>

#include <vector>
#include <algorithm>
#include <cmath>

#include <sstream>
#include <stdexcept>

// these are for temporary diagnostics...
#include <iostream>
#include <fstream>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

//--------------------------------------------------------------------

StateTable::StateTable( const int numDim )
{
  numDim_ = numDim;
}
//--------------------------------------------------------------------
StateTable::~StateTable()
{
  Table::const_iterator itbl;
  for( itbl=splineTbl_.begin(); itbl!=splineTbl_.end(); itbl++ ){
    delete itbl->second;
  }
}
//--------------------------------------------------------------------
void
StateTable::add_entry( const string & name,
                       const BSpline * const spline,
                       const vector<string> & indepVarNames,
                       const bool createCopy )
{
  // Is this the first entry?  If so, save off some information about the independent
  // variables.  We require all splines in a table to be functions of the same independent
  // variables.
  if( splineTbl_.size() == 0 ){
    indepVarNames_ = indepVarNames;
    numDim_ = indepVarNames.size();
  }
  
  // check that this spline is consistent with others in the table
  if( spline->get_dimension() != numDim_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Inconsistent dimensionality detected for entry '" << name << "'" << std::endl
           << "       Previous entries had " << numDim_ << " dimensions";
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
  
  // disallow duplicates.
  const BSpline * const entry = find_entry( name );
  if( entry == NULL ){
    //cout << "adding entry: '" << name << "' to table." << endl;
    if( createCopy ){
      splineTbl_[name] = spline->clone();
    }else{              
      splineTbl_[name] = spline;
    }
  }
  else{
    std::ostringstream errmsg;
    errmsg << "ERROR: Attempted to add a duplicate entry to the table." << std::endl
           << "       An entry with the name: '" << name << "' already exists.";
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }

  // ensure that this spline's independent variable names match the others in the table.
  if( indepVarNames != indepVarNames_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: One or more independent variable names in spline '" << name << "'" << std::endl
           << "       do not match those already in the table." << std::endl
           << "       Inconsistent pairs follow: " << endl;
    vector<string>::const_iterator istr =indepVarNames_.begin();
    vector<string>::const_iterator istr2=indepVarNames.begin();
    for( ; istr!=indepVarNames_.end(); ++istr, ++istr2 ) {
      if( *istr != *istr2 ){
        errmsg << "      ('" << *istr << "','" << *istr2 << "')" << std::endl;
      }
    }
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
  // populate list of dependent variable names
  depVarNames_.push_back( name );
}
//--------------------------------------------------------------------
const BSpline *
StateTable::find_entry( const string & name ) const
{
  Table::const_iterator itbl = splineTbl_.begin();
  for( ; itbl!=splineTbl_.end(); itbl++ ){
    if( itbl->first == name ){
      return itbl->second;
    }
  }
  return NULL;
}
//--------------------------------------------------------------------
bool
StateTable::has_indepvar( const string & name ) const
{
  vector<string>::const_iterator istr;
  for( istr=indepVarNames_.begin(); istr!=indepVarNames_.end(); istr++ ){
    if( name == *istr ){
      return true;
    }
  }
  return false;
}
//--------------------------------------------------------------------
void
StateTable::write_hdf5( const string & prefix )
{
#if defined( HAVE_HDF5 )
  const string fileName = prefix+".h5";

  //
  // open an HDF5 file and generate the group for
  // the requested table at the root of the file.
  //
  const hid_t fid = H5Fcreate( fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );

  if( fid < 0 ){
    std::ostringstream errmsg;
    errmsg << "ERROR creating file '" << fileName << "'" << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
  
  const hid_t baseGroup = H5Gcreate( fid, prefix.c_str(), 0 );

  if( baseGroup < 0 ) {
    throw InternalError( "ERROR: could not create an HDF5 base group!!", __FILE__, __LINE__ );
  }else {
    write_hdf5( baseGroup );
  }

  H5Gclose( baseGroup );

/*
  // check to see if there are any open HDF5 identifiers
  const int nopen = H5Fget_obj_count( fid, H5F_OBJ_DATASET|H5F_OBJ_GROUP|H5F_OBJ_ATTR );
  if( nopen != 0 ){
    std::ostringstream errmsg;
    errmsg << "There are " << nopen << " open HDF5 identifiers." << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
*/
  H5Fclose( fid );
#endif
}

//--------------------------------------------------------------------

void
StateTable::write_hdf5( const hid_t & group )
{
#if defined( HAVE_HDF5 )
  const hid_t h5s = H5Screate( H5S_SCALAR );

  // write the dimension to the file
  hid_t h5a = H5Acreate( group, "TableDim", H5T_NATIVE_INT, h5s, H5P_DEFAULT );
  if( h5a < 0 ){
    std::ostringstream errmsg;
    errmsg << "ERROR from StateTable::write_hdf5()" << std::endl
           << "      could not write to the specified group.  " << std::endl
           << "      Ensure that the file is open and the group is valid.";
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
  H5Awrite( h5a, H5T_NATIVE_INT, &numDim_ );
  H5Aclose( h5a );
  
  //
  // write the names to the file
  //
  hid_t h5StrID = H5Tcopy( H5T_C_S1 );
  int icount=0;
  vector<string>::const_iterator inam;
  for( inam=indepVarNames_.begin(); inam!=indepVarNames_.end(); inam++, icount++ ){
    std::ostringstream label;
    label << "IndepVar_" << icount;

    H5Tset_size( h5StrID, inam->size() );
    h5a = H5Acreate( group, label.str().c_str(), h5StrID, h5s, H5P_DEFAULT );
    H5Awrite( h5a, h5StrID, inam->c_str() );
    H5Aclose( h5a );
  }
  H5Tclose( h5StrID );

  
  //
  // write the number of entries (dep vars) in the table
  //
  const int nEntries = splineTbl_.size();
  h5a = H5Acreate( group, "NDepVars", H5T_NATIVE_INT, h5s, H5P_DEFAULT );
  H5Awrite( h5a, H5T_NATIVE_INT, &nEntries );
  H5Aclose( h5a );

  H5Sclose( h5s );
  
  //
  // write the splined dataset
  //
  int splineCount=0;
  Table::const_iterator itbl;
  for( itbl =splineTbl_.begin();itbl!=splineTbl_.end(); ++itbl, ++splineCount )
  {
    const string splineName = itbl->first;
    const hid_t spGroup = H5Gcreate( group, splineName.c_str(), 0 );
    if( spGroup < 0 ){
      std::ostringstream errmsg;
      errmsg << "ERROR: Could not create HDF5 group named '" << splineName << "'" << std::endl;
      throw InternalError( errmsg.str(), __FILE__, __LINE__ );
    }
    
    //
    // write the spline
    //
    itbl->second->write_hdf5( spGroup );
    H5Gclose( spGroup );
  }
#endif
}
//--------------------------------------------------------------------
void
StateTable::read_hdf5( const string & prefix,
                       string inputGroupName )
{
#if defined( HAVE_HDF5 )
  const string fileName = prefix + ".h5";

  if( inputGroupName.empty() ){
    inputGroupName = prefix;
  }
  //
  // open an HDF5 file and generate the group for
  // the requested table at the root of the file.
  //
  const hid_t fid = H5Fopen( fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
  if( (int)fid < 0 ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Could not open HDF5 file: '" << fileName << "'"  << std::endl
           << "       Check for file's existence and verify file name." << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }

  const hid_t baseGroup = H5Gopen( fid, inputGroupName.c_str() );
  if( baseGroup < 0 ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Could not open group named '" << prefix
           << "' from file '" << fileName << "'" << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }

  read_hdf5( baseGroup );

  H5Gclose( baseGroup );
  H5Fclose( fid );
#endif
}

//--------------------------------------------------------------------

void
StateTable::read_hdf5( const hid_t & group )
{
#if defined( HAVE_HDF5 )
  // check for valid group
  if( group < 0 ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Tried to read an invalid group!" << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__  );
  }
    
  //
  // what dimensionality?
  //
  hid_t h5a = H5Aopen_name( group, "TableDim" );
  H5Aread( h5a, H5T_NATIVE_INT, &numDim_ );
  H5Aclose( h5a );

  //
  // load the names of the independent variables
  //
  hid_t h5StrID = H5Tcopy( H5T_C_S1 );
  for( int iname=0; iname<numDim_; iname++ ){
    std::ostringstream label;
    label << "IndepVar_" << iname;
    h5a = H5Aopen_name( group, label.str().c_str() );
    H5Tset_size(h5StrID,50);
    char varName[50] = "";
    H5Aread( h5a, h5StrID, varName );
    H5Aclose( h5a );
    indepVarNames_.push_back( varName );
  }

  //
  // load the number of dep vars
  // 
  h5a = H5Aopen_name( group, "NDepVars" );
  unsigned int nEntries = 0;
  H5Aread( h5a, H5T_NATIVE_INT, &nEntries );
  H5Aclose( h5a );

  hsize_t nn;
  H5Gget_num_objs(group,&nn);

  if( nn != nEntries ){
    std::ostringstream errmsg;
    char gnam[50];
    H5Iget_name( group, gnam, 50 );
    errmsg << "ERROR: Inconsistent number of objects found in group: '" << gnam << "'." <<std::endl
           << "       Found " << nn << " but expecting " << nEntries << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }

  for( unsigned int isp=0; isp<nEntries; ++isp ) {
    
    //
    // get the name for this dataset
    //
    char spGroupName[50];
    H5Gget_objname_by_idx( group, isp, spGroupName, 50 );
    
    const hid_t spGroup = H5Gopen( group, spGroupName );
    
    //
    // construct a skeletal spline, and then read it in
    //
    BSpline * entry = NULL;
    switch( indepVarNames_.size() ){
    case 1:
      entry = new BSpline1D( true );
      break;
    case 2:
      entry = new BSpline2D( true );
      break;
    case 3:
      entry = new BSpline3D( true );
      break;
    case 4:
      entry = new BSpline4D( true );
      break;
    case 5:
      entry = new BSpline5D( true );
      break;
    default:
      std::ostringstream errmsg;
      errmsg << "ERROR: unsupported dimension for BSpline creation!";
      throw InternalError( errmsg.str(), __FILE__, __LINE__ );
    }
    entry->read_hdf5( spGroup );
    H5Gclose( spGroup );
    
    //
    // add the newly created entry to the table.
    //
    add_entry( spGroupName, entry, indepVarNames_ );
  }
#endif
}

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
  vector<string>::const_iterator inam;
  for( inam=indepVarNames_.begin(); inam!=indepVarNames_.end(); ++inam ){
    fout << "  \"" << *inam << "\"";
  }

  // write dependent variable labels
  Table::const_iterator itbl;
  for( itbl=splineTbl_.begin(); itbl!=splineTbl_.end(); ++itbl ){
    fout << "  \"" << itbl->first << "\"";
  }

  fout << endl;
  
  // write information about number of points - data will be written in block format
  fout << "ZONE ";
  string label[] = {"I=", "J=", "K="};
  for(unsigned int i=0; i<npts.size(); ++i ){
    fout << label[i] << npts[i] << ", ";
  }
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
      for( int k=0; k<npts[2]; ++k ){
        for( int j=0; j<npts[1]; ++j ){
          for( int i=0; i<npts[0]; ++i ){
            fout << " " << i*spacing+shift << endl;
          }
        }
      }
      fout << endl;

      fout << endl << "# " << indepVarNames_[1] << endl;
      spacing = (upbounds[1]-lobounds[1])/double(npts[1]-1);
      shift = lobounds[1];
      for( int k=0; k<npts[2]; ++k ){
        for( int j=0; j<npts[1]; ++j ){
          for( int i=0; i<npts[0]; ++i ){
            fout << " " << j*spacing+shift << endl;
          }
        }
      }
      fout << endl;

      fout << endl << "# " << indepVarNames_[2] << endl;
      spacing = (upbounds[2]-lobounds[2])/double(npts[2]-1);
      shift = lobounds[2];
      for( int k=0; k<npts[2]; ++k ){
        for( int j=0; j<npts[1]; ++j ){
          for( int i=0; i<npts[0]; ++i ){
            fout << " " << k*spacing+shift << endl;
          }
        }
      }
      fout << endl;
      break;
    }
    case 2:
    {
      fout << endl << "# " << indepVarNames_[0] << endl;
      double spacing = (upbounds[0]-lobounds[0])/double(npts[0]-1);
      double shift = lobounds[0];
      for( int j=0; j<npts[1]; ++j ){
        for( int i=0; i<npts[0]; ++i ){
          fout << " " << i*spacing+shift << endl;
        }
      }
      fout << endl;

      fout << endl << "# " << indepVarNames_[1] << endl;
      spacing = (upbounds[1]-lobounds[1])/double(npts[1]-1);
      shift = lobounds[1];
      for( int j=0; j<npts[1]; ++j ){
        for( int i=0; i<npts[0]; ++i ){
          fout << " " << j*spacing+shift << endl;
        }
      }
      fout << endl;
      break;
    }
    case 1:
    {
      fout << endl << "# " << indepVarNames_[0] << endl;
      double spacing = (upbounds[0]-lobounds[0])/double(npts[0]-1);
      double shift = lobounds[0];
      for( int i=0; i<npts[0]; ++i ){
        fout << " " << i*spacing+shift << endl;
      }
      fout << endl;
      break;
    }
    
  }
  
  double query[3];
  for( itbl=splineTbl_.begin(); itbl!=splineTbl_.end(); ++itbl ){

    fout << endl << "# " << itbl->first << endl;
    
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
              const double val = itbl->second->value( query );
              fout << " " << val << endl;
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
            const double val = itbl->second->value( query );
            fout << " " << val << endl;
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
          const double val = itbl->second->value( query );
          fout << " " << val << endl;
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

  int ntot=1;
  for( std::vector<int>::const_iterator ii=npts.begin(); ii!=npts.end(); ++ii )
    ntot *= *ii;

  // values
  fout << "ivar = cell(" << nivar << ",1);" << endl;
  for( int ivar=0; ivar<nivar; ++ivar ){
    fout << "ivar{" << ivar+1 << "} = [ ..." << endl;
    const double spacing = (upbounds[ivar]-lobounds[ivar])/double(npts[ivar]-1);
    const double shift = lobounds[ivar];
    for( int i=0; i<ntot; ++i ){
      const int ipt = getIndex( i, npts, ivar );
      double value = ipt*spacing+shift;
      if( logScale[ivar] ){
        value=std::pow(10,value);
      }
      fout << "  " << value << " ..." << endl;
    }
    fout << "];" << endl << endl;
  }


  //
  // dependent variables
  //
  const int ndvar = splineTbl_.size();
  fout << "ndvar=" << ndvar  << ";  % number of dependent variables" << endl << endl;

  fout << "dvarNames = {" << endl;
  Table::const_iterator itbl;
  for( itbl=splineTbl_.begin(); itbl!=splineTbl_.end(); ++itbl ){
    fout << "  '" << itbl->first << "'" << endl;
  }
  fout << "};" << endl << endl;

  fout << "dvar = cell(" << splineTbl_.size() << ",1);" << endl;
  int iii=1;
  std::vector<double> query(3,0);
  for( itbl=splineTbl_.begin(); itbl!=splineTbl_.end(); ++itbl, ++iii ){

    fout << "dvar{" << iii << "} = [ ..." << endl;
    for( int i=0; i<ntot; ++i ){

      // load the independent variable vector for this point.
      for( int ivar=0; ivar!=nivar; ++ivar ){
        const double spacing = (upbounds[ivar]-lobounds[ivar])/double(npts[ivar]-1);
        const double shift = lobounds[ivar];
        const int ipt = getIndex( i, npts, ivar );
        double value = ipt*spacing+shift;
        if( logScale[ivar] ){
          value = std::pow(10,value);
        }
        query[ivar] = value;
      }

      const double val = itbl->second->value( query );
      fout << "  " << val << " ..." << endl;

    } // loop over points in the table.

    fout << "];" << endl << endl;

  } // loop over dependent variables
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
      ipt =  i/npts[0];
      break;
    case 2:
      ipt =  i/(npts[0]*npts[1]);
      break;
    }
    break;
  default:
    throw InternalError( "Unsupported number of independent variables", __FILE__, __LINE__ );
  }
  return ipt;
}

//--------------------------------------------------------------------

/*
#include <cmath>
#include <iostream>
bool test_state_tbl()
{
  bool isOkay = true;

  // create a 1-D spline
  const int n = 15;
  vector<double> x, phi1, phi2;
  for( int i=0; i<n; i++ ){
    x.push_back( 4.0*double(i)/double(n-1) );
    phi1.push_back( 1.5*std::sin(3.1415 * x[i]) );
    phi2.push_back( 1.5*std::cos(3.1415 * x[i]) );
  }
  BSpline1D sp1( 4, x, phi1 );
  BSpline1D sp2( 3, x, phi2 );

  vector<string> indepNames;
  indepNames.push_back("x1");
  StateTable tbl( indepNames.size() );
  vector<string> junkNames;
  junkNames.push_back("bad name 1");

  try{
    tbl.add_entry( "phi1", &sp1, indepNames );
    tbl.add_entry( "phi2", &sp2, indepNames );
  }
  catch (const std::runtime_error & e ){
    isOkay = false;
    cout << endl << e.what() << endl;
  }
  
  tbl.write_hdf5( "tbl" );

  return isOkay;
}

int main()
{
  using namespace std;
  test_state_tbl();
  cout << "done" << endl;
}
*/
