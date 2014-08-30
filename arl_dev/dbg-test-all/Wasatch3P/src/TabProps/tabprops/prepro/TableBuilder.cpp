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

#include <sstream>
#include <stdexcept>
#include <algorithm>

#include <tabprops/StateTable.h>

#include <tabprops/prepro/TableBuilder.h>
#include <tabprops/prepro/CanteraTblBuilder.h>

#include <boost/algorithm/string.hpp>

using std::map;
using std::set;
using std::vector;

const static string tstr  = "temperature";
const static string vstr  = "viscosity";
const static string cpstr = "specificheat";
const static string cond  = "conductivity";
const static string dstr  = "density";
const static string sstr  = "species";
const static string mfstr = "molefrac";
const static string rrstr = "reactionrate";
const static string histr = "speciesenthalpy";
const static string hstr  = "enthalpy";
const static string hsstr = "sensibleenthalpy";
const static string hastr = "adiabaticenthalpy";
const static string mwstr = "molecularweight";

//====================================================================

StateVarEvaluator::StateVars get_state_var( const std::string& varname )
{
  using boost::algorithm::iequals;  // case insensitive string comparison

  StateVarEvaluator::StateVars v;

  if     ( iequals(varname,tstr ) )  v=StateVarEvaluator::TEMPERATURE;
  else if( iequals(varname,vstr ) )  v=StateVarEvaluator::VISCOSITY;
  else if( iequals(varname,dstr ) )  v=StateVarEvaluator::DENSITY;
  else if( iequals(varname,cpstr) )  v=StateVarEvaluator::SPECIFIC_HEAT;
  else if( iequals(varname,cond ) )  v=StateVarEvaluator::CONDUCTIVITY;
  else if( iequals(varname,sstr ) )  v=StateVarEvaluator::SPECIES;
  else if( iequals(varname,mfstr) )  v=StateVarEvaluator::MOLEFRAC;
  else if( iequals(varname,rrstr) )  v=StateVarEvaluator::SPECIES_RR;
  else if( iequals(varname,histr) )  v=StateVarEvaluator::SPECIES_ENTH;
  else if( iequals(varname,hstr ) )  v=StateVarEvaluator::ENTHALPY;
  else if( iequals(varname,hsstr) )  v=StateVarEvaluator::SENS_ENTH;
  else if( iequals(varname,hastr) )  v=StateVarEvaluator::AD_ENTH;
  else if( iequals(varname,mwstr) )  v=StateVarEvaluator::MIXTURE_MW;
  else{
    std::ostringstream errmsg;
    errmsg << "ERROR: invalid output specification: '" << varname << "'" << endl
           << "       from " << __FILE__ << " : " << __LINE__ << endl;
    throw std::runtime_error( errmsg.str() );
  }
  return v;
}

std::istringstream &
operator >> ( std::istringstream & istr,
              StateVarEvaluator::StateVars& v )
{
  string s;
  istr >> s;
  v = get_state_var(s);
  return istr;
}

//====================================================================

TableBuilder::TableBuilder( Cantera_CXX::IdealGasMix & gas,
                            const vector<std::string> & indepVarNames,
                            const int order )
  : canteraProps_( gas ),
    indepVarNames_( indepVarNames ),
    nDim_( indepVarNames.size() ),
    order_( order )
{
  mesh_.resize( nDim_ );
  npts_.resize( nDim_ );
  tablePrefix_ = "StateTable";
  firstEntry_ = true;
}
//--------------------------------------------------------------------
TableBuilder::~TableBuilder()
{
  OutputRequest::const_iterator ii;
  for( ii =requestedOutputVars_.begin();
       ii!=requestedOutputVars_.end();
       ++ii )
  {
    delete *ii;
  }
}
//--------------------------------------------------------------------
void
TableBuilder::request_output( const StateVarEvaluator::StateVars stateVar )
{
  // avoid duplicates

  StateVarEvaluator * varEvaluator = NULL;
  pair<OutputRequest::const_iterator,bool> result;

  switch (stateVar){

  case StateVarEvaluator::DENSITY:{
    varEvaluator = new DensityEvaluator( canteraProps_ );
    break;
  }
  case StateVarEvaluator::VISCOSITY:{
    varEvaluator = new ViscosityEvaluator( canteraProps_ );
    break;
  }
  case StateVarEvaluator::SPECIFIC_HEAT:{
    varEvaluator = new SpecificHeatEvaluator( canteraProps_ );
    break;
  }
  case StateVarEvaluator::CONDUCTIVITY:{
    varEvaluator = new ConductivityEvaluator( canteraProps_ );
    break;
  }
  case StateVarEvaluator::ENTHALPY:{
    varEvaluator = new EnthalpyEvaluator( canteraProps_ );
    break;
  }
  case StateVarEvaluator::AD_ENTH:
  case StateVarEvaluator::SENS_ENTH:{
    std::ostringstream errmsg;
    // we don't have all of the information required to construct a
    // SensEnthEvaluator - it should be constructed by the appropriate
    // reaction model class.
    errmsg << "sensible and adiabatic enthalpy should be automatically included"
           << std::endl << "by reaction models with heat loss."
           << std::endl << std::endl
           << "If you are requesting these, then probably your model does not"
           << "support calculating these quantities - perhaps it is adiabatic?"
           << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
  case StateVarEvaluator::TEMPERATURE:{
    varEvaluator = new TemperatureEvaluator();
    break;
  }
  case StateVarEvaluator::SPECIES:{
    // add all species to the list
    const vector<std::string> & specNames = canteraProps_.speciesNames();
    vector<std::string>::const_iterator istr;
    for( istr =specNames.begin(); istr!=specNames.end(); ++istr )
      request_output( StateVarEvaluator::SPECIES, *istr );
    break;
  }
  case StateVarEvaluator::MOLEFRAC:{
    // add all species to the list
    const vector<std::string> & specNames = canteraProps_.speciesNames();
    vector<std::string>::const_iterator istr;
    for( istr =specNames.begin(); istr!=specNames.end(); ++istr )
      request_output( StateVarEvaluator::MOLEFRAC, *istr );
    break;
  }
  case StateVarEvaluator::SPECIES_RR:{
    // add all reaction rates to the list
    const vector<std::string>& specNames = canteraProps_.speciesNames();
    vector<std::string>::const_iterator istr;
    for( istr =specNames.begin(); istr!=specNames.end(); ++istr )
      request_output( new ReactionRateEvaluator( canteraProps_, *istr ) );
    break;
  }
  case StateVarEvaluator::SPECIES_ENTH:{
    // add all species enthalpies to the list
    const vector<std::string>& specNames = canteraProps_.speciesNames();
    vector<std::string>::const_iterator istr;
    for( istr =specNames.begin(); istr!=specNames.end(); ++istr )
      request_output( new SpecEnthEvaluator( canteraProps_, *istr ) );
    break;
  }
  case StateVarEvaluator::MIXTURE_MW:{
    varEvaluator = new MolecularWeightEvaluator( canteraProps_ );
    break;
  }
  default:{
    std::ostringstream errmsg;
    errmsg << "Unsuported state variable chosen." << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
    break;
  }
  } // switch( stateVar )

  if( varEvaluator != NULL ){
    result = requestedOutputVars_.insert( varEvaluator );
    if( !result.second ) delete varEvaluator;
  }
}
//--------------------------------------------------------------------
void
TableBuilder::request_output( const StateVarEvaluator::StateVars stateVar,
                              const std::string & speciesName )
{
  StateVarEvaluator* seval;

  // add to the list
  switch( stateVar ){
  case StateVarEvaluator::MOLEFRAC: seval = new MoleFracEvaluator( canteraProps_, speciesName ); break;
  case StateVarEvaluator::SPECIES : seval = new  SpeciesEvaluator( canteraProps_, speciesName ); break;
  default:{
    std::ostringstream errmsg;
    errmsg << "Only species selections may have names provided." << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
  }

  pair<OutputRequest::const_iterator,bool> result = requestedOutputVars_.insert( seval );
  if( !result.second ) delete seval;
}
//--------------------------------------------------------------------
void
TableBuilder::request_output( StateVarEvaluator * const varEvaluator )
{
  pair<OutputRequest::const_iterator,bool> result = requestedOutputVars_.insert( varEvaluator );
  if( !result.second ) delete varEvaluator;
}
//--------------------------------------------------------------------
void
TableBuilder::set_mesh( const std::vector< std::vector<double> > & mesh )
{
  int i=0;
  vector< vector<double> >::const_iterator imesh;
  for( imesh=mesh.begin(); imesh!=mesh.end(); ++imesh, ++i )
    set_mesh( i, *imesh );
}
//--------------------------------------------------------------------
void
TableBuilder::set_mesh( const int dimension,
                        const std::vector<double> & mesh )
{
  if( dimension >= nDim_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Tried to assign values to dimension '" << dimension << "'" << std::endl
           << "       but the table only has '" << nDim_ << "' dimensions." << std::endl
           << "       Note that the dimension index is 0-based (C-style)" << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  std::cout << "setting mesh[" << dimension << "] <-- " << mesh.size() << std::endl;

  mesh_[dimension] = mesh;
  npts_[dimension] = mesh.size();
}
//--------------------------------------------------------------------
void
TableBuilder::insert_entry( const vector<double> & point,
                            const double temperature,
                            const double pressure,
                            const vector<double> & species )
{
  // jcs:
  //   currently there is no checking to prevent a duplicate entry from being added.

  if( firstEntry_ ){
    firstEntry_ = false;

    // how many entries total?
    totEntries_ = 1;
    int idim=0;
    vector< vector<double> >::const_iterator imesh;
    for( imesh=mesh_.begin(); imesh!=mesh_.end(); ++imesh, ++idim ){
      const int n = imesh->size();
      if( n <= order_ ){
        std::ostringstream errmsg;
        errmsg << "ERROR: the mesh must have a minimum of " << order_+1 << " entries in each dimension." << std::endl
               << "       Found " << n << " entries in dimension " << idim << std::endl
               << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( errmsg.str() );
      }
      totEntries_ *= n;
    }

    set<StateVarEvaluator*>::iterator ivar;
    for( ivar =requestedOutputVars_.begin();
         ivar!=requestedOutputVars_.end();
         ++ivar )
    {
      std::vector<double> & entry = propEntries_[(*ivar)->get_name()];
      entry.resize( totEntries_ );
    }
  }


  if( point.size() != (unsigned int)nDim_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Entry has inconsistent dimensions." << std::endl
           << "       expecting " << nDim_ << " independent variables, and found " << point.size() << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  // ensure that this point conforms to the specified mesh
  int idim=0;
  vector<int> indx(nDim_);
  vector<double>::const_iterator ientry;
  vector< vector<double> >::const_iterator imesh;
  vector<double>::const_iterator ipt = point.begin();
  for( imesh =mesh_.begin();
       imesh!=mesh_.end();
       ++imesh, ++ipt, ++idim )
  {
    ientry = std::find( imesh->begin(), imesh->end(), *ipt );
    if( ientry == imesh->end() ){
      std::ostringstream errmsg;
      errmsg << "ERROR: could not find an entry with value (" << *ipt << ") in dimension "
             << idim+1 << " of the mesh"  << std::endl
             << "       Check your mesh." << std::endl
             << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( errmsg.str() );
    }
    indx[idim] = ientry-imesh->begin();
  }

  // set the 1-D index
  int flatIndex = 0;
  for( int idim=mesh_.size()-1; idim>=0; --idim ){
    int tmp=1;
    for( int j=0; j<idim; ++j )  tmp *= mesh_[j].size();
    flatIndex += indx[idim]*tmp;
  }

  if( flatIndex >= totEntries_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: flat index (" << flatIndex
           << ") exceeds mesh extent (" << totEntries_ << ")" << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  // create entries for each requested output
  set<StateVarEvaluator*>::iterator ivar;
  for( ivar =requestedOutputVars_.begin();
       ivar!=requestedOutputVars_.end();
       ++ivar )
  {
    std::vector<double> & entry = propEntries_[(*ivar)->get_name()];
    const double val = (*ivar)->evaluate(temperature, pressure, species);
    entry[flatIndex] = val;
  }

}
//--------------------------------------------------------------------
void
TableBuilder::insert( const InterpT & interpT,
                      const InterpT & interpP,
                      const std::vector<const InterpT*> & interpY )
{
// jcs could also get totEntries_ by product of npts_

  if( !firstEntry_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: TableBuilder::insert() may only be used once!" << endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
  else{
    totEntries_ = 1;
    int idim=0;
    vector< vector<double> >::const_iterator imesh;
    for( imesh=mesh_.begin(); imesh!=mesh_.end(); ++imesh, ++idim ){
      const int n = imesh->size();
      if( n <= order_ ){
        std::ostringstream errmsg;
        errmsg << "ERROR: the mesh must have a minimum of " << order_+1 << " entries in each dimension." << std::endl
               << "       Found " << n << " entries in dimension " << idim << std::endl
               << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( errmsg.str() );
      }
      totEntries_ *= n;
    }

    set<StateVarEvaluator*>::iterator ivar;
    for( ivar =requestedOutputVars_.begin();
         ivar!=requestedOutputVars_.end();
         ++ivar )
    {
      std::vector<double> & entry = propEntries_[(*ivar)->get_name()];
      entry.resize( totEntries_ );
    }

    firstEntry_ = false;
  }


  std::ostringstream errmsg;
  errmsg << "This interface for TableBuilder::insert() is not functional." << std::endl
         << __FILE__ << " : " << __LINE__ << std::endl;
  throw std::runtime_error( errmsg.str() );

  const int nspec = canteraProps_.nSpecies();
  vector<double> ys( nspec, 0.0 );
  vector<double> val(nDim_,0.0);
  vector<int> indx(nDim_,0);
  int flatIndex=0;
  for( int n=nDim_-1; n>=0; --n ){
    for( int i=0; i<npts_[n]; ++i ){
      indx[n]++;
      val[n] = mesh_[n][indx[n]];
      for( int nn=n; nn>=0; --nn ){
        ++flatIndex;
        indx[nn]++;
        val[nn] = mesh_[nn][indx[nn]];

        const double temperature = interpT.value( &val[0] );
        const double pressure    = interpP.value( &val[0] );
        for( int i=0; i<nspec; i++ )
          ys[i] = interpY[i]->value( &val[0] );

        // create entries for each requested output
        set<StateVarEvaluator*>::iterator ivar;
        for( ivar =requestedOutputVars_.begin();
             ivar!=requestedOutputVars_.end();
             ++ivar )
        {
          std::vector<double> & entry = propEntries_[(*ivar)->get_name()];
          const double val = (*ivar)->evaluate( temperature, pressure, ys );
          entry[flatIndex] = val;
        }

      }
    }
  }

  return;

  /*
  const int nspec = canteraProps_.nSpecies();
  vector<double> ys( nspec, 0.0 );
  vector< vector<double> >::const_iterator imesh;
  int flatIndex=0;
  for( imesh=mesh_.begin(); imesh!=mesh_.end(); ++imesh, ++flatIndex ){

    const vector<double> & point = *imesh;
    const double temperature = Tinterp.value( &point[0] );
    const double pressure    = Pinterp.value( &point[0] );
    for( int i=0; i<nspec; i++ )
      ys[i] = Yinterp[i]->value( &point[0] );

    // create entries for each requested output
    set<StateVarEvaluator*>::iterator ivar;
    for( ivar =requestedOutputVars_.begin();
         ivar!=requestedOutputVars_.end();
         ++ivar )
    {
      std::vector<double> & entry = propEntries_[(*ivar)->get_name()];
      const double val = (*ivar)->evaluate( temperature, pressure, ys );
      entry[flatIndex] = val;
    }
  }
  */
}
//--------------------------------------------------------------------
void
TableBuilder::generate_table()
{
  cout << "-----------------------------------------------" << endl
       << "Generating table using interpolants of order: " << order_ << endl
       << "-----------------------------------------------" << endl;

  StateTable table;
  const bool clip = true; // clip values outside allowable range.

  // load each entry into the table
  PropertyEntries::const_iterator iprop;
  for( iprop =propEntries_.begin();
       iprop!=propEntries_.end();
       ++iprop )
  {
    const vector<double> & values = iprop->second;
    cout << "Adding property: '" << iprop->first << "' to table" << endl;

    // interpolate this property
    InterpT * interp = NULL;
    switch ( nDim_ ){
    case 1: interp = new Interp1D( order_, mesh_[0],                               values, clip ); break;
    case 2: interp = new Interp2D( order_, mesh_[0], mesh_[1],                     values, clip ); break;
    case 3: interp = new Interp3D( order_, mesh_[0], mesh_[1], mesh_[2],           values, clip ); break;
    case 4: interp = new Interp4D( order_, mesh_[0], mesh_[1], mesh_[2], mesh_[3], values, clip ); break;
    default:
      std::ostringstream errmsg;
      errmsg << "ERROR: unsupported dimension for interpolant creation!"
             << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( errmsg.str() );

    }

    // load this entry into the table, transferring ownership
    // of the interpolant to the table (hence the "false" flag)
    table.add_entry( iprop->first,
                     interp,
                     indepVarNames_,
                     false );
  }

  // write the table to disk.
  table.write_table( tablePrefix_+".tbl" );
}
//--------------------------------------------------------------------

//====================================================================
