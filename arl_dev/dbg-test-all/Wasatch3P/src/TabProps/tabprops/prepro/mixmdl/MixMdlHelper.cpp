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

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

using std::string;
using std::vector;
using std::cout;
using std::endl;

#include <tabprops/TabProps.h>
#include <tabprops/prepro/mixmdl/PresumedPDFMixMdl.h>
#include <tabprops/prepro/mixmdl/Integrator.h>
#include <tabprops/prepro/mixmdl/GaussKronrod.h>   // for default integrator
#include <tabprops/prepro/mixmdl/MixMdlHelper.h>
#include <tabprops/prepro/mixmdl/MixMdlFunctor.h>

#include <tabprops/StateTable.h>
#include <tabprops/Archive.h>

//====================================================================

MixMdlHelper::MixMdlHelper( PresumedPDFMixMdl & mixingModel,
			    StateTable & rxnMdl,
			    const string & convVarNam,
			    const int order )
  : mixMdl_( mixingModel ),
    rxnMdl_( rxnMdl ),
    nDimRxn_( rxnMdl.get_ndim()   ),
    nDim_   ( rxnMdl.get_ndim()+1 ), // add the variance dimension
    order_  ( order ),

    mixMdlTable_( new StateTable( rxnMdl.get_ndim()+1 ) )
{
  // set independent variable names, adding variance as the last variable
  indepVarNames_ = rxnMdl.get_indepvar_names();
  indepVarNames_.push_back(convVarNam+"Variance");

  // determine the index for the convolution variable
  convVarIndex_ = findix( convVarNam );

  // error checking
  if( convVarIndex_ < 0 ){
    std::ostringstream errmsg;
    errmsg << "ERROR: the requested convolution variable '" << convVarNam << "'" << endl
	   << "       was not found.  Available independent variables include: " << endl;
    for(unsigned int i=0; i<indepVarNames_.size(); ++i )
      errmsg << "    -> " << indepVarNames_[i] << endl;
    errmsg << endl;
    throw std::runtime_error( errmsg.str() );
  }

  // default bounds: [0,1]
  loBound_ = 0.0;
  hiBound_ = 1.0;

  // size the mesh (number of dimensions)
  mesh_.resize( nDim_ );

}
//--------------------------------------------------------------------
MixMdlHelper::~MixMdlHelper()
{
  delete mixMdlTable_;
}
//--------------------------------------------------------------------
int
MixMdlHelper::findix( const string & name ) const
{
  for( int i=0; i<nDim_; ++i )
    if( name == indepVarNames_[i] )
      return i;
  return -1;
}
//--------------------------------------------------------------------
void
MixMdlHelper::set_integrator( Integrator * i )
{
  mixMdl_.set_integrator(i);
}
//--------------------------------------------------------------------
void
MixMdlHelper::set_mesh( const string & varName,
			const vector<double> & vals )
{
  const int dim = findix( varName );
  if( dim > nDim_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR: specified dimension (" << dim
	   << ") exceeds number of dimensions (" << nDim_ << ")." << std::endl
	   << "       Note that mesh indexing is zero-based." << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  mesh_[dim] = vals;
}
//--------------------------------------------------------------------
void
MixMdlHelper::set_mesh( const vector< vector<double> > & mesh )
{
  for(unsigned int i=0; i<mesh.size(); ++i )
    set_mesh( indepVarNames_[i], mesh[i] );
}
//--------------------------------------------------------------------
void
MixMdlHelper::implement()
{
  cout << endl << endl
      << "----------------------------" << endl
      << "Implementing mixing model..." << endl
      << "----------------------------" << endl << endl;
  npts_.assign( nDim_, 0 );
  for( int i=0; i<nDim_; ++i ){
    npts_[i] = mesh_[i].size();
    if( npts_[i] < 2 ){
      std::ostringstream errmsg;
      errmsg << "ERROR: Invalid mesh for dimension '" << indepVarNames_[i] << "'." << endl;
      throw std::runtime_error( errmsg.str() );
    }
    npts_[i] = mesh_[i].size();
  }

  //
  // do we have an integrator assigned?  If not, provide a default
  //
  Integrator * defaultIntegrator = NULL;
  if( mixMdl_.get_integrator() == NULL ){
    defaultIntegrator =  new GaussKronrod( loBound_, hiBound_ );
    mixMdl_.set_integrator( defaultIntegrator );
  }

  //
  // loop over each entry in the table
  //
  StateTable::Table::const_iterator ientry;
  for( ientry=rxnMdl_.begin(); ientry!=rxnMdl_.end(); ++ientry ){

    const string & varName = ientry->first;
    const InterpT* const interp = ientry->second;

    apply_on_mesh( varName, interp );
  }

  delete defaultIntegrator;

  // export the table to disk
  std::ofstream outFile( "MixingModel.tbl", std::ios_base::out|std::ios_base::trunc );
  OutputArchive oa(outFile);
  oa << boost::serialization::make_nvp("mixMdlTable_", *mixMdlTable_ );
}
//--------------------------------------------------------------------
void
MixMdlHelper::apply_on_mesh( const string & varName,
			     const InterpT * const interp )
{
  cout << "Applying mixing model for: '" << varName << "'" << endl;

  // how many values, excluding the variance dimension?
  const int nRxn = std::accumulate( npts_.begin(), npts_.end()-1, 1, std::multiplies<int>() );

  // retrieve the mesh for the variance direction.  This is the last entry in the mesh.
  vector<double> & varMesh = mesh_[nDim_-1];
  const unsigned int nVariance = varMesh.size();

  vector<double> outputVarVals( nRxn*nVariance,  0.0 );
  vector<double>        values(       nDimRxn_,  0.0 );

  // generate the functor
  FunctorDoubleBase * const func =
    new FunctorDoubleVec<InterpT>( interp, &InterpT::value, convVarIndex_, values );

  // hook the functor up to the mixing model
  mixMdl_.set_convolution_func( func );

  // Loop over the mesh in each dimension.  Here we use a flat layout for all dimensions
  // except the variance dimension, which will be dealt with in the inner-loop.  The
  // appropriate values for independent variables are set in setup_mesh_point()

  for( int ipt=0; ipt<nRxn; ++ipt ){

    // For this global index, obtain the vector of values that we will use to evaluate the
    // reaction model.  The convolution variable (mean) and its variance are treated
    // specially since these are important to the presumed-pdf mixing model.
    setup_mesh_point( ipt, values );

    static_cast<FunctorDoubleVec<InterpT>*>(func)->reset_values( values );
    func->reset_n_calls();

    mixMdl_.set_mean( values[convVarIndex_] );

    for(unsigned int ivar=0; ivar<nVariance; ++ivar ){

      mixMdl_.set_scaled_variance( varMesh[ivar] );

      // set the index, apply the mixing model and store the result
      const int ix = ipt + ivar*nRxn;
      outputVarVals[ix] = mixMdl_.integrate();

    } // variance loop

  } // mesh point loop (excluding variance)

  export_variable( outputVarVals, varName );
  delete func;
}
//--------------------------------------------------------------------
void
MixMdlHelper::setup_mesh_point( const int ipoint,
				vector<double> & values )
{
  //
  // which point in n-d space does this index correspond to?
  // exclude last dimension, which is the variance dimension
  //
  // Memory layout convention:
  //   the first dimension varies fastest, last varies slowest.
  //

  // jcs: should store this off.
  // get offsets
  vector<int> ncum(nDimRxn_);
  ncum[0] = 1;
  for( int i=1; i<nDimRxn_; ++i )
    ncum[i] = ncum[i-1]*npts_[i-1];

  // set values
  for( int i=0; i<nDimRxn_; i++ ){
    const int ix = ipoint/ncum[i] % npts_[i];
    values[i] = mesh_[i][ix];
  }
}
//--------------------------------------------------------------------
void
MixMdlHelper::export_variable( const vector<double> & values,
			       const string & name )
{
  const bool allowValueClipping = true;

  const string clip = allowValueClipping ? " with" : " without";
  cout << "  Note: mixing model is using interpolation of order " << order_ << clip
       << " value clipping for variable: " << name << endl;

  InterpT* interp = NULL;
  switch( nDim_ ){
    case 1: interp = new Interp1D( order_, mesh_[0],                               values, allowValueClipping ); break;
    case 2: interp = new Interp2D( order_, mesh_[0], mesh_[1],                     values, allowValueClipping ); break;
    case 3: interp = new Interp3D( order_, mesh_[0], mesh_[1], mesh_[2],           values, allowValueClipping ); break;
    case 4: interp = new Interp4D( order_, mesh_[0], mesh_[1], mesh_[2], mesh_[3], values, allowValueClipping ); break;
    default: {
      std::ostringstream errmsg;
      errmsg << "Unsupported dimensionality (" << nDim_ << ") for mixing model table." << std::endl;
      throw std::runtime_error( errmsg.str() );
    }
  } // switch

  try{
    verify_interp( values, interp, name );
  }
  catch( std::runtime_error & e ){
    std::cout << e.what() << std::endl
	      << "Execution continuing..." << std::endl;
  }

  // export this entry to the table, copying this entry in.
  mixMdlTable_->add_entry( name, interp, indepVarNames_ );

}
//--------------------------------------------------------------------
void
MixMdlHelper::verify_interp( const vector<double> & values,
			     const InterpT* const interp,
			     const string & name )
{
  // check to be sure that the values match the interpolated values

  const int nRxn = std::accumulate( npts_.begin(), npts_.end()-1, 1, std::multiplies<int>() );
  const vector<double> & varMesh = mesh_[nDim_-1];
  const size_t nVar = varMesh.size();

  bool haveError = false;
  std::ostringstream msg;
  msg << endl << "ERROR found in interpolation for variable '" << name << "'" << endl;

  vector<double>indepVars( nDim_, 0.0 );
  for( int irxn=0; irxn<nRxn; irxn++ ){
    setup_mesh_point( irxn, indepVars );
    for( size_t ivar=0; ivar<nVar; ++ivar ){
      indepVars[nDim_-1] = varMesh[ivar];
      const double xx = interp->value( indepVars );

      const double exact = values[irxn+nRxn*ivar];
      const double perr = ( exact-xx )/exact * 100.0;

      if( perr > 1.0e-6 ){
	haveError = true;
	msg << "     interpolation failure at point: [";
	for( vector<double>::const_iterator ii = indepVars.begin(); ii!=indepVars.end(); ++ii ){
	  msg << *ii << ",";
	}
	msg << "].  %error=" << perr << endl;
      }
    }
  }

  if( haveError ) throw std::runtime_error( msg.str() );

}
