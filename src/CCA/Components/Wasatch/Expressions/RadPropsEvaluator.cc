/**
 *  \file   RadPropsEvaluator.cc
 *  \date   Jun 6, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2016 The University of Utah
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

#include "RadPropsEvaluator.h"
#include <radprops/Particles.h>


template< typename FieldT >
RadPropsEvaluator<FieldT>::
RadPropsEvaluator( const Expr::Tag& tempTag,
                   const RadSpecMap& species,
                   const std::string& fileName )
  : Expr::Expression<FieldT>(),
    greyGas_( new GreyGas(fileName) )
{
  // obtain the list of species from the file:
  Expr::TagList indepVarNames;
  const std::vector<RadiativeSpecies>& specOrder = greyGas_->species();
  BOOST_FOREACH( const RadiativeSpecies& sp, specOrder ){
    const RadSpecMap::const_iterator isp = species.find(sp);
    if( isp == species.end() ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "Species '" << species_name(sp) << "' was found in file " << fileName
          << " but no corresponding species was provided in the input file." << std::endl;
      throw std::invalid_argument( msg.str() );
    }
    else{
      indepVarNames.push_back( isp->second );
    }
  }
  
   temp_ = this->template create_field_request<FieldT>(tempTag);
  this->template create_field_vector_request<FieldT>(indepVarNames, indepVars_);
}

//--------------------------------------------------------------------

template< typename FieldT >
RadPropsEvaluator<FieldT>::
~RadPropsEvaluator()
{
  delete greyGas_;
}

//--------------------------------------------------------------------

template< typename FieldT >
void
RadPropsEvaluator<FieldT>::
evaluate()
{
  FieldT& result = this->value();

  typedef typename FieldT::const_iterator Iterator;
  typedef std::vector<Iterator> IVarIter;

  IVarIter ivarIters;
  for (size_t i=0; i<indepVars_.size(); ++i) {
    const FieldT iVar = indepVars_[i]->field_ref();
    ivarIters.push_back(iVar.begin());
  }

  std::vector<double> ivarsPoint( indepVars_.size(), 0.0 );
  const FieldT& temp = temp_->field_ref();
  // loop over grid points.  iii is a dummy variable.
  Iterator itemp=temp.begin();
  for( typename FieldT::iterator iprop=result.begin(); iprop!=result.end(); ++iprop, ++itemp ){

    // extract indep vars at this grid point
    ivarsPoint.clear();
    for( typename IVarIter::iterator i=ivarIters.begin(); i!=ivarIters.end(); ++i ){
      ivarsPoint.push_back( **i );
      ++(*i);
    }

    // for now, we only pull out the effective radiative coefficient
    double planck=0, rosseland=0;
    greyGas_->mixture_coeffs( planck, rosseland, *iprop, ivarsPoint, *itemp );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
RadPropsEvaluator<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& tempTag,
                  const RadSpecMap& species,
                  const std::string& fileName  )
  : ExpressionBuilder( resultTag ),
    rsm_( species ),
    tempTag_( tempTag ),
    fileName_( fileName )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
RadPropsEvaluator<FieldT>::
Builder::build() const
{
  return new RadPropsEvaluator<FieldT>( tempTag_, rsm_, fileName_ );
}

//==========================================================================


template< typename FieldT >
ParticleRadProps<FieldT>::
ParticleRadProps( const ParticleRadProp prop,
                  const Expr::Tag& tempTag,
                  const Expr::Tag& pRadiusTag,
                  const std::complex<double>& refIndex )
  : Expr::Expression<FieldT>(),
    props_( new ParticleRadCoeffs(refIndex,
                                  1e-7,   /// min particle size
                                  1e-4,   // max particle size
                                  10,     // number of sizes
                                  1 ) ),  // order of interpolant
    prop_      ( prop       )
{
   temp_ = this->template create_field_request<FieldT>(tempTag);
   pRadius_ = this->template create_field_request<FieldT>(pRadiusTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
ParticleRadProps<FieldT>::
~ParticleRadProps()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
ParticleRadProps<FieldT>::
evaluate()
{
  FieldT& result = this->value();

  typedef typename FieldT::const_iterator Iterator;

  const FieldT& temp = temp_->field_ref();
  const FieldT& pRadius = pRadius_->field_ref();
  Iterator itemp=temp.begin(), irad = pRadius.begin();
  const Iterator ite=temp.end();
  for( typename FieldT::iterator iprop=result.begin(); itemp!=ite; ++iprop, ++itemp, ++irad ){

    switch( prop_ ){
    case PLANCK_ABSORPTION_COEFF   : *iprop = props_->planck_abs_coeff( *irad, *itemp ); break;
    case PLANCK_SCATTERING_COEFF   : *iprop = props_->planck_sca_coeff( *irad, *itemp ); break;
    case ROSSELAND_ABSORPTION_COEFF: *iprop = props_->ross_abs_coeff  ( *irad, *itemp ); break;
    case ROSSELAND_SCATTERING_COEFF: *iprop = props_->ross_sca_coeff  ( *irad, *itemp ); break;
    }

  }
}

//--------------------------------------------------------------------

template< typename FieldT >
ParticleRadProps<FieldT>::
Builder::Builder( const ParticleRadProp prop,
                  const Expr::Tag& resultTag,
                  const Expr::Tag& tempTag,
                  const Expr::Tag& pRadiusTag,
                  const std::complex<double> refIndex )
  : ExpressionBuilder( resultTag ),
    prop_( prop ),
    tempTag_( tempTag ),
    pRadiusTag_( pRadiusTag ),
    refIndex_( refIndex )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ParticleRadProps<FieldT>::
Builder::build() const
{
  return new ParticleRadProps<FieldT>( prop_, tempTag_, pRadiusTag_, refIndex_ );
}


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class RadPropsEvaluator<SpatialOps::SVolField>;
template class ParticleRadProps <SpatialOps::SVolField>;
//==========================================================================
