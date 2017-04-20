/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#include "DynamicSmagorinskyCoefficient.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/Nebo.h>

#define ALLOCATE_TENSOR_FIELD(T) \
{\
  int jmin=0; \
  for (int i=0; i<3; i++) { \
    for (int j=jmin; j<3; j++) { \
      T[i].push_back(SpatialFieldStore::get<SVolField>( dynSmagConst )); \
    } \
    jmin++;\
  }\
}

#define ALLOCATE_VECTOR_FIELD(T) \
{\
  for (int i=0; i<3; i++) { \
    T.push_back(SpatialFieldStore::get<SVolField>( dynSmagConst )); \
  } \
}


//********************************************************************
// DYNAMIC SMAGORINSKY COEFFICIENT
//********************************************************************

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
DynamicSmagorinskyCoefficient<ResultT, Vel1T, Vel2T, Vel3T>::
DynamicSmagorinskyCoefficient( const Expr::TagList& velTags,
                               const Expr::Tag& rhoTag,
                               const bool isConstDensity )
: StrainTensorBase<ResultT, Vel1T, Vel2T, Vel3T>( velTags     ),
  isConstDensity_ (isConstDensity),
  doExtraFiltering_(false)
{
  this->set_gpu_runnable(true);
  if (!isConstDensity_)  rho_ = this->template create_field_request<SVolField>(rhoTag);
}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
DynamicSmagorinskyCoefficient<ResultT, Vel1T, Vel2T, Vel3T>::
~DynamicSmagorinskyCoefficient()
{}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
void
DynamicSmagorinskyCoefficient<ResultT, Vel1T, Vel2T, Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  StrainTensorBase<ResultT, Vel1T, Vel2T, Vel3T>::bind_operators(opDB);
  exOp_  = opDB.retrieve_operator<ExOpT>();
  xexOp_ = opDB.retrieve_operator<XExOpT>();
  yexOp_ = opDB.retrieve_operator<YExOpT>();
  zexOp_ = opDB.retrieve_operator<ZExOpT>();
  
  boxFilterOp_  = opDB.retrieve_operator<BoxFilterT>();
  xBoxFilterOp_ = opDB.retrieve_operator<XBoxFilterT>();
  yBoxFilterOp_ = opDB.retrieve_operator<YBoxFilterT>();
  zBoxFilterOp_ = opDB.retrieve_operator<ZBoxFilterT>();
  vel1InterpOp_ = opDB.retrieve_operator<Vel1InterpT>();
  vel2InterpOp_ = opDB.retrieve_operator<Vel2InterpT>();
  vel3InterpOp_ = opDB.retrieve_operator<Vel3InterpT>();
}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
void
DynamicSmagorinskyCoefficient<ResultT, Vel1T, Vel2T, Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  typedef SpatFldPtr<SVolField> SVolFldPtr;
  typedef typename Expr::Expression<SVolField>::ValVec SVolFieldVec;
  SVolFieldVec& results   = this->get_value_vec();
  SVolField& strTsrMag    = *results[0];
  SVolField& dynSmagConst = *results[1];
  // No need to initialize strTsrMag or dynSmagConst. These will be filled in
  // with values further downstream
  
  const double eps = std::numeric_limits<double>::epsilon();
  
  // NOTE: hats denote test filetered. u is grid filtered
  
  //----------------------------------------------------------------------------
  // CALCULATE the filtered density, if needed. Filter(rho)
  //----------------------------------------------------------------------------
  SVolFldPtr rhoHat = SpatialFieldStore::get<SVolField>( dynSmagConst );
  SVolFldPtr invRhoHat = SpatialFieldStore::get<SVolField>( dynSmagConst );
  if (!isConstDensity_) {
    const SVolField& rho = rho_->field_ref();
    *rhoHat <<= (*boxFilterOp_)( rho );
    // pay attention to this. may require fine tuning.
    exOp_->apply_to_field(*rhoHat, 0.0);
    *invRhoHat <<= cond( *rhoHat<=2*eps, 1.0/ rho )
                       ( 1.0/ *rhoHat );
  }
  
  //----------------------------------------------------------------------------
  // CALCULATE test filtered staggered velocities. Filter(ui)
  //----------------------------------------------------------------------------
  const Vel1T& u = this->u_->field_ref();
  const Vel2T& v = this->v_->field_ref();
  const Vel3T& w = this->w_->field_ref();

  SpatFldPtr<Vel1T> uhat = SpatialFieldStore::get<Vel1T>( dynSmagConst );
  SpatFldPtr<Vel2T> vhat = SpatialFieldStore::get<Vel2T>( dynSmagConst );
  SpatFldPtr<Vel3T> what = SpatialFieldStore::get<Vel3T>( dynSmagConst );
  *uhat <<= (*xBoxFilterOp_)( u );
  *vhat <<= (*yBoxFilterOp_)( v );
  *what <<= (*zBoxFilterOp_)( w );
  
  // extrapolate filtered, staggered, velocities from the interior.
  xexOp_->apply_to_field(*uhat);
  yexOp_->apply_to_field(*vhat);
  zexOp_->apply_to_field(*what);
  
  //----------------------------------------------------------------------------
  // CALCULATE cell centered velocities
  //----------------------------------------------------------------------------
  SVolVecT rhoVelCC;
  ALLOCATE_VECTOR_FIELD(rhoVelCC); // allocate cell centered velocity field
  
  if( isConstDensity_ ){
    *rhoVelCC[0] <<= (*vel1InterpOp_)(u);  // u cell centered
    *rhoVelCC[1] <<= (*vel2InterpOp_)(v);  // v cell centered
    *rhoVelCC[2] <<= (*vel3InterpOp_)(w);  // w cell centered
  }
  else{
    const SVolField& rho = rho_->field_ref();
    *rhoVelCC[0] <<= rho * (*vel1InterpOp_)(u);  // u cell centered
    *rhoVelCC[1] <<= rho * (*vel2InterpOp_)(v);  // v cell centered
    *rhoVelCC[2] <<= rho * (*vel3InterpOp_)(w);  // w cell centered
  }

  // extrapolate cell centered velocities to ghost cells
  for (int i=0; i<3; i++) {
    exOp_->apply_to_field(*rhoVelCC[i]);
  }

  //----------------------------------------------------------------------------
  // CALCULATE test-filtered, cell centered, velocities
  //----------------------------------------------------------------------------
  // NOTE: since cell centered velocities have already been extrapolated to ghost cells
  // we should have valid test-filtered data at the interior.
  SVolVecT rhoVelCCHat;
  ALLOCATE_VECTOR_FIELD(rhoVelCCHat);
  for (int i=0; i<3; i++) {
    rhoVelCCHat.push_back(SpatialFieldStore::get<SVolField>( dynSmagConst )); // allocate spatial field pointer
    *rhoVelCCHat[i] <<= (*boxFilterOp_)( *rhoVelCC[i] );
  }

  //----------------------------------------------------------------------------
  // CALCULATE test-filtered velocity products - those will be used in the Leonard stress tensor
  // that is: Filter(rho ui uj)
  //----------------------------------------------------------------------------
  SVolTensorT uiujhat(3);
  ALLOCATE_TENSOR_FIELD(uiujhat);
  // uu = uiujhat[0][0], uv = uiujhat[0][1], uw = uiujhat[0][2]
  // vv = uiujhat[1][0], vw = uiujhat[1][1]
  // ww = uiujhat[2][0]  
  int jmin=0;
  for( int i=0; i<3; ++i ){
    for( int j=jmin; j<3; ++j ){
      //exOp_->apply_to_field(*rhoVelCC[i] * *rhoVelCC[j]); // we may need this...
      *uiujhat[i][j-jmin] <<= (*boxFilterOp_)( *rhoVelCC[i] * *rhoVelCC[j] ) ;
    }
    ++jmin;
  }

  //----------------------------------------------------------------------------
  // CALCULATE the Leonard stress tensor, Lij = Filter(rho ui uj) - 1/Filter(rho) * Filter(rho ui) * Filter(rho uj)
  //----------------------------------------------------------------------------  
  SVolTensorT Lij(3);
  ALLOCATE_TENSOR_FIELD(Lij);
  // L11 = Lij[0][0], L12 = Lij[0][1], L13 = Lij[0][2]
  // L22 = Lij[1][0], L23 = Lij[1][1]
  // L33 = Lij[2][0]
  jmin=0;
  for( int i=0; i<3; ++i ){
    for( int j=jmin; j<3; ++j ){
      if( isConstDensity_ ){
        *Lij[i][j-jmin] <<= *uiujhat[i][j-jmin] - *rhoVelCCHat[i] * *rhoVelCCHat[j];
      }
      else{
        *Lij[i][j-jmin] <<= *uiujhat[i][j-jmin] - *invRhoHat * *rhoVelCCHat[i] * *rhoVelCCHat[j];
      }
    }
    ++jmin;
  }

  // release uiujhat since it is no longer needed.  It is now invalid.
  BOOST_FOREACH( SVolVecT& fvec, uiujhat ){
    BOOST_FOREACH( SVolPtr& f, fvec ){
      f.detach();
    }
  }

  //----------------------------------------------------------------------------
  // CALCULATE test-filtered strain tensor.Filter(rho)*Filter(S)*Filter(Sij) (second term in the Mij Tensor. See Wasatch documentation.
  //----------------------------------------------------------------------------
  SVolTensorT Shatij(3);
  ALLOCATE_TENSOR_FIELD(Shatij);
  // Shat11 = Shatij[0][0], Shat12 = Shatij[0][1], S13 = Shatij[0][2]
  // Shat22 = Shatij[1][0], Shat23 = Shatij[1][1]
  // Shat33 = Shatij[2][0]
  this->calculate_strain_tensor_components(strTsrMag,*uhat,*vhat,*what,*Shatij[0][0],*Shatij[0][1],*Shatij[0][2],*Shatij[1][0],*Shatij[1][1],*Shatij[2][0]);
  
  // release some temporary fields
  uhat.detach(); vhat.detach(); what.detach();

  if( !isConstDensity_ ) strTsrMag <<= *rhoHat * strTsrMag;

  jmin=0;
  for( int i=0; i<3; ++i ){
    for( int j=jmin; j<3; ++j ){
      // now multiply S by Filter{S_ij}
      *Shatij[i][j-jmin] <<= strTsrMag * *Shatij[i][j-jmin];
    }
    ++jmin;
  }
  // we now have: Shatij = Filter(rho) * Filter(S) * Filter(Sij). This is the second term in the Mij tensor
  
  //----------------------------------------------------------------------------
  // CALCULATE grid-filtered strain tensor
  //----------------------------------------------------------------------------
  // NOTE: This STEP should be done AFTER the test-filtered rate of strain because we
  // are saving the grid-filtered StrainTensorMagnitude as a computed field for
  // this expression BUT we're using the SAME field_ref (i.e. strTsrMag) for both
  // quantities to reduce memory usage.
  
  SVolTensorT Sij(3);
  // S11 = Sij[0][0], S12 = Sij[0][1], S13 = Sij[0][2]
  // S22 = Sij[1][0], S23 = Sij[1][1]
  // S33 = Sij[2][0]
  ALLOCATE_TENSOR_FIELD(Sij);
  this->calculate_strain_tensor_components(strTsrMag,u,v,w,*Sij[0][0],*Sij[0][1],*Sij[0][2],*Sij[1][0],*Sij[1][1],*Sij[2][0]);
  exOp_->apply_to_field(strTsrMag, 0.0);
  
  jmin=0;
  {
    SVolFldPtr tmp = SpatialFieldStore::get<SVolField>( strTsrMag );
    for( int i=0; i<3; ++i ){
      for( int j=jmin; j<3; ++j ){
        exOp_->apply_to_field(*Sij[i][j-jmin]);
        *tmp <<= strTsrMag * *Sij[i][j-jmin];
        // now multiply \bar{S} by \bar{S_ij} and test filter them
        if( isConstDensity_ ){
          *Sij[i][j-jmin] <<= (*boxFilterOp_)( *tmp );
        }
        else{
          const SVolField& rho = rho_->field_ref();
          *Sij[i][j-jmin] <<= (*boxFilterOp_)( rho * *tmp );
        }
      }
      ++jmin;
    }
  }
  // note that we now have S_ij = \FILTER { rho |\bar{S}| \bar{S_ij} } (see wasatch documentation for notation).
  
  // !!!!!!!!!!!!! ATTENTION:
  // DO NOT MODIFY strTsrMag - at this point, strTsrMag stores
  // the magnitude of the grid-filtered strain tensor, strTsrMag = Sqrt(2 * S_ij * S_ij),
  // and this quantity will be used in calculating the Turbulent Viscosity.
  // !!!!!!!!!!!!! END ATTENATION

  //----------------------------------------------------------------------------
  // CALCULATE M_ij
  //----------------------------------------------------------------------------
  const double filRatio = 9.0; // this quantity is hard-coded at the moment
  // because the dynamic model is allowed to work only in 3D
  SVolTensorT Mij(3);
  // M11 = Mij[0][0], M12 = Mij[0][1], M13 = Mij[0][2]
  // M22 = Mij[1][0], M23 = Mij[1][1]
  // M33 = Mij[2][0]
  ALLOCATE_TENSOR_FIELD(Mij);
  jmin=0;
  for( int i=0; i<3; ++i ){
    for( int j=jmin; j<3; ++j ){
      *Mij[i][j-jmin] <<= *Sij[i][j-jmin] - filRatio * *Shatij[i][j-jmin];
    }
    ++jmin;
  }
  
  // release Shatij - it is no longer valid!
  BOOST_FOREACH( SVolVecT& fvec, Shatij ){
    BOOST_FOREACH( SVolPtr& f, fvec ){
      f.detach();
    }
  }
  // release Sij - it is no longer valid!
  BOOST_FOREACH( SVolVecT& fvec, Sij ){
    BOOST_FOREACH( SVolPtr& f, fvec ){
      f.detach();
    }
  }

  //----------------------------------------------------------------------------
  // CALCULATE the dynamic constant!
  //----------------------------------------------------------------------------
  SVolFldPtr LM = SpatialFieldStore::get<SVolField>( dynSmagConst );
  //  *LM <<= 0.0;
  //  *LM = *L11 * *M11 + *L22 * *M22 + *L33 * *M33 + 2.0 * (*L12 * *M12 + *L13 * *M13  + *L23 * *M23);
  *LM <<=   *Lij[0][0] * *Mij[0][0] // L11 * M11
          + *Lij[1][0] * *Mij[1][0] // L22 * M22
          + *Lij[2][0] * *Mij[2][0] // L33 * M33
          + 2.0 * (  *Lij[0][1] * *Mij[0][1] // L12 * M12
                   + *Lij[0][2] * *Mij[0][2] // L13 * M13
                   + *Lij[1][1] * *Mij[1][1] // L23 * M23
                   );
  
  BOOST_FOREACH( SVolVecT& fvec, Lij ){
    BOOST_FOREACH( SVolPtr& f, fvec ){
      f.detach();
    }
  }


  // filtering the numerator and denominator here requires an MPI communication
  // at patch boundaries for it to work. We could potentially split this expression
  // and cleave things... but I don't think this is worth the effort at all...
  if( doExtraFiltering_ ){
    SVolFldPtr tmp = SpatialFieldStore::get<SVolField>( *LM );
    exOp_->apply_to_field(*LM, 0.0);
    *tmp <<= (*boxFilterOp_)(*LM);
    *LM <<= *tmp;
  }
  
  SVolFldPtr MM = SpatialFieldStore::get<SVolField>( dynSmagConst );
  //  *MM <<= 0.0;
  //  *MM = *M11 * *M11 + *M22 * *M22 + *M33 * *M33 + 2.0 * (*M12 * *M12 + *M13 * *M13 + *M23 * *M23);
  *MM <<=   *Mij[0][0] * *Mij[0][0] // M11 * M11
          + *Mij[1][0] * *Mij[1][0] // M22 * M22
          + *Mij[2][0] * *Mij[2][0] // M33 * M33
          + 2.0 * (  *Mij[0][1] * *Mij[0][1] // M12 * M12
                   + *Mij[0][2] * *Mij[0][2] // M13 * M13
                   + *Mij[1][1] * *Mij[1][1] // M23 * M23
                   );

  if( doExtraFiltering_ ){
    SVolFldPtr tmp = SpatialFieldStore::get<SVolField>( *MM );
    exOp_->apply_to_field(*MM, 0.0);
    *tmp <<= (*boxFilterOp_)(*MM);
    *MM <<= *tmp;
  }
  
  // PREVENT BACKSCATTER, i.e. c_smag >= 0.0  : given that MM (denominator) is positive,
  // we only need to check when LM < 0.0. If LM < 0, then set the dynamic coefficient to 0
  // PREVENT NANs: in the event that MM = 0, we can safely set the dynamic constant
  // to zero since this means that our test and grid filtered fields are equally
  // resolved (within the gradient diffusion modeling assumption).
  dynSmagConst <<= cond( *LM < 0.0 || *MM <= 2.0*eps , 0.0 )
                       ( 0.5 * *LM / *MM );
}

//------------------------------------------------------
//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>

template class DynamicSmagorinskyCoefficient< SpatialOps::SVolField,
                                               SpatialOps::XVolField,
                                               SpatialOps::YVolField,
                                               SpatialOps::ZVolField>;

template class DynamicSmagorinskyCoefficient< SpatialOps::SVolField,
                                               SpatialOps::SVolField,
                                               SpatialOps::SVolField,
                                               SpatialOps::SVolField >;
