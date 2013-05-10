/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

DynamicSmagorinskyCoefficient::
DynamicSmagorinskyCoefficient( const Expr::Tag& vel1Tag,
                               const Expr::Tag& vel2Tag,
                               const Expr::Tag& vel3Tag,
                               const Expr::Tag& rhoTag,
                               const bool isConstDensity )
: StrainTensorBase(vel1Tag,vel2Tag,vel3Tag),
  rhot_          ( rhoTag       ),
  isConstDensity_(isConstDensity),
  doExtraFiltering_(false)
{
//  std::cout << "is constant density ? = " << isConstDensity << std::endl;
  // Disallow using the dynamic model in 1 or 2 dimensions
  if (!(doX_ && doY_ && doZ_)) {
    std::ostringstream msg;
    msg << "WARNING: You cannot use the Dynamic Smagorinsky Model in one or two dimensions. Please revise your input file and make sure that you specify all three velocity/momentum components." << std::endl;
    std::cout << msg.str() << std::endl;
    throw std::runtime_error(msg.str());
  }
}

//--------------------------------------------------------------------

DynamicSmagorinskyCoefficient::
~DynamicSmagorinskyCoefficient()
{}

//--------------------------------------------------------------------

void
DynamicSmagorinskyCoefficient::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  StrainTensorBase::advertise_dependents(exprDeps);
  if(!isConstDensity_)
    exprDeps.requires_expression( rhot_ );
}

//--------------------------------------------------------------------

void
DynamicSmagorinskyCoefficient::
bind_fields( const Expr::FieldManagerList& fml )
{
  StrainTensorBase::bind_fields(fml);
  if(!isConstDensity_)
    rho_ = &fml.field_manager<SVolField>().field_ref(rhot_);
}

//--------------------------------------------------------------------

void
DynamicSmagorinskyCoefficient::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  StrainTensorBase::bind_operators(opDB);
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

void
DynamicSmagorinskyCoefficient::
evaluate()
{
  using namespace SpatialOps;
  typedef SpatFldPtr<SVolField> SVolFldPtr;
  typedef std::vector<SVolField*> SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();
  SVolField& strTsrMag = *results[0];
  SVolField& dynSmagConst = *results[1];
  // strTsrMag <<= 0.0; // No need to initialize this. There is a function call downstream that will initialize it
  dynSmagConst <<= 0.0;
  
  const double eps = std::numeric_limits<double>::epsilon();
  
  // NOTE: hats denote test filetered. u is grid filtered
  
  //----------------------------------------------------------------------------
  // CALCULATE the filtered density, if needed. Filter(rho)
  //----------------------------------------------------------------------------
  SpatFldPtr<SVolField> rhoHat = SpatialFieldStore::get<SVolField>( dynSmagConst );
  SpatFldPtr<SVolField> invRhoHat = SpatialFieldStore::get<SVolField>( dynSmagConst );
  if (!isConstDensity_) {
    boxFilterOp_->apply_to_field( *rho_, *rhoHat );
    // pay attention to this. may require fine tuning.
    exOp_->apply_to_field(*rhoHat, 0.0);
    *invRhoHat <<= cond( *rhoHat<=2*eps, 1.0/ *rho_ )
                       ( 1.0/ *rhoHat );
  }
  
  //----------------------------------------------------------------------------
  // CALCULATE test filtered staggered velocities. Filter(ui)
  //----------------------------------------------------------------------------
  SpatFldPtr<XVolField> uhat = SpatialFieldStore::get<XVolField>( dynSmagConst );
  SpatFldPtr<YVolField> vhat = SpatialFieldStore::get<YVolField>( dynSmagConst );
  SpatFldPtr<ZVolField> what = SpatialFieldStore::get<ZVolField>( dynSmagConst );
  xBoxFilterOp_->apply_to_field( *vel1_, *uhat );
  yBoxFilterOp_->apply_to_field( *vel2_, *vhat );
  zBoxFilterOp_->apply_to_field( *vel3_, *what );
  
  // extrapolate filtered, staggered, velocities from the interior.
  xexOp_->apply_to_field(*uhat);
  yexOp_->apply_to_field(*vhat);
  zexOp_->apply_to_field(*what);
  
  //----------------------------------------------------------------------------
  // CALCULATE cell centered velocities
  //----------------------------------------------------------------------------
  std::vector< SpatFldPtr<SVolField> > velcc;
  ALLOCATE_VECTOR_FIELD(velcc); // allocate cell centered velocity field
  
  vel1InterpOp_->apply_to_field( *vel1_, *velcc[0] );  // u cell centered
  vel2InterpOp_->apply_to_field( *vel2_, *velcc[1] );  // v cell centered
  vel3InterpOp_->apply_to_field( *vel3_, *velcc[2] );  // w cell centered
  
  if (!isConstDensity_) {
    for (int i=0; i<3; i++) {
      *velcc[i] <<= *rho_ * *velcc[i]; // velcc is now = rho * velcc
    }
  }

  // extrapolate cell centered velocities to ghost cells
  for (int i=0; i<3; i++) {
    exOp_->apply_to_field(*velcc[i]);
  }

  //----------------------------------------------------------------------------
  // CALCULATE test-filtered, cell centered, velocities
  //----------------------------------------------------------------------------
  // NOTE: since cell centered velocities have already been extrapolated to ghost cells
  // we should have valid test-filtered data at the interior.
  std::vector< SpatFldPtr<SVolField> > velcchat;
  ALLOCATE_VECTOR_FIELD(velcchat);
  for (int i=0; i<3; i++) {
    velcchat.push_back(SpatialFieldStore::get<SVolField>( dynSmagConst )); // allocate spatial field pointer
    boxFilterOp_->apply_to_field( *velcc[i], *velcchat[i] );
  }

  //----------------------------------------------------------------------------
  // CALCULATE test-filtered velocity products - those will be used in the Leonard stress tensor
  // that is: Filter(rho ui uj)
  //----------------------------------------------------------------------------  
  typedef std::vector< SpatFldPtr<SVolField> > vecSvol;
  std::vector< vecSvol > uiujhat(3);
  ALLOCATE_TENSOR_FIELD(uiujhat);
  // uu = uiujhat[0][0], uv = uiujhat[0][1], uw = uiujhat[0][2]
  // vv = uiujhat[1][0], vw = uiujhat[1][1]
  // ww = uiujhat[2][0]  
  SpatFldPtr<SVolField> tmp   = SpatialFieldStore::get<SVolField>( dynSmagConst );
  int jmin=0;
  for (int i=0; i<3; i++) {
    for (int j=jmin; j<3; j++) {
      *tmp <<= *velcc[i] * *velcc[j];
      //exOp_->apply_to_field(*tmp); // we may need this...
      boxFilterOp_->apply_to_field(*tmp, *uiujhat[i][j - jmin]);
    }
    jmin++;
  }

  //----------------------------------------------------------------------------
  // CALCULATE the Leonard stress tensor, Lij = Filter(rho ui uj) - 1/Filter(rho) * Filter(rho ui) * Filter(rho uj)
  //----------------------------------------------------------------------------  
  std::vector< vecSvol > Lij(3);
  ALLOCATE_TENSOR_FIELD(Lij);
  // L11 = Lij[0][0], L12 = Lij[0][1], L13 = Lij[0][2]
  // L22 = Lij[1][0], L23 = Lij[1][1]
  // L33 = Lij[2][0]
  jmin=0;
  for (int i=0; i<3; i++) {
    for (int j=jmin; j<3; j++) {
      *Lij[i][j-jmin] <<= cond( isConstDensity_, *uiujhat[i][j-jmin] - *velcchat[i] * *velcchat[j] )
                              ( *uiujhat[i][j-jmin] - *invRhoHat * *velcchat[i] * *velcchat[j] );
    }
    jmin++;
  }

  //----------------------------------------------------------------------------
  // CALCULATE test-filtered strain tensor.Filter(rho)*Filter(S)*Filter(Sij) (second term in the Mij Tensor. See Wasatch documentation.
  //----------------------------------------------------------------------------
  std::vector< vecSvol > Shatij(3);
  ALLOCATE_TENSOR_FIELD(Shatij);
  // Shat11 = Shatij[0][0], Shat12 = Shatij[0][1], S13 = Shatij[0][2]
  // Shat22 = Shatij[1][0], Shat23 = Shatij[1][1]
  // Shat33 = Shatij[2][0]
  calculate_strain_tensor_components(strTsrMag,*uhat,*vhat,*what,*Shatij[0][0],*Shatij[0][1],*Shatij[0][2],*Shatij[1][0],*Shatij[1][1],*Shatij[2][0]);
  
  if(!isConstDensity_) strTsrMag <<= *rhoHat * strTsrMag;
  jmin=0;
  for (int i=0; i<3; i++) {
    for (int j=jmin; j<3; j++) {
      // now multiply S by Filter{S_ij}
      *Shatij[i][j-jmin] <<= strTsrMag * *Shatij[i][j-jmin];
    }
    jmin++;
  }
  // we now have: Shatij = Filter(rho) * Filter(S) * Filter(Sij). This is the second term in the Mij tensor
  
  //----------------------------------------------------------------------------
  // CALCULATE grid-filtered strain tensor
  //----------------------------------------------------------------------------
  // NOTE: This STEP should be done AFTER the test-filtered rate of strain because we
  // are saving the grid-filtered StrainTensorMagnitude as a computed field for
  // this expression BUT we're using the SAME field_ref (i.e. strTsrMag) for both
  // quantities to reduce memory usage.
  
  std::vector< vecSvol > Sij(3);
  // S11 = Sij[0][0], S12 = Sij[0][1], S13 = Sij[0][2]
  // S22 = Sij[1][0], S23 = Sij[1][1]
  // S33 = Sij[2][0]
  ALLOCATE_TENSOR_FIELD(Sij);
  calculate_strain_tensor_components(strTsrMag,*vel1_,*vel2_,*vel3_,*Sij[0][0],*Sij[0][1],*Sij[0][2],*Sij[1][0],*Sij[1][1],*Sij[2][0]);
  exOp_->apply_to_field(strTsrMag, 0.0);
  
  jmin=0;
  for (int i=0; i<3; i++) {
    for (int j=jmin; j<3; j++) {      
      exOp_->apply_to_field(*Sij[i][j-jmin]);
      // now multiply \bar{S} by \bar{S_ij} and test filter them
      *tmp <<= strTsrMag * *Sij[i][j-jmin];
      if (!isConstDensity_) *tmp <<= *rho_ * *tmp;
      boxFilterOp_->apply_to_field(*tmp, *Sij[i][j-jmin]);
    }
    jmin++;
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
  std::vector< vecSvol > Mij(3);
  // M11 = Mij[0][0], M12 = Mij[0][1], M13 = Mij[0][2]
  // M22 = Mij[1][0], M23 = Mij[1][1]
  // M33 = Mij[2][0]
  ALLOCATE_TENSOR_FIELD(Mij);
  jmin=0;
  for (int i=0; i<3; i++) {
    for (int j=jmin; j<3; j++) {
      *Mij[i][j-jmin] <<= *Sij[i][j-jmin] - filRatio * *Shatij[i][j-jmin];
    }
    jmin++;
  }

  
  //----------------------------------------------------------------------------
  // CALCULATE the dynamic constant!
  //----------------------------------------------------------------------------
  SpatFldPtr<SVolField> LM = SpatialFieldStore::get<SVolField>( dynSmagConst );
  //  *LM <<= 0.0;
  //  *LM = *L11 * *M11 + *L22 * *M22 + *L33 * *M33 + 2.0 * (*L12 * *M12 + *L13 * *M13  + *L23 * *M23);
  *LM <<=   *Lij[0][0] * *Mij[0][0] // L11 * M11
          + *Lij[1][0] * *Mij[1][0] // L22 * M22
          + *Lij[2][0] * *Mij[2][0] // L33 * M33
          + 2.0 * (  *Lij[0][1] * *Mij[0][1] // L12 * M12
                   + *Lij[0][2] * *Mij[0][2] // L13 * M13
                   + *Lij[1][1] * *Mij[1][1] // L23 * M23
                   );
  
  // filtering the numerator and denominator here requires an MPI communication
  // at patch boundaries for it to work. We could potentially split this expression
  // and cleave things... but I don't think this is worth the effort at all...
  if (doExtraFiltering_) {
    exOp_->apply_to_field(*LM, 0.0);
    boxFilterOp_->apply_to_field(*LM,*tmp);
    *LM <<= *tmp;
  }
  
  SpatFldPtr<SVolField> MM = SpatialFieldStore::get<SVolField>( dynSmagConst );
  //  *MM <<= 0.0;
  //  *MM = *M11 * *M11 + *M22 * *M22 + *M33 * *M33 + 2.0 * (*M12 * *M12 + *M13 * *M13 + *M23 * *M23);
  *MM <<=   *Mij[0][0] * *Mij[0][0] // M11 * M11
          + *Mij[1][0] * *Mij[1][0] // M22 * M22
          + *Mij[2][0] * *Mij[2][0] // M33 * M33
          + 2.0 * (  *Mij[0][1] * *Mij[0][1] // M12 * M12
                   + *Mij[0][2] * *Mij[0][2] // M13 * M13
                   + *Mij[1][1] * *Mij[1][1] // M23 * M23
                   );

  if (doExtraFiltering_) {
    exOp_->apply_to_field(*MM, 0.0);
    boxFilterOp_->apply_to_field(*MM,*tmp);
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
