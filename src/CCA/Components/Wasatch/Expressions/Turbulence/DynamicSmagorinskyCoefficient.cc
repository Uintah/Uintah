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

//********************************************************************
// DYNAMIC SMAGORINSKY COEFFICIENT
//********************************************************************

DynamicSmagorinskyCoefficient::
DynamicSmagorinskyCoefficient( const Expr::Tag vel1Tag,
                               const Expr::Tag vel2Tag,
                               const Expr::Tag vel3Tag,
                               const Expr::Tag rhoTag )
: StrainTensorBase(vel1Tag,vel2Tag,vel3Tag),
  rhot_       ( rhoTag       )
{
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
  exprDeps.requires_expression( rhot_       );  
}

//--------------------------------------------------------------------

void
DynamicSmagorinskyCoefficient::
bind_fields( const Expr::FieldManagerList& fml )
{
  StrainTensorBase::bind_fields(fml);
  rho_  =  &fml.field_manager<SVolField>().field_ref(rhot_);
}

//--------------------------------------------------------------------

void
DynamicSmagorinskyCoefficient::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  StrainTensorBase::bind_operators(opDB);
  exOp_    = opDB.retrieve_operator<ExOpT>();
  xexOp_    = opDB.retrieve_operator<XExOpT>();
  yexOp_    = opDB.retrieve_operator<YExOpT>();
  zexOp_    = opDB.retrieve_operator<ZExOpT>();
  
  BoxFilterOp_    = opDB.retrieve_operator<BoxFilterT>();
  xBoxFilterOp_   = opDB.retrieve_operator<XBoxFilterT>();
  yBoxFilterOp_   = opDB.retrieve_operator<YBoxFilterT>();
  zBoxFilterOp_   = opDB.retrieve_operator<ZBoxFilterT>();
  vel1InterpOp_   = opDB.retrieve_operator<Vel1InterpT>();
  vel2InterpOp_   = opDB.retrieve_operator<Vel2InterpT>();
  vel3InterpOp_   = opDB.retrieve_operator<Vel3InterpT>();
}

//--------------------------------------------------------------------

void
DynamicSmagorinskyCoefficient::
evaluate()
{
  using namespace SpatialOps;
  typedef std::vector<SVolField*> SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();
  SVolField& strTsrMag = *results[0];
  SVolField& DynSmagConst = *results[1];
  // strTsrMag <<= 0.0; // No need to initialize this. There is a function call downstream that will initialize it
  DynSmagConst <<= 0.0;
  
  // NOTE: hats denote test filetered. u is grid filtered
  
  // CALCULATE test filtered staggered velocities/momentum
  SpatFldPtr<XVolField> uhat = SpatialFieldStore::get<XVolField>( DynSmagConst );
  SpatFldPtr<YVolField> vhat = SpatialFieldStore::get<YVolField>( DynSmagConst );
  SpatFldPtr<ZVolField> what = SpatialFieldStore::get<ZVolField>( DynSmagConst );  
  xBoxFilterOp_->apply_to_field( *vel1_, *uhat );
  yBoxFilterOp_->apply_to_field( *vel2_, *vhat );
  zBoxFilterOp_->apply_to_field( *vel3_, *what );
  // extrapolate filtered, staggered, velocities from the interior.
  xexOp_->apply_to_field(*uhat);
  yexOp_->apply_to_field(*vhat);
  zexOp_->apply_to_field(*what);
  
  // CALCULATE cell centered velocities
  SpatFldPtr<SVolField> ucc = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> vcc = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> wcc = SpatialFieldStore::get<SVolField>( DynSmagConst );
  vel1InterpOp_->apply_to_field( *vel1_, *ucc );  // u cell centered
  vel2InterpOp_->apply_to_field( *vel2_, *vcc );  // v cell centered
  vel3InterpOp_->apply_to_field( *vel3_, *wcc );  // w cell centered
  // extrapolate cell centered velocities to ghost cells
  exOp_->apply_to_field(*ucc);
  exOp_->apply_to_field(*vcc);
  exOp_->apply_to_field(*wcc);
  
  // CALCULATE test-filtered velocities
  // NOTE: since cell centered velocities have already been extrapolated to ghost cells
  // we should have valid test-filtered data at the interior.
  SpatFldPtr<SVolField> ucchat = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> vcchat = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> wcchat = SpatialFieldStore::get<SVolField>( DynSmagConst );  
  BoxFilterOp_->apply_to_field( *ucc, *ucchat );
  BoxFilterOp_->apply_to_field( *vcc, *vcchat );
  BoxFilterOp_->apply_to_field( *wcc, *wcchat );
  
  // CALCULATE test-filtered velocity products - those will be used in the Leonard stress tensor
  SpatFldPtr<SVolField> tmp   = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> uuhat = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> uvhat = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> uwhat = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> vvhat = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> vwhat = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> wwhat = SpatialFieldStore::get<SVolField>( DynSmagConst );
  *tmp <<= *ucc * *ucc;
  BoxFilterOp_->apply_to_field(*tmp, *uuhat);
  *tmp <<= *ucc * *vcc;
  BoxFilterOp_->apply_to_field(*tmp, *uvhat);
  *tmp <<= *ucc * *wcc;
  BoxFilterOp_->apply_to_field(*tmp, *uwhat);
  *tmp <<= *vcc * *vcc;
  BoxFilterOp_->apply_to_field(*tmp, *vvhat);
  *tmp <<= *vcc * *wcc;
  BoxFilterOp_->apply_to_field(*tmp, *vwhat);
  *tmp <<= *wcc * *wcc;
  BoxFilterOp_->apply_to_field(*tmp, *wwhat);
  
  // calculate the Leonard stress tensor, Lij
  SpatFldPtr<SVolField> L11 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  *L11 <<= *uuhat - *ucchat * *ucchat;
  SpatFldPtr<SVolField> L12 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  *L12 <<= *uvhat - *ucchat * *vcchat;
  SpatFldPtr<SVolField> L13 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  *L13 <<= *uwhat - *ucchat * *wcchat;
  SpatFldPtr<SVolField> L22 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  *L22 <<= *vvhat - *vcchat * *vcchat;
  SpatFldPtr<SVolField> L23 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  *L23 <<= *vwhat - *vcchat * *wcchat;
  SpatFldPtr<SVolField> L33 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  *L33 <<= *wwhat - *wcchat * *wcchat;

  // OPTIONAL
//  SpatFldPtr<SVolField> Lkk = SpatialFieldStore::get<SVolField>( DynSmagConst );
//  *Lkk <<= 1.0/3.0*(*L11 + *L22 + *L33);
//  *L11 <<= *L11 - *Lkk;
//  *L22 <<= *L22 - *Lkk;
//  *L33 <<= *L33 - *Lkk;

  //----------------------------------------------------------------------------
  // CALCULATE test-filtered strain tensor
  //----------------------------------------------------------------------------
  SpatFldPtr<SVolField> Shat11 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> Shat12 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> Shat13 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> Shat22 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> Shat23 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> Shat33 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  calculate_strain_tensor_components(strTsrMag,*uhat,*vhat,*what,*Shat11,*Shat12,*Shat13,*Shat22,*Shat23,*Shat33);
  
  // now multiply \filter{S} by \filter{S_ij} 
  *Shat11 <<= strTsrMag * *Shat11;
  *Shat12 <<= strTsrMag * *Shat12;
  *Shat13 <<= strTsrMag * *Shat13;
  *Shat22 <<= strTsrMag * *Shat22;
  *Shat23 <<= strTsrMag * *Shat23;
  *Shat33 <<= strTsrMag * *Shat33;    

  //----------------------------------------------------------------------------
  // CALCULATE grid-filtered strain tensor
  //----------------------------------------------------------------------------
  // NOTE: This should be done AFTER the test-filtered rate of strain because we
  // are saving the grid-filtered StrainTensorMagnitude as a computed field for
  // this expression BUT we're using the SAME field_ref (i.e. strTsrMag) for both
  // quantities to reduce memory usage.
  SpatFldPtr<SVolField> S11 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> S12 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> S13 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> S22 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> S23 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  SpatFldPtr<SVolField> S33 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  calculate_strain_tensor_components(strTsrMag,*vel1_,*vel2_,*vel3_,*S11,*S12,*S13,*S22,*S23,*S33);
  exOp_->apply_to_field(strTsrMag);
  exOp_->apply_to_field(*S11);
  exOp_->apply_to_field(*S12);
  exOp_->apply_to_field(*S13);
  exOp_->apply_to_field(*S22);
  exOp_->apply_to_field(*S23);
  exOp_->apply_to_field(*S33);

  // !!!!!!!!!!!!! ATTENTION:
  // DO NOT MODIFY strTsrMag - at this point, strTsrMag stores
  // the magnitude of the grid-filtered strain tensor, strTsrMag = Sqrt(2 * S_ij * S_ij),
  // and this quantity will be used in calculating the Turbulent Viscosity.
  // !!!!!!!!!!!!! END ATTENATION

  // now multiply \bar{S} by \bar{S_ij} and test filter them
//  *tmp <<= 0.0;
  
  *tmp <<= strTsrMag * *S11;
  BoxFilterOp_->apply_to_field(*tmp, *S11);
  
  *tmp <<= strTsrMag * *S12;
  BoxFilterOp_->apply_to_field(*tmp, *S12);
  
  *tmp <<= strTsrMag * *S13;
  BoxFilterOp_->apply_to_field(*tmp, *S13);
  
  *tmp <<= strTsrMag * *S22;
  BoxFilterOp_->apply_to_field(*tmp, *S22);
  
  *tmp <<= strTsrMag * *S23;
  BoxFilterOp_->apply_to_field(*tmp, *S23);
  
  *tmp <<= strTsrMag * *S33;
  BoxFilterOp_->apply_to_field(*tmp, *S33);
  // note that we now have S_ij = \FILTER { |\bar{S}| \bar{S_ij} }

  //----------------------------------------------------------------------------
  // CALCULATE M_ij
  //----------------------------------------------------------------------------
  const double filRatio = 9.0; // this quantity is hard-coded at the moment
  // because the dynamic model is allowed to work only in 3D
  SpatFldPtr<SVolField> M11 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  //*M11 <<= 0.0;
  *M11 <<= *S11 - filRatio * *Shat11;

  SpatFldPtr<SVolField> M12 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  //*M12 <<= 0.0;
  *M12 <<= *S12 - filRatio * *Shat12;

  SpatFldPtr<SVolField> M13 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  //*M13 <<= 0.0;
  *M13 <<= *S13 - filRatio * *Shat13;

  SpatFldPtr<SVolField> M22 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  //*M22 <<= 0.0;
  *M22 <<= *S22 - filRatio * *Shat22;
  
  SpatFldPtr<SVolField> M23 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  //*M23 <<= 0.0;
  *M23 <<= *S23 - filRatio * *Shat23;
  
  SpatFldPtr<SVolField> M33 = SpatialFieldStore::get<SVolField>( DynSmagConst );
  //*M33 <<= 0.0;
  *M33 <<= *S33 - filRatio * *Shat33;

  
  //----------------------------------------------------------------------------
  // CALCULATE the dynamic constant!
  //----------------------------------------------------------------------------
  SpatFldPtr<SVolField> LM = SpatialFieldStore::get<SVolField>( DynSmagConst );
  //*LM <<= 0.0;
  *LM <<= *L11 * *M11 + *L22 * *M22 + *L33 * *M33 + 2.0 * (*L12 * *M12 + *L13 * *M13  + *L23 * *M23);
  
  // filtering the numerator and denominator here requires an MPI communication
  // at patch boundaries for it to work. We could potentially split this expression
  // and cleave things... but I don't think this is worth the effort at all...
//  *tmp<<= 0.0;
//  BoxFilterOp_->apply_to_field(*num,*tmp);
//  *num <<= *tmp;
  
  SpatFldPtr<SVolField> MM = SpatialFieldStore::get<SVolField>( DynSmagConst );
  //*MM <<= 0.0;
  *MM <<= *M11 * *M11 + *M22 * *M22 + *M33 * *M33 + 2.0 * (*M12 * *M12 + *M13 * *M13 + *M23 * *M23);
//  *tmp<<= 0.0;
//  BoxFilterOp_->apply_to_field(*denom,*tmp);
//  *denom <<= *tmp;
  
  // PREVENT BACKSCATTER, i.e. c_smag >= 0.0: given that MM (denominator) is positive,
  // we only need to check when LM < 0.0. If LM < 0, then set the dynamic coefficient to 0
  // PREVENT NANs: in the event that MM = 0, we can safely set the dynamic constant
  // to zero since this means that our test and grid filtered fields are equally
  // resolved (within the gradient diffusion modeling assumption).
  const double eps = std::numeric_limits<double>::epsilon();
  DynSmagConst <<= cond( *LM < 0.0 || *MM <= 2.0*eps , 0.0 )
                       ( 0.5 * *LM / *MM );
}
