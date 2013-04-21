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

#include "StrainTensorBase.h"

StrainTensorBase::
StrainTensorBase( const Expr::Tag& vel1tag,
                 const Expr::Tag& vel2tag,
                 const Expr::Tag& vel3tag )
: Expr::Expression<SVolField>(),
  vel1t_( vel1tag ),
  vel2t_( vel2tag ),
  vel3t_( vel3tag ),
  doX_  ( vel1t_ != Expr::Tag() ),
  doY_  ( vel2t_ != Expr::Tag() ),
  doZ_  ( vel3t_ != Expr::Tag() )
{}

//--------------------------------------------------------------------

StrainTensorBase::
~StrainTensorBase()
{}

//--------------------------------------------------------------------

void
StrainTensorBase::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( doX_ ) exprDeps.requires_expression( vel1t_ );
  if( doY_ ) exprDeps.requires_expression( vel2t_ );
  if( doZ_ ) exprDeps.requires_expression( vel3t_ );
}

//--------------------------------------------------------------------

void
StrainTensorBase::
bind_fields( const Expr::FieldManagerList& fml )
{
  if ( doX_ ) vel1_ = &fml.field_manager<XVolField>().field_ref( vel1t_ );
  if ( doY_ ) vel2_ = &fml.field_manager<YVolField>().field_ref( vel2t_ );
  if ( doZ_ ) vel3_ = &fml.field_manager<ZVolField>().field_ref( vel3t_ );
}

//--------------------------------------------------------------------

void
StrainTensorBase::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  dudxOp_ = opDB.retrieve_operator<dudxT>();
  dudyOp_ = opDB.retrieve_operator<dudyT>();
  dudzOp_ = opDB.retrieve_operator<dudzT>();
  
  dvdxOp_ = opDB.retrieve_operator<dvdxT>();
  dvdyOp_ = opDB.retrieve_operator<dvdyT>();
  dvdzOp_ = opDB.retrieve_operator<dvdzT>();
  
  dwdxOp_ = opDB.retrieve_operator<dwdxT>();
  dwdyOp_ = opDB.retrieve_operator<dwdyT>();
  dwdzOp_ = opDB.retrieve_operator<dwdzT>();
  
  xyInterpOp_ = opDB.retrieve_operator<XYInterpT>();
  yxInterpOp_ = opDB.retrieve_operator<YXInterpT>();
  
  xzInterpOp_ = opDB.retrieve_operator<XZInterpT>();
  zxInterpOp_ = opDB.retrieve_operator<ZXInterpT>();
  
  yzInterpOp_ = opDB.retrieve_operator<YZInterpT>();
  zyInterpOp_ = opDB.retrieve_operator<ZYInterpT>();
}

//--------------------------------------------------------------------

void
StrainTensorBase::
evaluate()
{}

//--------------------------------------------------------------------

StrainTensorBase::
Builder::Builder( const Expr::Tag& result,
                 const Expr::Tag& vel1tag,
                 const Expr::Tag& vel2tag,
                 const Expr::Tag& vel3tag )
: ExpressionBuilder(result),
  v1t_( vel1tag ),
  v2t_( vel2tag ),
  v3t_( vel3tag )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
StrainTensorBase::Builder::build() const
{
  return new StrainTensorBase( v1t_, v2t_, v3t_ );
}

//--------------------------------------------------------------------

void
StrainTensorBase::
calculate_strain_tensor_components(SVolField& strTsrMag,
                                   const XVolField& u,
                                   const YVolField& v,
                                   const ZVolField& w,
                                   SVolField& S11,
                                   SVolField& S12,
                                   SVolField& S13,
                                   SVolField& S22,
                                   SVolField& S23,
                                   SVolField& S33)
{
  using namespace SpatialOps;
  strTsrMag <<= 0.0;
  //
  SpatFldPtr<SVolField> tmp1 = SpatialFieldStore::get<SVolField>( strTsrMag );
  SpatFldPtr<SVolField> tmp2 = SpatialFieldStore::get<SVolField>( strTsrMag );
  *tmp1 <<= 0.0;
  *tmp2 <<= 0.0;
  
  SpatFldPtr<structured::XSurfYField> xyfield = SpatialFieldStore::get<structured::XSurfYField>( strTsrMag );
  SpatFldPtr<structured::YSurfXField> yxfield = SpatialFieldStore::get<structured::YSurfXField>( strTsrMag );
  *xyfield <<= 0.0;
  *yxfield <<= 0.0;
  
  SpatFldPtr<structured::XSurfZField> xzfield = SpatialFieldStore::get<structured::XSurfZField>( strTsrMag );
  SpatFldPtr<structured::ZSurfXField> zxfield = SpatialFieldStore::get<structured::ZSurfXField>( strTsrMag );
  *xzfield <<= 0.0;
  *zxfield <<= 0.0;
  
  SpatFldPtr<structured::YSurfZField> yzfield = SpatialFieldStore::get<structured::YSurfZField>( strTsrMag );
  SpatFldPtr<structured::ZSurfYField> zyfield = SpatialFieldStore::get<structured::ZSurfYField>( strTsrMag );
  *yzfield <<= 0.0;
  *zyfield <<= 0.0;
  //
  //-------------------------
  S11 <<= 0.0;
  dudxOp_->apply_to_field( u, S11 );     // S_11 = 0.5 * (du/dx + du/dx) = du/dx
  strTsrMag <<= strTsrMag + S11 * S11;     // S_11 * S_11
  
  S22 <<= 0.0;
  dvdyOp_->apply_to_field( v, S22 );     // S_22 = 0.5 * (dv/dy + dv/dy) = dv/dy
  strTsrMag <<= strTsrMag + S22 * S22;     // S_22 * S_22
  
  S33 <<= 0.0;
  dwdzOp_->apply_to_field( w, S33 );     // S_33 = 0.5 * (dw/dz + dw/dz) = dwdz
  strTsrMag <<= strTsrMag + S33 * S33;   // S_33 * S_33
  
  //-------------------------
  S12 <<= 0.0;
  dudyOp_->apply_to_field( u, *xyfield );    // du/dy
  xyInterpOp_->apply_to_field( *xyfield, *tmp1);  // interpolate to scalar cells
  
  dvdxOp_->apply_to_field( v, *yxfield );    // dv/dx
  yxInterpOp_->apply_to_field( *yxfield, *tmp2);  // interpolate to scalar cells
  
  S12 <<= 0.5 * (*tmp1 + *tmp2);                         // S_12 = S_21 = 0.5 * (du/dy + dv/dx)
  strTsrMag <<= strTsrMag + 2.0 * S12 * S12; // 2*S_12 * S_12 + 2*S_21 * S_21 = 4*S_12*S_12
  
  //-------------------------
  S13 <<= 0.0;
  dudzOp_->apply_to_field( u, *xzfield );    // du/dz
  xzInterpOp_->apply_to_field( *xzfield, *tmp1);
  
  dwdxOp_->apply_to_field( w, *zxfield );    // dw/dx
  zxInterpOp_->apply_to_field( *zxfield, *tmp2);
  
  S13 <<= 0.5 * (*tmp1 + *tmp2);                  // S_13 = S_31 =0.5 (du/dz + dw/dx)
  strTsrMag <<= strTsrMag + 2.0 * S13 * S13;   //    |S|^2 = 2.0 * S_ij * S_ij (we take account for S_ij and Sji at the same time)
  
  //-------------------------
  S23 <<= 0.0;
  dvdzOp_->apply_to_field( v, *yzfield );    // dv/dz
  yzInterpOp_->apply_to_field( *yzfield, *tmp1);
  
  dwdyOp_->apply_to_field( w, *zyfield );    // dw/dy
  zyInterpOp_->apply_to_field( *zyfield, *tmp2);
  
  S23 <<= 0.5*(*tmp1 + *tmp2);                         // S_23 = S_32 = 0.5 *(dv/dz + dw/dy)
  strTsrMag <<= strTsrMag + 2.0 * S23 * S23;   // |S|^2 / 2 = S_ij * S_ij (we take account for S_ij and Sji at the same time)
  strTsrMag <<= sqrt(2.0 * strTsrMag);
}

//------------------------------------------------------
