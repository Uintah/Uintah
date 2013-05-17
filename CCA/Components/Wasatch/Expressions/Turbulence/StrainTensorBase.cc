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
StrainTensorBase( const Expr::TagList& velTags )
: Expr::Expression<SVolField>(),
  velTags_( velTags ),
  doX_  ( velTags[0] != Expr::Tag() ),
  doY_  ( velTags[1] != Expr::Tag() ),
  doZ_  ( velTags[2] != Expr::Tag() )
{
  if (!(doX_ && doY_ && doZ_)) {
    std::ostringstream msg;
    msg << "WARNING: You cannot use the Dynamic Smagorinsky Model in one or two dimensions. Please revise your input file and make sure that you specify all three velocity/momentum components." << std::endl;
    std::cout << msg.str() << std::endl;
    throw std::runtime_error(msg.str());
  }
}

//--------------------------------------------------------------------

StrainTensorBase::
~StrainTensorBase()
{}

//--------------------------------------------------------------------

void
StrainTensorBase::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( velTags_ );
}

//--------------------------------------------------------------------

void
StrainTensorBase::
bind_fields( const Expr::FieldManagerList& fml )
{
  vel1_ = &fml.field_ref<XVolField>( velTags_[0] );
  vel2_ = &fml.field_ref<YVolField>( velTags_[1] );
  vel3_ = &fml.field_ref<ZVolField>( velTags_[2] );
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
                  const Expr::TagList& velTags )
: ExpressionBuilder(result),
  velTags_(velTags)
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
StrainTensorBase::Builder::build() const
{
  return new StrainTensorBase( velTags_ );
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
