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

#include "StrainTensorMagnitude.h"

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/StringNames.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//--------------------------------------------------------------------

Expr::Tag straintensormagnitude_tag() {
  const Wasatch::StringNames& sName = Wasatch::StringNames::self();
  return Expr::Tag( sName.straintensormag, Expr::STATE_NONE );
}

Expr::Tag wale_tensormagnitude_tag() {
  const Wasatch::StringNames& sName = Wasatch::StringNames::self();
  return Expr::Tag( sName.waletensormag, Expr::STATE_NONE );
}

Expr::Tag vreman_tensormagnitude_tag() {
  const Wasatch::StringNames& sName = Wasatch::StringNames::self();
  return Expr::Tag( sName.vremantensormag, Expr::STATE_NONE );
}

//********************************************************************
// STRAIN TENSOR SQUARE (used for Smagorinsky, Vreman, and WALE models)
//********************************************************************

StrainTensorSquare::
StrainTensorSquare( const Expr::Tag& s11Tag,
                    const Expr::Tag& s21Tag,
                    const Expr::Tag& s31Tag,
                    const Expr::Tag& s22Tag,
                    const Expr::Tag& s32Tag,
                    const Expr::Tag& s33Tag )
  : Expr::Expression<SVolField>(),
    S11Tag_(s11Tag),
    S21Tag_(s21Tag),
    S31Tag_(s31Tag),
    S22Tag_(s22Tag),
    S32Tag_(s32Tag),
    S33Tag_(s33Tag)
{}

//--------------------------------------------------------------------

StrainTensorSquare::
~StrainTensorSquare()
{}

//--------------------------------------------------------------------

void
StrainTensorSquare::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( S11Tag_   );
  exprDeps.requires_expression( S21Tag_   );
  exprDeps.requires_expression( S31Tag_   );

  exprDeps.requires_expression( S22Tag_   );
  exprDeps.requires_expression( S32Tag_   );

  exprDeps.requires_expression( S33Tag_   );
}

//--------------------------------------------------------------------

void
StrainTensorSquare::
bind_fields( const Expr::FieldManagerList& fml )
{
  namespace SS = SpatialOps::structured;
  
  S11_ = &fml.field_manager<SS::XSurfXField>().field_ref(S11Tag_);
  S21_ = &fml.field_manager<SS::XSurfYField>().field_ref(S21Tag_);
  S31_ = &fml.field_manager<SS::XSurfZField>().field_ref(S31Tag_);
  
  S22_ = &fml.field_manager<SS::YSurfYField>().field_ref(S22Tag_);
  S32_ = &fml.field_manager<SS::YSurfZField>().field_ref(S32Tag_);

  S33_ = &fml.field_manager<SS::ZSurfZField>().field_ref(S33Tag_);
}

//--------------------------------------------------------------------

void
StrainTensorSquare::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  xxInterpOp_ = opDB.retrieve_operator<XXInterpT>();
  yyInterpOp_ = opDB.retrieve_operator<YYInterpT>();
  zzInterpOp_ = opDB.retrieve_operator<ZZInterpT>();
  xyInterpOp_ = opDB.retrieve_operator<XYInterpT>();
  xzInterpOp_ = opDB.retrieve_operator<XZInterpT>();
  yzInterpOp_ = opDB.retrieve_operator<YZInterpT>();  
}

//--------------------------------------------------------------------

void
StrainTensorSquare::
evaluate()
{
  using namespace SpatialOps;
  SVolField& StrTsrMag = this->value();
  StrTsrMag <<= 0.0;
  StrTsrMag <<=   (*xxInterpOp_)(*S11_) * (*xxInterpOp_)(*S11_) // S11*S11
              + (*yyInterpOp_)(*S22_) * (*yyInterpOp_)(*S22_) // S22*S22
              + (*zzInterpOp_)(*S33_) * (*zzInterpOp_)(*S33_) // S33*S33
              + 2.0 * (*xyInterpOp_)(*S21_) * (*xyInterpOp_)(*S21_) // S12*S12 + S21*S21 = 2.0*S21*S21
              + 2.0 * (*xzInterpOp_)(*S31_) * (*xzInterpOp_)(*S31_) // S13*S13 + S31*S31 = 2.0*S31*S31
              + 2.0 * (*yzInterpOp_)(*S32_) * (*yzInterpOp_)(*S32_);// S23*S23 + S32*S32 = 2.0*S32*S32 */
}

//--------------------------------------------------------------------

StrainTensorSquare::
Builder::Builder( const Expr::Tag& result,
                 const Expr::Tag& s11Tag,
                 const Expr::Tag& s21Tag,
                 const Expr::Tag& s31Tag,
                 const Expr::Tag& s22Tag,
                 const Expr::Tag& s32Tag,
                 const Expr::Tag& s33Tag)
  : ExpressionBuilder(result),
    S11Tag_(s11Tag),
    S21Tag_(s21Tag),
    S31Tag_(s31Tag),
    S22Tag_(s22Tag),
    S32Tag_(s32Tag),
    S33Tag_(s33Tag)
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
StrainTensorSquare::Builder::build() const
{
  return new StrainTensorSquare(S11Tag_, S21Tag_, S31Tag_,
                                S22Tag_, S32Tag_,
                                S33Tag_);
}

//********************************************************************
// WALE MODEL
//********************************************************************

WaleTensorMagnitude::
WaleTensorMagnitude( const Expr::Tag& vel1tag,
                      const Expr::Tag& vel2tag,
                      const Expr::Tag& vel3tag )
: StrainTensorBase( vel1tag, vel2tag, vel3tag )
{}

//--------------------------------------------------------------------

WaleTensorMagnitude::
~WaleTensorMagnitude()
{}

//--------------------------------------------------------------------

void
WaleTensorMagnitude::
evaluate()
{
  using namespace SpatialOps;
  SVolField& waleTsrMag = this->value();
  waleTsrMag <<= 0.0;

  SpatFldPtr<structured::XSurfYField> xyfield = SpatialFieldStore::get<structured::XSurfYField>( waleTsrMag );
  SpatFldPtr<structured::YSurfXField> yxfield = SpatialFieldStore::get<structured::YSurfXField>( waleTsrMag );
  *xyfield <<= 0.0;
  *yxfield <<= 0.0;

  SpatFldPtr<structured::XSurfZField> xzfield = SpatialFieldStore::get<structured::XSurfZField>( waleTsrMag );
  SpatFldPtr<structured::ZSurfXField> zxfield = SpatialFieldStore::get<structured::ZSurfXField>( waleTsrMag );
  *xzfield <<= 0.0;
  *zxfield <<= 0.0;

  SpatFldPtr<structured::YSurfZField> yzfield = SpatialFieldStore::get<structured::YSurfZField>( waleTsrMag );
  SpatFldPtr<structured::ZSurfYField> zyfield = SpatialFieldStore::get<structured::ZSurfYField>( waleTsrMag );
  *zyfield <<= 0.0;
  *yzfield <<= 0.0;

  SpatFldPtr<SVolField> g11 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g12 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g13 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g21 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g22 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g23 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g31 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g32 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g33 = SpatialFieldStore::get<SVolField>( waleTsrMag );


  //  if (!doX_) {
  *g11 <<= 0.0;
  *g12 <<= 0.0;
  *g13 <<= 0.0;
  *g21 <<= 0.0;
  *g31 <<= 0.0;
  //  }

  //  if (!doY_) {
  *g21 <<= 0.0;
  *g22 <<= 0.0;
  *g23 <<= 0.0;
  *g12 <<= 0.0;
  *g32 <<= 0.0;
  //  }

  //  if (!doZ_) {
  *g31 <<= 0.0;
  *g32 <<= 0.0;
  *g33 <<= 0.0;
  *g23 <<= 0.0;
  *g13 <<= 0.0;
  //  }


  // du fields
  if (doX_) dudxOp_->apply_to_field( *vel1_, *g11 );
  if (doY_) dvdyOp_->apply_to_field( *vel2_, *g22 );
  if (doZ_) dwdzOp_->apply_to_field( *vel3_, *g33 );

  if (doX_ && doY_) {
    dudyOp_->apply_to_field( *vel1_, *xyfield );   // du/dy
    xyInterpOp_->apply_to_field( *xyfield, *g12);  // interpolate to scalar cells

    dvdxOp_->apply_to_field( *vel2_, *yxfield );   // dv/dx
    yxInterpOp_->apply_to_field( *yxfield, *g21);  // interpolate to scalar cells
  }

  if (doX_ && doZ_) {
    dudzOp_->apply_to_field( *vel1_, *xzfield );   // du/dz
    xzInterpOp_->apply_to_field( *xzfield, *g13);

    dwdxOp_->apply_to_field( *vel3_, *zxfield );   // dw/dx
    zxInterpOp_->apply_to_field( *zxfield, *g31);
  }

  if (doY_ && doZ_) {
    dvdzOp_->apply_to_field( *vel2_, *yzfield );   // dv/dz
    yzInterpOp_->apply_to_field( *yzfield, *g23);

    dwdyOp_->apply_to_field( *vel3_, *zyfield );   // dw/dy
    zyInterpOp_->apply_to_field( *zyfield, *g32);
  }

  // NOTE: the gd_ij tensor corresponds to the \bar(g^2)_ij tensor in the 
  // Nicoud and Ducros original paper.
  SpatFldPtr<SVolField> gd11 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> gd12 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> gd13 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> gd21 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> gd22 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> gd23 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> gd31 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> gd32 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> gd33 = SpatialFieldStore::get<SVolField>( waleTsrMag );

  //  if (!doX_) {
  *gd11 <<= 0.0;
  *gd12 <<= 0.0;
  *gd13 <<= 0.0;
  *gd21 <<= 0.0;
  *gd31 <<= 0.0;
  //  }

  //  if (!doY_) {
  *gd21 <<= 0.0;
  *gd22 <<= 0.0;
  *gd23 <<= 0.0;
  *gd12 <<= 0.0;
  *gd32 <<= 0.0;
  //  }

  //  if (!doZ_) {
  *gd31 <<= 0.0;
  *gd32 <<= 0.0;
  *gd33 <<= 0.0;
  *gd23 <<= 0.0;
  *gd13 <<= 0.0;
  //  }

  *gd11 <<= *g11 * *g11 + *g12 * *g21 + *g13 * *g31;
  *gd12 <<= *g11 * *g12 + *g12 * *g22 + *g13 * *g32;
  *gd13 <<= *g11 * *g13 + *g12 * *g23 + *g13 * *g33;

  *gd21 <<= *g21 * *g11 + *g22 * *g21 + *g23 * *g31;
  *gd22 <<= *g21 * *g12 + *g22 * *g22 + *g23 * *g32;
  *gd23 <<= *g21 * *g13 + *g22 * *g23 + *g23 * *g33;

  *gd31 <<= *g31 * *g11 + *g32 * *g21 + *g33 * *g31;
  *gd32 <<= *g31 * *g12 + *g32 * *g22 + *g33 * *g32;
  *gd33 <<= *g31 * *g13 + *g32 * *g23 + *g33 * *g33;

  SpatFldPtr<SVolField> dilsq = SpatialFieldStore::get<SVolField>( waleTsrMag );
  *dilsq<<=0.0; // gd_kk
  *dilsq<<= (1.0/3.0)*(*gd11 + *gd22 + *gd33);

  waleTsrMag <<=  (*gd11 - *dilsq) * (*gd11 - *dilsq)       // Sd_11 * Sd_11
                 + 0.5 * (*gd12 + *gd21) * (*gd12 + *gd21)   // Sd_12*Sd_12 + Sd_21*Sd_21 = 2.0 Sd_12*Sd_12
                 + 0.5 * (*gd13 + *gd31) * (*gd13 + *gd31)   // Sd_13*Sd_13 + Sd_31*Sd_31 = 2.0 Sd_13*Sd_13
                 + (*gd22 - *dilsq) * (*gd22 - *dilsq)       // Sd_22 * Sd_22
                 + 0.5 * (*gd23 + *gd32) * (*gd23 + *gd32)   // Sd_23*Sd_23 + Sd_32*Sd_32 = 2.0 Sd_23*Sd_23
                 + (*gd33 - *dilsq) * (*gd33 - *dilsq);      // Sd_33 * Sd_33
}

//--------------------------------------------------------------------

WaleTensorMagnitude::
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
WaleTensorMagnitude::Builder::build() const
{
  return new WaleTensorMagnitude( v1t_, v2t_, v3t_ );
}

//********************************************************************
// VREMAN MODEL
//********************************************************************

VremanTensorMagnitude::
VremanTensorMagnitude( const Expr::Tag& vel1tag,
                            const Expr::Tag& vel2tag,
                            const Expr::Tag& vel3tag )
: StrainTensorBase( vel1tag, vel2tag, vel3tag )
{}

//--------------------------------------------------------------------

VremanTensorMagnitude::
~VremanTensorMagnitude()
{}

//--------------------------------------------------------------------

void
VremanTensorMagnitude::
evaluate()
{
  using namespace SpatialOps;
  SVolField& VremanTsrMag = this->value();
  VremanTsrMag <<= 0.0;
  
  SpatFldPtr<structured::XSurfYField> xyfield = SpatialFieldStore::get<structured::XSurfYField>( VremanTsrMag );
  SpatFldPtr<structured::YSurfXField> yxfield = SpatialFieldStore::get<structured::YSurfXField>( VremanTsrMag );
  *xyfield <<= 0.0;
  *yxfield <<= 0.0;
  
  SpatFldPtr<structured::XSurfZField> xzfield = SpatialFieldStore::get<structured::XSurfZField>( VremanTsrMag );
  SpatFldPtr<structured::ZSurfXField> zxfield = SpatialFieldStore::get<structured::ZSurfXField>( VremanTsrMag );
  *xzfield <<= 0.0;
  *zxfield <<= 0.0;
  
  SpatFldPtr<structured::YSurfZField> yzfield = SpatialFieldStore::get<structured::YSurfZField>( VremanTsrMag );
  SpatFldPtr<structured::ZSurfYField> zyfield = SpatialFieldStore::get<structured::ZSurfYField>( VremanTsrMag );
  *zyfield <<= 0.0;
  *yzfield <<= 0.0;
  
  SpatFldPtr<SVolField> a11 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> a12 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> a13 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> a21 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> a22 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> a23 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> a31 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> a32 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> a33 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  
  
  *a11 <<= 0.0;
  *a12 <<= 0.0;
  *a13 <<= 0.0;
  *a21 <<= 0.0;
  *a31 <<= 0.0;

  *a21 <<= 0.0;
  *a22 <<= 0.0;
  *a23 <<= 0.0;
  *a12 <<= 0.0;
  *a32 <<= 0.0;

  *a31 <<= 0.0;
  *a32 <<= 0.0;
  *a33 <<= 0.0;
  *a23 <<= 0.0;
  *a13 <<= 0.0;
  
  
  // du fields
  if (doX_) dudxOp_->apply_to_field( *vel1_, *a11 ); // dudx
  if (doY_) dvdyOp_->apply_to_field( *vel2_, *a22 ); // dvdy
  if (doZ_) dwdzOp_->apply_to_field( *vel3_, *a33 ); // dwdz
  
  if (doX_ && doY_) {
    dudyOp_->apply_to_field( *vel1_, *xyfield );   // du/dy
    xyInterpOp_->apply_to_field( *xyfield, *a21);  // interpolate to scalar cells
    
    dvdxOp_->apply_to_field( *vel2_, *yxfield );   // dv/dx
    yxInterpOp_->apply_to_field( *yxfield, *a12);  // interpolate to scalar cells
  }
  
  if (doX_ && doZ_) {
    dudzOp_->apply_to_field( *vel1_, *xzfield );   // du/dz
    xzInterpOp_->apply_to_field( *xzfield, *a31);  // interpolate to scalar cells
    
    dwdxOp_->apply_to_field( *vel3_, *zxfield );   // dw/dx
    zxInterpOp_->apply_to_field( *zxfield, *a13);  // interpolate to scalar cells
  }
  
  if (doY_ && doZ_) {
    dvdzOp_->apply_to_field( *vel2_, *yzfield );   // dv/dz
    yzInterpOp_->apply_to_field( *yzfield, *a32);  // interpolate to scalar cells
    
    dwdyOp_->apply_to_field( *vel3_, *zyfield );   // dw/dy
    zyInterpOp_->apply_to_field( *zyfield, *a23);  // interpolate to scalar cells
  }
  
  SpatFldPtr<SVolField> b11 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> b12 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> b13 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> b21 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> b22 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> b23 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> b31 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> b32 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  SpatFldPtr<SVolField> b33 = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  

  *b11 <<= 0.0;
  *b12 <<= 0.0;
  *b13 <<= 0.0;
  *b21 <<= 0.0;
  *b31 <<= 0.0;

  *b21 <<= 0.0;
  *b22 <<= 0.0;
  *b23 <<= 0.0;
  *b12 <<= 0.0;
  *b32 <<= 0.0;

  *b31 <<= 0.0;
  *b32 <<= 0.0;
  *b33 <<= 0.0;
  *b23 <<= 0.0;
  *b13 <<= 0.0;
  
  *b11 <<= *a11 * *a11 + *a21 * *a21 + *a31 * *a31;
  *b12 <<= *a11 * *a12 + *a21 * *a22 + *a31 * *a32;
  *b13 <<= *a11 * *a13 + *a21 * *a23 + *a31 * *a33;
  
  *b21 <<= *a12 * *a11 + *a22 * *a21 + *a32 * *a31;
  *b22 <<= *a12 * *a12 + *a22 * *a22 + *a32 * *a32;
  *b23 <<= *a12 * *a13 + *a22 * *a23 + *a32 * *a33;
  
  *b31 <<= *a13 * *a11 + *a23 * *a21 + *a33 * *a31;
  *b32 <<= *a13 * *a12 + *a23 * *a22 + *a33 * *a32;
  *b33 <<= *a13 * *a13 + *a23 * *a23 + *a33 * *a33;

  SpatFldPtr<SVolField> abeta = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  *abeta <<= 0.0; // abeta = aij * aij
  *abeta <<=  *a11 * *a11 + *a12 * *a12 + *a13 * *a13
            + *a21 * *a21 + *a22 * *a22 + *a32 * *a32
            + *a31 * *a31 + *a32 * *a32 + *a33 * *a33;

  SpatFldPtr<SVolField> bbeta = SpatialFieldStore::get<SVolField>( VremanTsrMag );
  *bbeta<<=0.0;
  *bbeta <<= *b11 * *b22 - *b12 * *b12 + *b11 * *b33 - *b13 * *b13 + *b22 * *b33 - *b23 * *b23;

  // TSAAD: The reason that we are using conditionals over here has to do with
  // embedded boundaries. When embedded boundaries are present, and when taking
  // the velocity field from Arches, bbeta/abeta are negative.
  // This can be easily avoided by multiplying abeta and bbeta by the volume
  // fraction. It seems, however, that some cells still exhibit undesirable behavior.
  // It seems that the most conveninent and compact way of dealing with this is
  // to check if either abeta or beta are negative * less than numeric_limits::epsilone *.
  const double eps = std::numeric_limits<double>::epsilon();
  VremanTsrMag <<= cond ( *abeta <= eps || *bbeta <= eps, 0.0    )
                        ( sqrt(*bbeta / *abeta) );
}

//--------------------------------------------------------------------

VremanTensorMagnitude::
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
VremanTensorMagnitude::Builder::build() const
{
  return new VremanTensorMagnitude( v1t_, v2t_, v3t_ );
}

//--------------------------------------------------------------------
