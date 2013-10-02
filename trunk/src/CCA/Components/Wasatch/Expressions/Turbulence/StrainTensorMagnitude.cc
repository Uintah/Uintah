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

#include <CCA/Components/Wasatch/Expressions/Turbulence/StrainTensorMagnitude.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

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
{
  this->set_gpu_runnable( true );
}

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
  
  S11_ = &fml.field_ref<SS::XSurfXField>(S11Tag_);
  S21_ = &fml.field_ref<SS::XSurfYField>(S21Tag_);
  S31_ = &fml.field_ref<SS::XSurfZField>(S31Tag_);

  S22_ = &fml.field_ref<SS::YSurfYField>(S22Tag_);
  S32_ = &fml.field_ref<SS::YSurfZField>(S32Tag_);

  S33_ = &fml.field_ref<SS::ZSurfZField>(S33Tag_);
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
  SVolField& strTsrMag = this->value();
  strTsrMag <<= 0.0;
  strTsrMag <<= (*xxInterpOp_)(*S11_) * (*xxInterpOp_)(*S11_) // S11*S11
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
                  const Expr::Tag& s33Tag )
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
  return new StrainTensorSquare( S11Tag_, S21Tag_, S31Tag_,
                                 S22Tag_, S32Tag_,
                                 S33Tag_ );
}

//********************************************************************
// WALE MODEL
//********************************************************************

WaleTensorMagnitude::
WaleTensorMagnitude( const Expr::TagList& velTags )
: StrainTensorBase( velTags )
{
  this->set_gpu_runnable( true );
}

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

  // gij = dui/dxj is the velocity gradient tensor
  SpatFldPtr<SVolField> g11 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g12 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g13 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g21 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g22 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g23 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g31 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g32 = SpatialFieldStore::get<SVolField>( waleTsrMag );
  SpatFldPtr<SVolField> g33 = SpatialFieldStore::get<SVolField>( waleTsrMag );

  // dui/dxi fields
  *g11 <<= (*dudxOp_)(*vel1_); // dudx
  *g22 <<= (*dvdyOp_)(*vel2_); // dvdy
  *g33 <<= (*dwdzOp_)(*vel3_); // dwdz

  // cell centered dui/dxj fields
  *g12 <<= (*xyInterpOp_)( (*dudyOp_)(*vel1_) ); // cell centered dudy
  *g21 <<= (*yxInterpOp_)( (*dvdxOp_)(*vel2_) ); // cell centered dvdx
  
  *g13 <<= (*xzInterpOp_)( (*dudzOp_)(*vel1_) ); // cell centered dudz
  *g31 <<= (*zxInterpOp_)( (*dwdxOp_)(*vel3_) ); // cell centered dwdx
  
  *g23 <<= (*yzInterpOp_)( (*dvdzOp_)(*vel2_) ); // cell centered dvdz
  *g32 <<= (*zyInterpOp_)( (*dwdyOp_)(*vel3_) ); // cell centered dwdy

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
  *dilsq<<= 0.0; // gd_kk
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
                  const Expr::TagList& velTags )
: ExpressionBuilder(result),
  velTags_( velTags )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
WaleTensorMagnitude::Builder::build() const
{
  return new WaleTensorMagnitude( velTags_ );
}

//********************************************************************
// VREMAN MODEL
//********************************************************************

VremanTensorMagnitude::
VremanTensorMagnitude( const Expr::TagList& velTags )
: StrainTensorBase( velTags )
{
  this->set_gpu_runnable( true );
}

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
  SVolField& vremanTsrMag = this->value();
  
  // aij corresponds to alpha_ij in the Vreman paper (eq. 6)
  SpatFldPtr<SVolField> a11 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> a12 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> a13 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> a21 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> a22 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> a23 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> a31 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> a32 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> a33 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  
  // dui/dxi fields
  *a11 <<= (*dudxOp_)(*vel1_); // dudx
  *a22 <<= (*dvdyOp_)(*vel2_); // dvdy
  *a33 <<= (*dwdzOp_)(*vel3_); // dwdz
  
  // cell centered duj/dxi fields
  *a21 <<= (*xyInterpOp_)( (*dudyOp_)(*vel1_) ); // cell centered dudy
  *a12 <<= (*yxInterpOp_)( (*dvdxOp_)(*vel2_) ); // cell centered dvdx
  
  *a31 <<= (*xzInterpOp_)( (*dudzOp_)(*vel1_) ); // cell centered dudz
  *a13 <<= (*zxInterpOp_)( (*dwdxOp_)(*vel3_) ); // cell centered dwdx

  *a32 <<= (*yzInterpOp_)( (*dvdzOp_)(*vel2_) ); // cell centered dvdz
  *a23 <<= (*zyInterpOp_)( (*dwdyOp_)(*vel3_) ); // cell centered dwdy
  
  // bij corresponds to beta_ij in the Vreman paper (eq. 7)
  SpatFldPtr<SVolField> b11 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> b12 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> b13 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> b21 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> b22 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> b23 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> b31 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> b32 = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  SpatFldPtr<SVolField> b33 = SpatialFieldStore::get<SVolField>( vremanTsrMag );  
  
  *b11 <<= *a11 * *a11 + *a21 * *a21 + *a31 * *a31;
  *b12 <<= *a11 * *a12 + *a21 * *a22 + *a31 * *a32;
  *b13 <<= *a11 * *a13 + *a21 * *a23 + *a31 * *a33;
  
  *b21 <<= *a12 * *a11 + *a22 * *a21 + *a32 * *a31;
  *b22 <<= *a12 * *a12 + *a22 * *a22 + *a32 * *a32;
  *b23 <<= *a12 * *a13 + *a22 * *a23 + *a32 * *a33;
  
  *b31 <<= *a13 * *a11 + *a23 * *a21 + *a33 * *a31;
  *b32 <<= *a13 * *a12 + *a23 * *a22 + *a33 * *a32;
  *b33 <<= *a13 * *a13 + *a23 * *a23 + *a33 * *a33;

  // aa = aij * aij - corresponds to alpha_ij * alpha_ij (Vreman paper, eq. 5, denominator)
  SpatFldPtr<SVolField> aa = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  *aa <<=  *a11 * *a11 + *a12 * *a12 + *a13 * *a13
         + *a21 * *a21 + *a22 * *a22 + *a32 * *a32
         + *a31 * *a31 + *a32 * *a32 + *a33 * *a33;

  // bbeta corresponds to B_\beta in the Vreman paper (eq. 8)
  SpatFldPtr<SVolField> bbeta = SpatialFieldStore::get<SVolField>( vremanTsrMag );
  *bbeta <<= *b11 * *b22 - *b12 * *b12
           + *b11 * *b33 - *b13 * *b13
           + *b22 * *b33 - *b23 * *b23;

  // TSAAD: The reason that we are using conditionals over here has to do with
  // embedded boundaries. When embedded boundaries are present, and when taking
  // the velocity field from Arches, bbeta/abeta are negative.
  // This can be easily avoided by multiplying abeta and bbeta by the volume
  // fraction. It seems, however, that some cells still exhibit undesirable behavior.
  // It seems that the most conveninent and compact way of dealing with this is
  // to check if either abeta or beta are negative * less than numeric_limits::epsilon *.
  const double eps = std::numeric_limits<double>::epsilon();
  vremanTsrMag <<= cond ( *aa <= eps || *bbeta <= eps, 0.0    )
                        ( sqrt(*bbeta / *aa) ); // Vreman eq. 5
}

//--------------------------------------------------------------------

VremanTensorMagnitude::
Builder::Builder( const Expr::Tag& result,
                  const Expr::TagList& velTags )
: ExpressionBuilder(result),
  velTags_( velTags )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
VremanTensorMagnitude::Builder::build() const
{
  return new VremanTensorMagnitude( velTags_ );
}

//--------------------------------------------------------------------
