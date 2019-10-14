/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
  : Expr::Expression<SVolField>()
{
  this->set_gpu_runnable( true );
   S11_ = create_field_request<S11T>(s11Tag);
   S21_ = create_field_request<S21T>(s21Tag);
   S31_ = create_field_request<S31T>(s31Tag);
  
   S22_ = create_field_request<S22T>(s22Tag);
   S32_ = create_field_request<S32T>(s32Tag);

   S33_ = create_field_request<S33T>(s33Tag);
}

//--------------------------------------------------------------------

StrainTensorSquare::
~StrainTensorSquare()
{}

//--------------------------------------------------------------------

void
StrainTensorSquare::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  this->xxInterpOp_ = opDB.retrieve_operator<XXInterpT>();
  this->yyInterpOp_ = opDB.retrieve_operator<YYInterpT>();
  this->zzInterpOp_ = opDB.retrieve_operator<ZZInterpT>();
  this->xyInterpOp_ = opDB.retrieve_operator<XYInterpT>();
  this->xzInterpOp_ = opDB.retrieve_operator<XZInterpT>();
  this->yzInterpOp_ = opDB.retrieve_operator<YZInterpT>();  
}

//--------------------------------------------------------------------

void
StrainTensorSquare::
evaluate()
{
  using namespace SpatialOps;
  SVolField& strTsrMag = this->value();
  
  const S11T& S11 = S11_->field_ref();
  const S21T& S21 = S21_->field_ref();
  const S31T& S31 = S31_->field_ref();

  const S22T& S22 = S22_->field_ref();
  const S32T& S32 = S32_->field_ref();

  const S33T& S33 = S33_->field_ref();
  
  strTsrMag <<= 0.0;
  strTsrMag <<= (*this->xxInterpOp_)(S11) * (*this->xxInterpOp_)(S11) // S11*S11
              + (*this->yyInterpOp_)(S22) * (*this->yyInterpOp_)(S22) // S22*S22
              + (*this->zzInterpOp_)(S33) * (*this->zzInterpOp_)(S33) // S33*S33
              + 2.0 * (*this->xyInterpOp_)(S21) * (*this->xyInterpOp_)(S21) // S12*S12 + S21*S21 = 2.0*S21*S21
              + 2.0 * (*this->xzInterpOp_)(S31) * (*this->xzInterpOp_)(S31) // S13*S13 + S31*S31 = 2.0*S31*S31
              + 2.0 * (*this->yzInterpOp_)(S32) * (*this->yzInterpOp_)(S32);// S23*S23 + S32*S32 = 2.0*S32*S32 */
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

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
WaleTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::
WaleTensorMagnitude( const Expr::TagList& velTags )
: StrainTensorBase<ResultT, Vel1T, Vel2T, Vel3T>( velTags )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
WaleTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::
~WaleTensorMagnitude()
{}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
void
WaleTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  SVolField& waleTsrMag = this->value();

  const Vel1T& u = this->u_->field_ref();
  const Vel2T& v = this->v_->field_ref();
  const Vel3T& w = this->w_->field_ref();
  
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
  *g11 <<= (*this->dudxOp_)(u); // dudx
  *g22 <<= (*this->dvdyOp_)(v); // dvdy
  *g33 <<= (*this->dwdzOp_)(w); // dwdz

  // cell centered dui/dxj fields
  *g12 <<= (*this->xyInterpOp_)( (*this->dudyOp_)(u) ); // cell centered dudy
  *g21 <<= (*this->yxInterpOp_)( (*this->dvdxOp_)(v) ); // cell centered dvdx
  
  *g13 <<= (*this->xzInterpOp_)( (*this->dudzOp_)(u) ); // cell centered dudz
  *g31 <<= (*this->zxInterpOp_)( (*this->dwdxOp_)(w) ); // cell centered dwdx
  
  *g23 <<= (*this->yzInterpOp_)( (*this->dvdzOp_)(v) ); // cell centered dvdz
  *g32 <<= (*this->zyInterpOp_)( (*this->dwdyOp_)(w) ); // cell centered dwdy

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

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
WaleTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::TagList& velTags )
: ExpressionBuilder(result),
  velTags_( velTags )
{}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
Expr::ExpressionBase*
WaleTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::Builder::build() const
{
  return new WaleTensorMagnitude( velTags_ );
}

//********************************************************************
// VREMAN MODEL
//********************************************************************

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
VremanTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::
VremanTensorMagnitude( const Expr::TagList& velTags )
: StrainTensorBase<ResultT, Vel1T, Vel2T, Vel3T>( velTags )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
VremanTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::
~VremanTensorMagnitude()
{}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
void
VremanTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  SVolField& vremanTsrMag = this->value();
  
  const Vel1T& u = this->u_->field_ref();
  const Vel2T& v = this->v_->field_ref();
  const Vel3T& w = this->w_->field_ref();

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
  *a11 <<= (*this->dudxOp_)(u); // dudx
  *a22 <<= (*this->dvdyOp_)(v); // dvdy
  *a33 <<= (*this->dwdzOp_)(w); // dwdz
  
  // cell centered duj/dxi fields
  *a21 <<= (*this->xyInterpOp_)( (*this->dudyOp_)(u) ); // cell centered dudy
  *a12 <<= (*this->yxInterpOp_)( (*this->dvdxOp_)(v) ); // cell centered dvdx
  
  *a31 <<= (*this->xzInterpOp_)( (*this->dudzOp_)(u) ); // cell centered dudz
  *a13 <<= (*this->zxInterpOp_)( (*this->dwdxOp_)(w) ); // cell centered dwdx

  *a32 <<= (*this->yzInterpOp_)( (*this->dvdzOp_)(v) ); // cell centered dvdz
  *a23 <<= (*this->zyInterpOp_)( (*this->dwdyOp_)(w) ); // cell centered dwdy
  
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

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
VremanTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::TagList& velTags )
: ExpressionBuilder(result),
  velTags_( velTags )
{}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
Expr::ExpressionBase*
VremanTensorMagnitude<ResultT, Vel1T, Vel2T, Vel3T>::Builder::build() const
{
  return new VremanTensorMagnitude( velTags_ );
}

//--------------------------------------------------------------------

template class WaleTensorMagnitude< SpatialOps::SVolField,
                                 SpatialOps::XVolField,
                                 SpatialOps::YVolField,
                                 SpatialOps::ZVolField >;

template class WaleTensorMagnitude< SpatialOps::SVolField,
                                 SpatialOps::SVolField,
                                 SpatialOps::SVolField,
                                 SpatialOps::SVolField >;

template class VremanTensorMagnitude< SpatialOps::SVolField,
                                 SpatialOps::XVolField,
                                 SpatialOps::YVolField,
                                 SpatialOps::ZVolField >;

template class VremanTensorMagnitude< SpatialOps::SVolField,
                                 SpatialOps::SVolField,
                                 SpatialOps::SVolField,
                                 SpatialOps::SVolField >;

