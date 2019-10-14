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

#include "StrainTensorBase.h"

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
StrainTensorBase<ResultT, Vel1T,  Vel2T, Vel3T>::
StrainTensorBase( const Expr::TagList& velTags )
: Expr::Expression<SVolField>(),
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
   u_ = this->template create_field_request<Vel1T>(velTags[0]);
   v_ = this->template create_field_request<Vel2T>(velTags[1]);
   w_ = this->template create_field_request<Vel3T>(velTags[2]);
}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
StrainTensorBase<ResultT, Vel1T,  Vel2T, Vel3T>::
~StrainTensorBase()
{}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
void
StrainTensorBase<ResultT, Vel1T,  Vel2T, Vel3T>::
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

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
void
StrainTensorBase<ResultT, Vel1T,  Vel2T, Vel3T>::
evaluate()
{}

//--------------------------------------------------------------------

template<typename ResultT, typename Vel1T, typename Vel2T, typename Vel3T>
void
StrainTensorBase<ResultT, Vel1T,  Vel2T, Vel3T>::
calculate_strain_tensor_components(ResultT& strTsrMag,
                                   const Vel1T& u,
                                   const Vel2T& v,
                                   const Vel3T& w,
                                   ResultT& S11,
                                   ResultT& S12,
                                   ResultT& S13,
                                   ResultT& S22,
                                   ResultT& S23,
                                   ResultT& S33)
{
  using namespace SpatialOps;

  S11 <<= (*dudxOp_)(u);               // S_11 = 0.5 * (du/dx + du/dx) = du/dx
  S22 <<= (*dvdyOp_)(v);               // S_22 = 0.5 * (dv/dy + dv/dy) = dv/dy
  S33 <<= (*dwdzOp_)(w);               // S_33 = 0.5 * (dw/dz + dw/dz) = dwdz

  S12 <<= 0.5 * ( (*xyInterpOp_)( (*dudyOp_)(u) ) + (*yxInterpOp_)( (*dvdxOp_)(v) )); // S_12 = S_21 = 0.5 * (du/dy + dv/dx)
  S13 <<= 0.5 * ( (*xzInterpOp_)( (*dudzOp_)(u) ) + (*zxInterpOp_)( (*dwdxOp_)(w) )); // S_13 = S_31 =0.5 (du/dz + dw/dx)
  S23 <<= 0.5 * ( (*yzInterpOp_)( (*dvdzOp_)(v) ) + (*zyInterpOp_)( (*dwdyOp_)(w) )); // S_23 = S_32 = 0.5 *(dv/dz + dw/dy)

  //-------------------------
  strTsrMag <<= sqrt(2.0 * (S11 * S11 + S22 * S22 + S33 * S33
                            + 2.0 * S12 * S12
                            + 2.0 * S13 * S13
                            + 2.0 * S23 * S23 ) );
}

//------------------------------------------------------
//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>

template class StrainTensorBase< SpatialOps::SVolField,
                                 SpatialOps::XVolField,
                                 SpatialOps::YVolField,
                                 SpatialOps::ZVolField >;

template class StrainTensorBase< SpatialOps::SVolField,
                                 SpatialOps::SVolField,
                                 SpatialOps::SVolField,
                                 SpatialOps::SVolField >;

