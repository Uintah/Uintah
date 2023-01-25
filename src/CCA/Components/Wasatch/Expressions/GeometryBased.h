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

#ifndef GeometryBased_h
#define GeometryBased_h

// Framework Includes
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>

// Wasatch Includes
#include <CCA/Components/Wasatch/TagNames.h>

// ExprLib Includes
#include <expression/Expression.h>

// SpatialOps Includes
#include <spatialops/OperatorDatabase.h>

/**
 *  \class   GeometryBased
 *  \author  Tony Saad
 *  \date    March, 2014
 *  \ingroup Expressions
 *
 *  \brief The GeometryBased expression provides a conveninent mechanism to set values on a field
 using geometry primities. This could be useful for setting complicated spatially varying conditions.
 */
template< typename FieldT >
class GeometryBased : public Expr::Expression<FieldT>
{
public:
  
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder(const Expr::Tag& result,
            std::multimap <Uintah::GeometryPieceP, double > geomObjects,
            const double outsideValue);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    std::multimap <Uintah::GeometryPieceP, double > geomObjects_;
    const double outsideValue_;
  };
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
private:
  typedef std::multimap <Uintah::GeometryPieceP, double > GeomValueMapT;  // temporary typedef map that stores boundary
  GeometryBased( GeomValueMapT geomObjects,const double outsideValue);
  GeomValueMapT geomObjects_;
  const double outsideValue_;
  DECLARE_FIELDS(SVolField, x_, y_, z_)

  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, FieldT >::type InpterpSrcT2DestT;
  const InpterpSrcT2DestT* interpSrcT2DestTOp_;

  void compute_volume_fraction(SVolField& volFrac);
};

//--------------------------------------------------------------------

template< typename FieldT >
GeometryBased<FieldT>::
GeometryBased( GeomValueMapT geomObjects,
               const double outsideValue )
: Expr::Expression<FieldT>(),
  geomObjects_(geomObjects),
  outsideValue_(outsideValue)
{
  const WasatchCore::TagNames& tags = WasatchCore::TagNames::self();
   x_ = this->template create_field_request<SVolField>(tags.xsvolcoord);
   y_ = this->template create_field_request<SVolField>(tags.ysvolcoord);
   z_ = this->template create_field_request<SVolField>(tags.zsvolcoord);
}

//--------------------------------------------------------------------

template<typename FieldT >
void
GeometryBased<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpSrcT2DestTOp_ = opDB.retrieve_operator<InpterpSrcT2DestT>();
}

//--------------------------------------------------------------------
template<typename FieldT>
void
GeometryBased<FieldT>::
compute_volume_fraction(SVolField& volFrac)
{
  using namespace SpatialOps;
  volFrac <<= outsideValue_;
  SVolField::iterator resultIter = volFrac.begin();
  
  GeomValueMapT::iterator geomIter;
  
  int i = 0;
  int ix,iy,iz;
  double xp,yp,zp;
  bool isInside = false;
  SpatialOps::IntVec localCellIJK;
  
  const SVolField& x = x_->field_ref();
  const SVolField& y = y_->field_ref();
  const SVolField& z = z_->field_ref();
  
  while ( resultIter != volFrac.end() ) {
    i = resultIter - volFrac.begin();
    localCellIJK  = volFrac.window_with_ghost().ijk_index_from_local(i);
    ix = x.window_with_ghost().flat_index(localCellIJK);
    xp = x[ix];
    iy = y.window_with_ghost().flat_index(localCellIJK);
    yp = y[iy];
    iz = z.window_with_ghost().flat_index(localCellIJK);
    zp = z[iz];
    Uintah::Point p(xp,yp,zp);
    
    // loop over all geometry objects
    geomIter = geomObjects_.begin();
    
    while (geomIter != geomObjects_.end()) {
      isInside = geomIter->first->inside(p);
      if (isInside) *resultIter = geomIter->second;
      ++geomIter;
    }
    
    isInside = false;
    ++resultIter;
  }
}

//--------------------------------------------------------------------

template<>
void
GeometryBased<SVolField>::
evaluate()
{
  using namespace SpatialOps;
  SVolField& result = this->value();
  compute_volume_fraction(result);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
GeometryBased<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= outsideValue_;

  SpatFldPtr<SVolField> tmp = SpatialFieldStore::get<SVolField>( result );
  SVolField& svolTmp = *tmp;
  compute_volume_fraction(svolTmp);
  
  // interpolate to get area fractions
  result <<= (*interpSrcT2DestTOp_)(svolTmp);
  // IMPORTANT: The interpolant above could results in values that are an average between the inside and outside value of the geometry (e.g. 0.5)
  // TO fix that, uncomment the following.
//  result <<= cond( result>0.0, 1.0 )
//                 ( 0.0 );
}

//--------------------------------------------------------------------

template< typename FieldT >
GeometryBased<FieldT>::Builder::
Builder( const Expr::Tag& result,
         GeomValueMapT geomObjects,
         const double outsideValue )
: ExpressionBuilder(result),
  geomObjects_(geomObjects),
  outsideValue_(outsideValue)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
GeometryBased<FieldT>::Builder::
build() const
{
  return new GeometryBased<FieldT>( geomObjects_, outsideValue_ );
}


#endif // GeometryBased_h
