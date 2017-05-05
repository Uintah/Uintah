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

#ifndef GeometryPieceWrapper_h
#define GeometryPieceWrapper_h

#include <expression/Expression.h>

#include <CCA/Components/Wasatch/TagNames.h>

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
/**
 *  \class   GeometryPieceWrapper
 *  \author  Tony Saad
 *  \date    December, 2012
 *  \ingroup Expressions
 *
 *  \brief The GeometryPieceWrapper expression provides a conveninent mechanism
 to parse and build volume fraction fields from the Uintah GeometryPiece. The
 volume fraction is a cell centered field that is used to represent embedded
 geometries.
 */
class GeometryPieceWrapper : public Expr::Expression<SVolField>
{
public:
  
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder(const Expr::Tag& result,
            std::vector<Uintah::GeometryPieceP> geomObjects,
            const bool inverted);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    std::vector<Uintah::GeometryPieceP> geomObjects_;
    const bool inverted_;
  };
  
  void evaluate();
  
private:
  
  GeometryPieceWrapper( std::vector<Uintah::GeometryPieceP> geomObjects,
                       const bool inverted);
  std::vector<Uintah::GeometryPieceP> geomObjects_;
  const bool inverted_;
  DECLARE_FIELDS(SVolField, x_, y_, z_)
};


GeometryPieceWrapper::
GeometryPieceWrapper( std::vector<Uintah::GeometryPieceP> geomObjects,
                     const bool inverted)
: Expr::Expression<SVolField>(),
geomObjects_(geomObjects),
inverted_(inverted)
{
  const WasatchCore::TagNames& tagNames = WasatchCore::TagNames::self();
   x_ = create_field_request<SVolField>(tagNames.xsvolcoord);
   y_ = create_field_request<SVolField>(tagNames.ysvolcoord);
   z_ = create_field_request<SVolField>(tagNames.zsvolcoord);

}

//--------------------------------------------------------------------

void
GeometryPieceWrapper::
evaluate()
{
  using namespace SpatialOps;
  const SVolField& x = x_->field_ref();
  const SVolField& y = y_->field_ref();
  const SVolField& z = z_->field_ref();
  
  SVolField& result = this->value();
  result <<= 1.0;
  SVolField::iterator resultIter = result.begin();
  std::vector<Uintah::GeometryPieceP>::iterator geomIter = geomObjects_.begin();
  int i = 0;
  int ix,iy,iz;
  double xc,yc,zc;
  bool isInside = false;
  SpatialOps::IntVec localCellIJK;
  
  while ( resultIter != result.end() ) {
    i = resultIter - result.begin();
    localCellIJK  = result.window_with_ghost().ijk_index_from_local(i);
    ix = x.window_with_ghost().flat_index(localCellIJK);
    xc  = x[ix];
    iy = y.window_with_ghost().flat_index(localCellIJK);
    yc  = y[iy];
    iz = z.window_with_ghost().flat_index(localCellIJK);
    zc  = z[iz];
    Uintah::Point p(xc,yc,zc);

    // loop over all geometry objects
    geomIter = geomObjects_.begin();
    while (geomIter != geomObjects_.end()) {
      isInside = isInside || (*geomIter)->inside(p);
      ++geomIter;
    }
    isInside = inverted_ ? !isInside : isInside;
    *resultIter = isInside ? 0.0 : 1.0;
    isInside = false;
    ++resultIter;
  }
}

//--------------------------------------------------------------------

GeometryPieceWrapper::Builder::
Builder( const Expr::Tag& result,
        std::vector<Uintah::GeometryPieceP> geomObjects,
        const bool inverted)
: ExpressionBuilder(result),
geomObjects_(geomObjects),
inverted_(inverted)
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
GeometryPieceWrapper::Builder::
build() const
{
  return new GeometryPieceWrapper( geomObjects_, inverted_ );
}


#endif // GeometryPieceWrapper_h
