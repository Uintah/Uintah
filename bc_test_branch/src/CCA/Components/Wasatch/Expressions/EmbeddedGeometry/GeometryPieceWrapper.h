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

#ifndef GeometryPieceWrapper_h
#define GeometryPieceWrapper_h

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <expression/Expression.h>
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
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  GeometryPieceWrapper( std::vector<Uintah::GeometryPieceP> geomObjects,
                       const bool inverted);
  std::vector<Uintah::GeometryPieceP> geomObjects_;
  const bool inverted_;
  const Expr::Tag xTag_, yTag_, zTag_;  
  const SVolField* x_;
  const SVolField* y_;
  const SVolField* z_;
};


GeometryPieceWrapper::
GeometryPieceWrapper( std::vector<Uintah::GeometryPieceP> geomObjects,
                     const bool inverted)
: Expr::Expression<SVolField>(),
geomObjects_(geomObjects),
inverted_(inverted),
xTag_(Expr::Tag("XSVOL",Expr::STATE_NONE)),
yTag_(Expr::Tag("YSVOL",Expr::STATE_NONE)),
zTag_(Expr::Tag("ZSVOL",Expr::STATE_NONE))
{}

//--------------------------------------------------------------------

void
GeometryPieceWrapper::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( zTag_ );
}

//--------------------------------------------------------------------

void
GeometryPieceWrapper::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& fm = fml.field_manager<SVolField>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
  z_ = &fm.field_ref( zTag_ );
}

//--------------------------------------------------------------------

void
GeometryPieceWrapper::
evaluate()
{
  using namespace SpatialOps;
  SVolField& result = this->value();
  result <<= 1.0;
  SVolField::interior_iterator resultIter = result.interior_begin();
  std::vector<Uintah::GeometryPieceP>::iterator geomIter = geomObjects_.begin();
  int i = 0;
  int ix,iy,iz;
  double x,y,z;
  bool isInside = false;
  SpatialOps::structured::IntVec localCellIJK;

  while ( resultIter != result.interior_end() ) {
    i = resultIter - result.interior_begin();
    localCellIJK  = result.window_without_ghost().ijk_index_from_local(i);
    ix = x_->window_without_ghost().flat_index(localCellIJK);
    x = (*x_)[ix];
    iy = y_->window_without_ghost().flat_index(localCellIJK);
    y = (*y_)[iy];
    iz = z_->window_without_ghost().flat_index(localCellIJK);
    z = (*z_)[iz];
    Uintah::Point p(x,y,z);

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
