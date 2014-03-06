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

#ifndef GeometryBased_h
#define GeometryBased_h

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <expression/Expression.h>
/**
 *  \class   GeometryBased
 *  \author  Tony Saad
 *  \date    March, 2014
 *  \ingroup Expressions
 *
 *  \brief The GeometryBased expression provides a conveninent mechanism to set values on a field
 using geometry primities. This could be useful for setting complicated spatially varying conditions.
 */

class GeometryBased : public Expr::Expression<SVolField>
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
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  typedef std::multimap <Uintah::GeometryPieceP, double > GeomValueMapT;  // temporary typedef map that stores boundary
  GeometryBased( GeomValueMapT geomObjects,const double outsideValue);
  GeomValueMapT geomObjects_;
  const double outsideValue_;
  const Expr::Tag xTag_, yTag_, zTag_;  
  const SVolField* x_;
  const SVolField* y_;
  const SVolField* z_;
};


GeometryBased::
GeometryBased( GeomValueMapT geomObjects,
               const double outsideValue)
: Expr::Expression<SVolField>(),
geomObjects_(geomObjects),
outsideValue_(outsideValue),
xTag_(Expr::Tag("XSVOL",Expr::STATE_NONE)),
yTag_(Expr::Tag("YSVOL",Expr::STATE_NONE)),
zTag_(Expr::Tag("ZSVOL",Expr::STATE_NONE))
{}

//--------------------------------------------------------------------

void
GeometryBased::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( zTag_ );
}

//--------------------------------------------------------------------

void
GeometryBased::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& fm = fml.field_manager<SVolField>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
  z_ = &fm.field_ref( zTag_ );
}

//--------------------------------------------------------------------

void
GeometryBased::
evaluate()
{
  using namespace SpatialOps;
  SVolField& result = this->value();
  result <<= outsideValue_;
  SVolField::iterator resultIter = result.begin();
  
  GeomValueMapT::iterator geomIter;

  int i = 0;
  int ix,iy,iz;
  double x,y,z;
  bool isInside = false;
  SpatialOps::structured::IntVec localCellIJK;
  
  while ( resultIter != result.end() ) {
    i = resultIter - result.begin();
    localCellIJK  = result.window_with_ghost().ijk_index_from_local(i);
    ix = x_->window_with_ghost().flat_index(localCellIJK);
    x = (*x_)[ix];
    iy = y_->window_with_ghost().flat_index(localCellIJK);
    y = (*y_)[iy];
    iz = z_->window_with_ghost().flat_index(localCellIJK);
    z = (*z_)[iz];
    Uintah::Point p(x,y,z);

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

GeometryBased::Builder::
Builder( const Expr::Tag& result,
        GeomValueMapT geomObjects,
        const double outsideValue)
: ExpressionBuilder(result),
geomObjects_(geomObjects),
outsideValue_(outsideValue)
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
GeometryBased::Builder::
build() const
{
  return new GeometryBased( geomObjects_, outsideValue_ );
}


#endif // GeometryBased_h
