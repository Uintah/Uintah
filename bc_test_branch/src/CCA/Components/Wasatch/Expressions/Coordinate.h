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

#ifndef Coordinate_Expr_h
#define Coordinate_Expr_h

#include <CCA/Components/Wasatch/CoordHelper.h>

#include <expression/PlaceHolderExpr.h>

namespace Wasatch{

  // note that this is in the Wasatch namespace since it is tied to
  // Uintah through the CoordHelper class.

  /**
   *  \class 	Coordinate
   *  \author 	James C. Sutherland
   *  \ingroup	Expressions
   *
   *  \brief shell expression to cause coordinates to be set.  Useful
   *         for initialization, MMS, etc.
   *
   *  \tparam FieldT the type of field to set for this coordinate
   */
  template< typename FieldT >
  class Coordinate
    : public Expr::PlaceHolder<FieldT>
  {
    Coordinate( CoordHelper& coordHelper,
                const Direction dir )
      : Expr::PlaceHolder<FieldT>()
    {
      coordHelper.requires_coordinate<FieldT>( dir );
    }

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  \brief Build a Coordinate expression.
       *  \param result the coordinate calculated by this expression
       *  \param coordHelper - the CoordHelper object.
       *  \param dir - the Direction to set for this coordinate (e.g. x, y, z)
       */
      Builder( const Expr::Tag& result, CoordHelper& coordHelper, const Direction dir )
        : ExpressionBuilder(result),
          coordHelper_( coordHelper ),
          dir_( dir )
      {}
      Expr::ExpressionBase* build() const{ return new Coordinate<FieldT>(coordHelper_,dir_); }
    private:
      CoordHelper& coordHelper_;
      const Direction dir_;
    };
    ~Coordinate(){}
  };

} // namespace Wasatch

#endif // Coordinate_Expr_h
