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

#ifndef Coordinates_h
#define Coordinates_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <CCA/Components/Wasatch/FieldTypes.h>

namespace WasatchCore{
  
  /**
   *  \class 	 Coordinates
   *  \ingroup   Expressions
   *  \ingroup	 WasatchCore
   *  \author 	 Tony Saad, James C. Sutherland
   *  \date 	 November, 2013
   *
   *  \brief Expression to compute coordinates.
   *
   *  NOTE: this expression BREAKS WITH CONVENTION!  Notably, it has
   *  uintah tenticles that reach into it, and mixes SpatialOps and
   *  Uintah constructs.
   *
   *  This expression does play well with expression graphs, however.
   *  There are only a few places where Uintah reaches in.
   *
   *  Because of the hackery going on here, this expression is placed in
   *  the Wasatch namespace.  This should reinforce the concept that it
   *  is not intended for external use.
   */
  template< typename FieldT>
  class Coordinates
  : public Expr::Expression<FieldT>
  {
    UintahPatchContainer* patchContainer_;
    int idir_; // x = 0, y = 1, z = 2
    Uintah::Vector shift_; // shift spacing by -dx/2 for staggered fields
    Coordinates(const int idir);
    
  public:
    
    class Builder : public Expr::ExpressionBuilder
    {
    private:
      int idir_;
    public:
      Builder( const Expr::Tag& result );
      ~Builder(){}
      Expr::ExpressionBase* build() const;
    };
    
    ~Coordinates();
        
    void bind_operators( const SpatialOps::OperatorDatabase& opDB );
    void evaluate();
    
  };
} // namespace WasatchCore

#endif // Coordinates_h
