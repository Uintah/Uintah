/*
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

#ifndef BoundaryConditionBase_Expr_h
#define BoundaryConditionBase_Expr_h

#include <expression/Expression.h>
/**
 *  \class 	BoundaryConditionBase
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    September, 2012
 *
 *  \brief Provides an expression to set basic Dirichlet and Neumann boundary
 *  conditions. Given a BCValue, we set the ghost value such that
 *  \f$ f[ghost] = \alpha f[interior] + \beta BCValue \f$
 *
 *  \tparam FieldT - the type of field for the RHS.
 */

template< typename FieldT >
class BoundaryConditionBase
: public Expr::Expression<FieldT>
{
  //BoundaryConditionBase(){}
public:
  void set_interior_coef( const double coef) { ci_ = coef;}
  void set_ghost_coef( const double coef) { cg_ = coef;}
  void set_interior_points( const std::vector<int> flatInteriorPoints ){flatInteriorPoints_ = flatInteriorPoints;}
  void set_ghost_points( const std::vector<int> flatGhostPoints ){flatGhostPoints_ = flatGhostPoints;}
protected:
  double ci_, cg_;
  std::vector<int> flatInteriorPoints_;  
  std::vector<int> flatGhostPoints_;
  //~BoundaryConditionBase(){}
};

#endif // BoundaryConditionBase_Expr_h