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

#ifndef Wasatch_GraphHelperTools_h
#define Wasatch_GraphHelperTools_h

#include <list>
#include <map>
#include <set>

#include <expression/ExpressionID.h>

/**
 *  \file GraphHelperTools.h
 *
 *  \brief Some helpful typedefs, structs, and enums for dealing with
 *  the graphs constructed in Wasatch.
 */

namespace Expr{
  class ExpressionFactory;
}

namespace Wasatch{

  class TransportEquation;  // forward declaration

  /** \addtogroup WasatchGraph
   *  @{
   */

  /**
   *  \enum Category
   *  \brief defines the broad categories for various kinds of tasks.
   *
   *  Tasks associated with a particular category are registered in
   *  the associated Wasatch method that Uintah calls to register
   *  tasks.  They are generally combined into one or more Expression
   *  trees that are wrapped using the Wasatch::TaskInterface.
   */
  enum Category{
    INITIALIZATION,	///< Tasks associated with simulation initialization
    TIMESTEP_SELECTION, ///< Tasks associated with choosing the size of the timestep
    ADVANCE_SOLUTION    ///< Tasks associated with advancing the solution forward in time
  };

  /**
   *  \brief a list of transport equations to be solved.
   */
  typedef std::list<TransportEquation*> TransEqns;

  /**
   *  \brief a set of ExpressionID generally to be used to store the
   *  "root" nodes of an ExpressionTree.
   */
  typedef std::set< Expr::ExpressionID > IDSet;

  /**
   *  \struct GraphHelper
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief Contains information helpful in constructing graphs from ExprLib
   */
  struct GraphHelper
  {
    Expr::ExpressionFactory* const exprFactory;  ///< The factory used to build expressions
    IDSet rootIDs;                               ///< The root IDs used to create the graph
    GraphHelper( Expr::ExpressionFactory* ef );
  };

  /**
   *  \brief Defines a map that provides GraphHelper objects given the
   *  \ref Wasatch::Category "Category" that they belong to.
   */
  typedef std::map< Category, GraphHelper* > GraphCategories;

  /** @} */

} // namespace Wasatch


#endif // Wasatch_GraphHelperTools_h
