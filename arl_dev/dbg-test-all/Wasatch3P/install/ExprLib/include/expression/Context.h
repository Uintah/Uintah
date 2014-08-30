/**
 * \file Context.h
 * \author James C. Sutherland
 *
 * Copyright (c) 2011 The University of Utah
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

#ifndef Expr_Context_h
#define Expr_Context_h

#include <string>
#include <ostream>

namespace Expr {

  /**
   *  \enum Context
   *  \brief enumerates field contexts
   */
  enum Context
  {
    STATE_N,		///< field at time level "n"
    STATE_NP1,		///< field at time level "n+1"
    STATE_NONE,		///< "scratch" field
    STATE_DYNAMIC,      ///< used for multistage integrators to allow the state of a field to ...???
    CARRY_FORWARD,	///< used for iterating when you need to carry an initial guess forward from an old field value
    INVALID_CONTEXT     ///< invalid
  };

  /**
   * @param a string representation of a context, e.g., "STATE_NP1"
   * @return the Context enum value, or INVALID_CONTEXT if no string match was found
   *
   * This function is useful when parsing context information from a text file.
   */
  Context
  str2context( std::string );

  /**
   * Obtain a string representation of a context
   * @param the context
   * @return its string representation
   */
  std::string
  context2str( const Context );

  std::ostream&
  operator<<( std::ostream& os, const Context );

}  // namespace Expr

#endif
