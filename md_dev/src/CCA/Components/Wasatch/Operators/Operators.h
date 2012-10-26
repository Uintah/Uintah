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

#ifndef Wasatch_Operators_h
#define Wasatch_Operators_h

/**
 *  \file Operators.h
 */

namespace SpatialOps{ class OperatorDatabase; }  // forward declaration
namespace Uintah{ class Patch; }

namespace Wasatch{

  /**
   *  \ingroup WasatchOperators
   *
   *  \brief constructs operators for use on the given patch and
   *         stores them in the supplied OperatorDatabase.
   *
   *  \param patch - the Uintah::Patch that the operators are built for.
   *  \param opDB  - the OperatorDatabase to store the operators in.
   *
   *  All supported operators will be constructed and stored in the
   *  supplied OperatorDatabase.  Note that these operators are
   *  associated with the supplied patch only.  Different patches may
   *  have different operators, particularly in the case of AMR.
   */
  void build_operators( const Uintah::Patch& patch,
                        SpatialOps::OperatorDatabase& opDB );


} // namespace Wasatch

#endif // Wasatch_Operators_h
