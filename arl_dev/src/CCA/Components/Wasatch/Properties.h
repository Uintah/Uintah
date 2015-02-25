/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#ifndef Wasatch_Properties_h
#define Wasatch_Properties_h

#include <Core/ProblemSpec/ProblemSpecP.h>

#include "GraphHelperTools.h"

/**
 *  \file Properties.h
 *
 *  \brief Parser handling for property specification.
 */

namespace Wasatch{


  /**
   *  \ingroup WasatchParser
   *  \brief handles parsing for the property evaluators.
   *  \param [in] params The parser block.  This block will be searched for
   *         one containing the \verbatim <PropertyEvaluator> \endverbatim tag.
   *  \param [in] gc
   *  \param [inout] lockedFields the fields that should be persistent (not
   *         scratch/temporary).  These will be managed by Uintah.
   */
  void setup_property_evaluation( Uintah::ProblemSpecP& params,
                                  GraphCategories& gc,
                                  std::set<std::string>& lockedFields );

} // namespace Wasatch

#endif // Wasatch_Properties_h
