/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef UINTAH_CCA_COMPONENTS_PARENT_APPLICATIONFACTORY_H
#define UINTAH_CCA_COMPONENTS_PARENT_APPLICATIONFACTORY_H

#include <Core/Grid/MaterialManagerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <string>


namespace Uintah {

  class ProcessorGroup;
  class UintahParallelComponent;

  class ApplicationFactory  {
  
  public:

    // this function has a switch for all known components
    static UintahParallelComponent* create(       ProblemSpecP     & prob_spec
                                          , const ProcessorGroup   * myworld
                                          , const MaterialManagerP   materialManager
                                          , const std::string      & uda
                                          );
  };

} // namespace Uintah


#endif
