/*
 * The MIT License
 *
 * Copyright (c) 2016-2018 The University of Utah
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

/**
 *  \file   PreconditioningParser.h
 *  \date   May, 2016
 *  \author James C. Sutherland
 */

#ifndef WASATCH_TRANSPORT_PRECONDITIONINGPARSER_H_
#define WASATCH_TRANSPORT_PRECONDITIONINGPARSER_H_

#include <Core/ProblemSpec/ProblemSpecP.h>

#include <CCA/Components/Wasatch/GraphHelperTools.h>

#include <expression/ExprFwd.h>

namespace WasatchCore {

  /**
   *  \class  Wasatch::PreconditioningParser
   *  \date   May, 2016
   *  \author James C. Sutherland
   */
  class PreconditioningParser
  {
    enum Method{
      ACOUSTIC_STIFFNESS_REDUCTION,
      PRESSURE_GRADIENT_SCALING,
      NO_PRECONDITIONING
    };

    Uintah::ProblemSpecP& wasatchSpec_;
    Expr::ExpressionFactory* const factory_;
    Method method_;

    void setup_asr( Uintah::ProblemSpecP );
    void setup_pgs( Uintah::ProblemSpecP );
    Expr::TagList vel_tags();

  public:
    /**
     *
     * @param wasatchParams the "Wasatch" scope spec
     */
    PreconditioningParser( Uintah::ProblemSpecP& wasatchSpec,
                           GraphCategories& gc );

    ~PreconditioningParser();
  };

} /* namespace WasatchCore */

#endif /* SRC_CCA_COMPONENTS_WASATCH_TRANSPORT_PRECONDITIONINGPARSER_H_ */
