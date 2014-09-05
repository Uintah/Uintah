/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UINTAH_MD_BOUNDCOND_H
#define UINTAH_MD_BOUNDCOND_H

#include <Core/Grid/Variables/NCVariable.h>

namespace Uintah {

using namespace SCIRun;

class SCIRun::Vector;

/**
 *  @class MDDoundCond
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   December, 2012
 *
 *  @brief
 *
 *  @param
 */
class MDBoundaryCondition {

  public:

    /**
     * @brief
     * @param
     * @return
     */
    MDBoundaryCondition();

    /**
     * @brief
     * @param
     * @return
     */
    ~MDBoundaryCondition();

    /**
     * @brief
     * @param
     * @return
     */
    void setBoundaryCondition(const Patch* patch,
                              int matID,
                              const string& type,
                              NCVariable<Vector>& variable,
                              string interp_type = "linear");

    /**
     * @brief
     * @param
     * @return
     */
    void setBoundaryCondition(const Patch* patch,
                              int matID,
                              const string& type,
                              NCVariable<double>& variable,
                              string interp_type = "linear");

};

}

#endif
