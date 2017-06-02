/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef __BOUND_COND_FACTORY_H__
#define __BOUND_COND_FACTORY_H__

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/BoundaryConditions/BoundCondBase.h>

#include <vector>

#include <Core/ProblemSpec/ProblemSpec.h>  // for Vector definition


namespace Uintah {

class BoundCondFactory
{
public:
  // this function has a switch for all known BC_types
  static void create(ProblemSpecP& ps,BoundCondBase* &bc, int& mat_id, const std::string face_label);

  static void customBC(BoundCondBase* &bc, 
    int mat_id, const std::string face_label, double value ,std::string label, std::string var);

  static void customBC(BoundCondBase* &bc, 
    int mat_id, const std::string face_label, Vector value ,std::string label, std::string var);

  static void customBC(BoundCondBase* &bc, 
    int mat_id, const std::string face_label, std::string value ,std::string label, std::string var);

};

} // End namespace Uintah

#endif /* __BOUND_COND_FACTORY_H__ */
