/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef DUMPFIELDS_TENSOR_DIAG_GEN_H
#define DUMPFIELDS_TENSOR_DIAG_GEN_H

#include <string>
#include <stdio.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>

#include "FieldDiags.h"
#include "FieldSelection.h"

namespace Uintah {

  // produce tensor diagnostic from tensor field
  class TensorDiag : public FieldDiag
  {
  public:
    virtual ~TensorDiag();
    
    virtual std::string name() const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, int index, 
                            NCVariable<Matrix3>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, int index, 
                            CCVariable<Matrix3>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, int index, 
                            ParticleSubset * parts,
                            ParticleVariable<Matrix3> & values) const = 0;
  };
  
  int              numberOfTensorDiags(const Uintah::TypeDescription * fldtype);
  std::string      tensorDiagName     (const Uintah::TypeDescription * fldtype, int idiag);
  TensorDiag const * createTensorDiag (const Uintah::TypeDescription * fldtype, int idiag,
                                       const TensorDiag * preop=0);
  
  void describeTensorDiags(ostream & os);
  
  // list of requested tensor diagnostics
  std::list<const Uintah::TensorDiag*> createTensorDiags(const Uintah::TypeDescription * fldtype, 
                                                         const SCIRun::FieldSelection & fldselection,
                                                         const Uintah::TensorDiag * preop=0);
  
  // create single tensor op, or null if none requested
  const TensorDiag * createTensorOp(const SCIRun::FieldSelection & fldselection);
  
}

#endif

