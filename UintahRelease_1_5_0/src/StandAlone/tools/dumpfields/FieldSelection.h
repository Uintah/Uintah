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

#ifndef DUMPFIELDS_FIELD_SELECTION_H
#define DUMPFIELDS_FIELD_SELECTION_H

/// help user select 
///    field
///    material
///    component (including diagnostics)
///

#include "Args.h"
#include <vector>
#include <string>

namespace SCIRun {
  
  class FieldSelection {
  public:
    FieldSelection(Args & args, const std::vector<std::string> & allfields);
    
    bool wantField      (std::string fieldname) const;
    bool wantDiagnostic (std::string diagname)  const;
    bool wantTensorOp   (std::string diagname)  const;
    bool wantMaterial   (int imat)              const;
    
    static std::string options() { 
      return \
        std::string("      -field f1,f2,f3        list of fields to write. default is all.") + "\n" +
        std::string("      -material m1,m2,m3     list of integer materials to write. default is all.") + "\n" +
        std::string("      -diagnostic d1,d2,d3   list of diagnostics to use. default is value.") + "\n" +
        std::string("      -tensor_op op          tensor transformation to make before generating diagnostics.");
    }
    
  private:
    std::vector<std::string> fieldnames;
    std::vector<std::string> diagnames;
    std::vector<int>         mats;
    std::vector<std::string> opnames;
  };
}

#endif
