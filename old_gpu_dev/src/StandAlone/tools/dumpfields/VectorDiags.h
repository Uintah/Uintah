/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef DUMPFIELDS_VECTOR_DIAG_GEN_H
#define DUMPFIELDS_VECTOR_DIAG_GEN_H

#include <string>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>
#include "FieldDiags.h"
#include "FieldSelection.h"
#include "TensorDiags.h"

namespace Uintah {
  
  // produce vector some field
  class VectorDiag : public FieldDiag
  {
  public:
    virtual ~VectorDiag();
    
    virtual std::string name()      const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, int index, 
                            NCVariable<Vector>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, int index, 
                            CCVariable<Vector>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, int index, 
                            ParticleSubset * parts,
                            ParticleVariable<Vector> & values) const = 0;
  };
  
  int                numberOfVectorDiags(const Uintah::TypeDescription * fldtype);
  std::string        vectorDiagName     (const Uintah::TypeDescription * fldtype, int idiag);
  VectorDiag const * createVectorDiag   (const Uintah::TypeDescription * fldtype, int idiag,
                                         const TensorDiag * preop = 0);
  
  void describeVectorDiags(ostream & os);
  
  std::list<Uintah::VectorDiag const *> createVectorDiags(const Uintah::TypeDescription * fldtype, 
                                             const SCIRun::FieldSelection & fldselection,
                                             const Uintah::TensorDiag * preop = 0);
}

#endif

