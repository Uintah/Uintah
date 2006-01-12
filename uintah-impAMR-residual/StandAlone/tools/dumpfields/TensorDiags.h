#ifndef DUMPFIELDS_TENSOR_DIAG_GEN_H
#define DUMPFIELDS_TENSOR_DIAG_GEN_H

#include <string>
#include <stdio.h>

#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include "FieldDiags.h"
#include "FieldSelection.h"

namespace Uintah {
  using namespace SCIRun;
  
  // produce tensor diagnostic from tensor field
  class TensorDiag : public FieldDiag
  {
  public:
    virtual ~TensorDiag();
    
    virtual std::string name() const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, double time, 
                            NCVariable<Matrix3>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, double time, 
                            CCVariable<Matrix3>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, double time, 
                            ParticleSubset * parts,
                            ParticleVariable<Matrix3> & values) const = 0;
  };
  
  int              numberOfTensorDiags(const TypeDescription * fldtype);
  std::string      tensorDiagName     (const TypeDescription * fldtype, int idiag);
  TensorDiag const * createTensorDiag (const TypeDescription * fldtype, int idiag,
                                       const TensorDiag * preop=0);
  
  void describeTensorDiags(ostream & os);
  
  // list of requested tensor diagnostics
  list<TensorDiag const *> createTensorDiags(const TypeDescription * fldtype, 
                                             const FieldSelection & fldselection,
                                             const TensorDiag * preop=0);
  
  // create single tensor op, or null if none requested
  TensorDiag const * createTensorOp(const FieldSelection & fldselection);
  
}

#endif

