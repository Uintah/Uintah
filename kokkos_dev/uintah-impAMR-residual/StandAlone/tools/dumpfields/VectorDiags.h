#ifndef DUMPFIELDS_VECTOR_DIAG_GEN_H
#define DUMPFIELDS_VECTOR_DIAG_GEN_H

#include <string>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include "FieldDiags.h"
#include "FieldSelection.h"
#include "TensorDiags.h"

namespace Uintah {
  using namespace SCIRun;
  
  // produce vector some field
  class VectorDiag : public FieldDiag
  {
  public:
    virtual ~VectorDiag();
    
    virtual std::string name()      const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, double time, 
                            NCVariable<Vector>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, double time, 
                            CCVariable<Vector>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, double time, 
                            ParticleSubset * parts,
                            ParticleVariable<Vector> & values) const = 0;
  };
  
  int                numberOfVectorDiags(const TypeDescription * fldtype);
  std::string        vectorDiagName     (const TypeDescription * fldtype, int idiag);
  VectorDiag const * createVectorDiag   (const TypeDescription * fldtype, int idiag,
                                         const TensorDiag * preop = 0);
  
  void describeVectorDiags(ostream & os);
  
  list<VectorDiag const *> createVectorDiags(const TypeDescription * fldtype, 
                                             const FieldSelection & fldselection,
                                             const TensorDiag * preop = 0);
}

#endif

