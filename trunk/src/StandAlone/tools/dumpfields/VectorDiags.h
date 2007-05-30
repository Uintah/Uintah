#ifndef DUMPFIELDS_VECTOR_DIAG_GEN_H
#define DUMPFIELDS_VECTOR_DIAG_GEN_H

#include <string>
#include <SCIRun/Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>
#include "FieldDiags.h"
#include "FieldSelection.h"
#include "TensorDiags.h"

namespace Uintah {
  //using namespace SCIRun;
  
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

