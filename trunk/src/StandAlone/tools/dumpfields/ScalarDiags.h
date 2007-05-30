#ifndef DUMPFIELDS_SCALAR_DIAG_GEN_H
#define DUMPFIELDS_SCALAR_DIAG_GEN_H

#include <string>
#include <stdio.h>

#include <SCIRun/Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>

#include "FieldDiags.h"
#include "FieldSelection.h"

namespace Uintah {
  //using namespace SCIRun;
  
  // produce scalar some field
  class ScalarDiag : public FieldDiag
  {
  public:
    virtual ~ScalarDiag();
    
    virtual std::string name() const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, int index, 
                            NCVariable<double>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, int index, 
                            CCVariable<double>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, int index,
                            ParticleSubset * pset,
                            ParticleVariable<double>  & values) const = 0;
  };
  
  int                numberOfScalarDiags(const Uintah::TypeDescription * fldtype);
  std::string        scalarDiagName     (const Uintah::TypeDescription * fldtype, int idiag);
  ScalarDiag const * createScalarDiag   (const Uintah::TypeDescription * fldtype, int idiag,
                                         const class TensorDiag * tensorpreop = 0);
  
  void describeScalarDiags(ostream & os);
  
  std::list<ScalarDiag const *> createScalarDiags(const Uintah::TypeDescription * fldtype, 
                                             const FieldSelection & fldselection,
                                             const class TensorDiag * tensorpreop = 0);
}

#endif

