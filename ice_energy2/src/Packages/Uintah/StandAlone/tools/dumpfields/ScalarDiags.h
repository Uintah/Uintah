#ifndef DUMPFIELDS_SCALAR_DIAG_GEN_H
#define DUMPFIELDS_SCALAR_DIAG_GEN_H

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
  
  // produce scalar some field
  class ScalarDiag : public FieldDiag
  {
  public:
    virtual ~ScalarDiag();
    
    virtual std::string name() const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, double time, 
                            NCVariable<double>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, double time, 
                            CCVariable<double>  & values) const = 0;
    
    virtual void operator()(DataArchive * da, const Patch * patch, 
                            const std::string & fieldname,
                            int imat, double time,
                            ParticleSubset * pset,
                            ParticleVariable<double>  & values) const = 0;
  };
  
  int                numberOfScalarDiags(const TypeDescription * fldtype);
  std::string        scalarDiagName     (const TypeDescription * fldtype, int idiag);
  ScalarDiag const * createScalarDiag   (const TypeDescription * fldtype, int idiag,
                                         const class TensorDiag * tensorpreop = 0);
  
  void describeScalarDiags(ostream & os);
  
  list<ScalarDiag const *> createScalarDiags(const TypeDescription * fldtype, 
                                             const FieldSelection & fldselection,
                                             const class TensorDiag * tensorpreop = 0);
}

#endif

