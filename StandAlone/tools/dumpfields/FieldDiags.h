#ifndef DUMPFIELDS_FIELD_DIAGS_GEN_H
#define DUMPFIELDS_FIELD_DIAGS_GEN_H

#include <string>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

namespace Uintah {
  using namespace SCIRun;
  
  class FieldDiag
  {
  public:
    virtual ~FieldDiag();
    
    virtual std::string name() const = 0;
    
    virtual bool has_mass(DataArchive * da, const Patch * patch, 
                          TypeDescription::Type fieldtype,
                          int imat, double time, const IntVector & pt) const;
  };
}

#endif

