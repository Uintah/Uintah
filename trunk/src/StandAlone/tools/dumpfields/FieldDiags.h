#ifndef DUMPFIELDS_FIELD_DIAGS_GEN_H
#define DUMPFIELDS_FIELD_DIAGS_GEN_H

#include <string>
#include <SCIRun/Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>

namespace Uintah {
  using namespace SCIRun;
  
  class FieldDiag
  {
  public:
    virtual ~FieldDiag();
    
    virtual std::string name() const = 0;
    
    virtual bool has_mass(DataArchive * da, const Patch * patch, 
                          Uintah::TypeDescription::Type fieldtype,
                          int imat, int index, const IntVector & pt) const;
  };
}

#endif

