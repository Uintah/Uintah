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
