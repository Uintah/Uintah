#ifndef __DERIVE_EIGENEVALUATOR_H__
#define __DERIVE_EIGENEVALUATOR_H__

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>

#include <string>

using std::string;
using namespace SCIRun;
namespace Uintah {

  class EigenEvaluator: public Module {
  public:
    EigenEvaluator(const string& id);
    virtual ~EigenEvaluator() {}
    
    virtual void execute(void);
    
  private:
    //    TCLstring tcl_status;
    GuiInt guiEigenSelect;
    FieldIPort *in;

    FieldOPort *sfout; // for eigen values
    FieldOPort *vfout; // for eigen vectors
  };
}

#endif // __DERIVE_EIGENEVALUATOR_H__

