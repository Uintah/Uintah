#ifndef __VISUALIZATION_EIGENEVALUATOR_H__
#define __VISUALIZATION_EIGENEVALUATOR_H__

#include <Dataflow/Network/Module.h>
#include <Core/Containers/String.h>
#include <Packages/Uintah/Core/Datatypes/TensorFieldPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/VectorFieldPort.h>

namespace Uintah {

using namespace SCIRun;

  class EigenEvaluator: public Module {
  public:
    EigenEvaluator(const clString& id);
    virtual ~EigenEvaluator() {}
    
    virtual void execute(void);
    
  private:
    //    GuiString tcl_status;
    GuiInt tclEigenSelect;
    TensorFieldIPort *in;

    ScalarFieldOPort *sfout; // for eigen values
    VectorFieldOPort *vfout; // for eigen vectors
  };

} // End namespace Uintah

#endif // __VISUALIZATION_EIGENEVALUATOR_H__
