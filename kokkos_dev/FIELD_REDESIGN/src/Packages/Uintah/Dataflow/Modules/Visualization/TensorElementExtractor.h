#ifndef __VISUALIZATION_TENSORELEMENTEXTRACTOR_H__
#define __VISUALIZATION_TENSORELEMENTEXTRACTOR_H__

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Containers/String.h>
#include <Uintah/Datatypes/TensorFieldPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>

namespace Uintah {
namespace Modules {
  using namespace SCICore::Containers;
  using namespace PSECore::Dataflow;
  using namespace SCICore::Datatypes;
  using namespace PSECore::Datatypes;

  class TensorElementExtractor: public Module {
  public:
    TensorElementExtractor(const clString& id);
    virtual ~TensorElementExtractor() {}
    
    virtual void execute(void);
    
  private:
    //    TCLstring tcl_status;
    TCLint tclRow;
    TCLint tclColumn;
    TensorFieldIPort *in;

    ScalarFieldOPort *sfout; // output elements
  };
}
}
#endif // __VISUALIZATION_TENSORELEMENTEXTRACTOR_H__

