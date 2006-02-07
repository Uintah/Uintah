// copyright etc.

#ifndef Ptolemy_Dataflow_Modules_Converters_PTIIDataToNrrd_h
#define Ptolemy_Dataflow_Modules_Converters_PTIIDataToNrrd_h


#include <Dataflow/Network/Module.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <Core/Containers/Array2.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;

namespace Ptolemy {

// borrowed heavily from SCITeem::NrrdReader
// and MatlabIO::MatlabNrrdsReader
class PTIIDataToNrrd : public Module {
public:
    PTIIDataToNrrd(GuiContext*);
    virtual ~PTIIDataToNrrd();

    virtual void execute();
//     virtual void tcl_command(GuiArgs&, void*);

private:
    NrrdOPort *outportPoints;
    NrrdOPort *outportConns;

    // using the Teem package's NrrdToField converter as a guide
    NrrdDataHandle points_handle_, connections_handle_;
    //NrrdDataHandle data_handle_;
};

}

#endif
