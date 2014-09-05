// copyright etc.

#ifndef Ptolemy_Dataflow_Modules_Converters_PTIIDataToNrrd_h
#define Ptolemy_Dataflow_Modules_Converters_PTIIDataToNrrd_h

#include <Dataflow/Network/Module.h>
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
    // see JNIUtils::scirunData
    bool sendJNIData(int np, int nc, int pDim, int cDim);

    Array2<double> points;
    Array2<double> connections;
//     virtual void tcl_command(GuiArgs&, void*);

private:
    NrrdOPort *outportPoints;
    NrrdOPort *outportConns;

    // using the Teem package's NrrdToField converter as a guide
    NrrdDataHandle points_handle_, connections_handle_;
    //NrrdDataHandle data_handle_;

    int nPts;
    int nConn;
    int ptsDim;
    int connDim;
    int nrrdType;
    int nrrdDim;

    const static int UNSTRUCTURED_REGULAR_NRRD_DIM = 2;
    const static int DEFAULT_NUM_POINTS = 100;
    const static int DEFAULT_POINTS_DIM = 3;
};

// eventually dynamic algo.?
// (child class of DynamicAlgoBase - see Core/Util/DynamicLoader.h)
// template<class T>
// class ConvertToNrrd {
// public:
//   bool convert_to_nrrd(Array2<T> points, Array2<T> connections);
// };


}

#endif
