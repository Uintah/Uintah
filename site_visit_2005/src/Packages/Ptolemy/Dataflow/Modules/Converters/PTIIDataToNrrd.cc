// copyright etc...

#include <Packages/Ptolemy/Dataflow/Modules/Converters/PTIIDataToNrrd.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>

#include <sci_defs/ptolemy_defs.h>

#include <iostream>

#include <stdlib.h> // for rand (test only)
#include <fstream> // for file streams (test only)

namespace Ptolemy {

DECLARE_MAKER(PTIIDataToNrrd)
PTIIDataToNrrd::PTIIDataToNrrd(GuiContext *ctx)
  : Module("PTIIDataToNrrd", ctx, Source, "Converters", "Ptolemy"), points(DEFAULT_NUM_POINTS, DEFAULT_POINTS_DIM), connections(DEFAULT_NUM_POINTS, DEFAULT_POINTS_DIM), points_handle_(0), connections_handle_(0), nPts(0), nConn(0), ptsDim(0), connDim(0), nrrdType(nrrdTypeUnknown), nrrdDim(UNSTRUCTURED_REGULAR_NRRD_DIM)
{
    std::cerr << "PTIIDataToNrrd::PTIIDataToNrrd" << std::endl;
}

PTIIDataToNrrd::~PTIIDataToNrrd()
{
    std::cerr << "PTIIDataToNrrd::~PTIIDataToNrrd" << std::endl;
}

// supporting only doubles for now
bool PTIIDataToNrrd::sendJNIData(int np, int nc, int pDim, int cDim)
{
    std::cerr << "PTIIDataToNrrd::sendJNIData()" << std::endl;
    // should check data ranges!
    nPts = np;
    nConn = nc;
    ptsDim = pDim;
    connDim = cDim;

    points.resize(nPts, pDim);
    connections.resize(nConn, cDim);
    
    nrrdType = nrrdTypeDouble;

    return true;
}

void PTIIDataToNrrd::execute()
{
    JNIUtils::dataSem().down();
    std::cerr << "PTIIDataToNrrd::execute()" << std::endl;

    outportConns = (NrrdOPort *)get_oport("Connections");
    if (!outportConns) {
        error("Unable to initialize oport 'Connections'.");
        return;
    }
    outportPoints = (NrrdOPort *)get_oport("Points");
    if (!outportPoints) {
        error("Unable to initialize oport 'Points'.");
        return;
    }

//     update_state(NeedData);
    NrrdData *ptsNrrd = scinew NrrdData;
    ASSERT(ptsNrrd);
    points_handle_ = ptsNrrd;

    NrrdData *connNrrd = scinew NrrdData;
    ASSERT(connNrrd);
    connections_handle_ = connNrrd;

//     update_state(Executing);

    std::cerr << "PTIIDataToNrrd::execute: points" << std::endl;
    int ptsNrrdDim = UNSTRUCTURED_REGULAR_NRRD_DIM;
    int ptsNrrdDims[NRRD_DIM_MAX];
    ptsNrrdDims[0] = ptsDim; // should be 3D points
    ptsNrrdDims[1] = nPts;
    nrrdAlloc_nva(ptsNrrd->nrrd, nrrdTypeDouble, ptsNrrdDim, ptsNrrdDims);
    //nrrdWrap_nva(ptsNrrd->nrrd, points.get_dataptr(), nrrdType, ptsNrrdDim, ptsNrrdDims);

    const char *ptslabelptr[NRRD_DIM_MAX];
    ptslabelptr[0] = "dim";
    ptslabelptr[1] = "points";
    nrrdAxisInfoSet_nva(ptsNrrd->nrrd, nrrdAxisInfoLabel, ptslabelptr);

    double *pData = (double *) ptsNrrd->nrrd->data;
    for (int i = 0; i < nPts; i++) {
        for (int j = 0; j < ptsDim; j++) {
            pData[j] = points(i, j);
        }
        pData += ptsDim;
    }

    std::cerr << "PTIIDataToNrrd::execute: connections" << std::endl;
    int connNrrdDim = UNSTRUCTURED_REGULAR_NRRD_DIM;
    int connNrrdDims[NRRD_DIM_MAX];
    connNrrdDims[0] = connDim; // initial test case: tetrahedron
    connNrrdDims[1] = nConn;
    nrrdAlloc_nva(connNrrd->nrrd, nrrdTypeDouble, connNrrdDim, connNrrdDims);
    //nrrdWrap_nva(connNrrd->nrrd, connections.get_dataptr(), nrrdType, connNrrdDim, connNrrdDims);

    const char *connlabelptr[NRRD_DIM_MAX];
    connlabelptr[0] = "dim";
    connlabelptr[1] = "connections";
    nrrdAxisInfoSet_nva(connNrrd->nrrd, nrrdAxisInfoLabel, connlabelptr);

    double *cData = (double *) connNrrd->nrrd->data;
    for (int i = 0; i < nConn; i++) {
        for (int j = 0; j < connDim; j++) {
            cData[j] = connections(i, j);
        }
        cData += connDim;
    }

    // Send the data downstream.
    if (points_handle_ != 0) {
        outportPoints->send(points_handle_);
    }
    if (connections_handle_ != 0) {
        outportConns->send(connections_handle_);
        if (connDim == 4) { // for now, assume all 4 pt connections are TetVolumes
            connections_handle_->set_property(std::string("Elem Type"), std::string("Tet"), false); // not transient
        }
    }
    JNIUtils::dataSem().up();
//     update_state(Completed);
}


// void PTIIDataToNrrd::tcl_command(GuiArgs& args, void* userdata)
// {
// }


}
