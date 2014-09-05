// copyright etc...

#include <Packages/Ptolemy/Dataflow/Modules/Converters/PTIIDataToNrrd.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>
//#include <Packages/Ptolemy/Core/PtolemyInterface/PTIIData.h>
#include <Core/Thread/CleanupManager.h>

#include <sci_defs/ptolemy_defs.h>

#include <iostream>

#include <stdlib.h> // for rand (test only)
#include <fstream> // for file streams (test only)

namespace Ptolemy {

#ifdef __linux
static void detach_callback(void *vwptr)
{
std::cerr << "detach_callback" << std::endl;
    if (JNIUtils::cachedJVM) {
        JNIUtils::cachedJVM->DetachCurrentThread();
    }
    if (JNIUtils::dataObjRef) {
        delete JNIUtils::dataObjRef;
    }
}
#endif

DECLARE_MAKER(PTIIDataToNrrd)
PTIIDataToNrrd::PTIIDataToNrrd(GuiContext *ctx)
  : Module("PTIIDataToNrrd", ctx, Source, "Converters", "Ptolemy"), points_handle_(0), connections_handle_(0)
{
    std::cerr << "PTIIDataToNrrd::PTIIDataToNrrd" << std::endl;
#ifdef __linux
    CleanupManager::add_callback(detach_callback, 0);
#endif
}

PTIIDataToNrrd::~PTIIDataToNrrd()
{
    std::cerr << "PTIIDataToNrrd::~PTIIDataToNrrd" << std::endl;
#ifdef __linux    
    CleanupManager::invoke_remove_callback(detach_callback, 0);
#endif
}

void PTIIDataToNrrd::execute()
{
    std::cerr << "PTIIDataToNrrd::execute()" << std::endl;
    update_state(NeedData);

    outportPoints = (NrrdOPort *)get_oport("Points");
    if (!outportPoints) {
        error("Unable to initialize oport 'Points'.");
        return;
    }
    outportConns = (NrrdOPort *)get_oport("Connections");
    if (!outportConns) {
        error("Unable to initialize oport 'Connections'.");
        return;
    }

    NrrdData *ptsNrrd;
    NrrdData *connNrrd;

    ptsNrrd = scinew NrrdData;
    ASSERT(ptsNrrd);
    points_handle_ = ptsNrrd;

    connNrrd = scinew NrrdData;
    ASSERT(connNrrd);
    connections_handle_ = connNrrd;

    if (JNIUtils::getMesh(points_handle_, connections_handle_)) {
        std::cerr << "Got object!" << std::endl;
        JNIUtils::sem().up();

        // Send the data downstream.
        if (points_handle_ != 0) {
            outportPoints->send(points_handle_);
        } else {
          error("No points available");
        }
        if (connections_handle_ != 0) {
            outportConns->send(connections_handle_);
        } else {
          error("No connections available");
        }
        update_state(Completed);
    } else {
        std::cerr << "Error: could not get object!" << std::endl;
    }
}

// void PTIIDataToNrrd::tcl_command(GuiArgs& args, void* userdata)
// {
// }


}
