/*
 *  GenerateTetVol.cc:
 *
 *  Written by:
 *   mcole
 *   TODAY'S DATE HERE
 *
 */

#include <sci_defs.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/VDT/share/share.h>

namespace VDT {
extern "C" {
#include <vdtpub.h>
}

using namespace SCIRun;

class VDTSHARE GenerateTetVol : public Module {
public:
  GenerateTetVol(const string& id);

  virtual ~GenerateTetVol();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" VDTSHARE Module* make_GenerateTetVol(const string& id) {
  return scinew GenerateTetVol(id);
}

GenerateTetVol::GenerateTetVol(const string& id)
  : Module("GenerateTetVol", id, Source, "Modeling", "VDT")
{
}

GenerateTetVol::~GenerateTetVol(){
}

void GenerateTetVol::execute()
{
  // just test that the lib is loaded and we can make a call into it for now.
  VDT gentv_vdt = VDT_new_mesher();
  cout << "completed" << endl;

}

void GenerateTetVol::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace VDT


