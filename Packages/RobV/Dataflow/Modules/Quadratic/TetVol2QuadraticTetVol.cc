/*
 *  TetVol2QuadraticTetVol.cc:
 *
 *  Written by:
 *   mcole
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/RobV/share/share.h>

namespace RobV {

using namespace SCIRun;

class RobVSHARE TetVol2QuadraticTetVol : public Module {
public:
  TetVol2QuadraticTetVol(const string& id);

  virtual ~TetVol2QuadraticTetVol();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" RobVSHARE Module* make_TetVol2QuadraticTetVol(const string& id) {
  return scinew TetVol2QuadraticTetVol(id);
}

TetVol2QuadraticTetVol::TetVol2QuadraticTetVol(const string& id)
  : Module("TetVol2QuadraticTetVol", id, Source, "Quadratic", "RobV")
{
}

TetVol2QuadraticTetVol::~TetVol2QuadraticTetVol(){
}

void TetVol2QuadraticTetVol::execute(){
}

void TetVol2QuadraticTetVol::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace RobV


