/*
 *  TetVolField2QuadraticTetVolField.cc:
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

class RobVSHARE TetVolField2QuadraticTetVolField : public Module {
public:
  TetVolField2QuadraticTetVolField(const string& id);

  virtual ~TetVolField2QuadraticTetVolField();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" RobVSHARE Module* make_TetVolField2QuadraticTetVolField(const string& id) {
  return scinew TetVolField2QuadraticTetVolField(id);
}

TetVolField2QuadraticTetVolField::TetVolField2QuadraticTetVolField(const string& id)
  : Module("TetVolField2QuadraticTetVolField", id, Source, "Quadratic", "RobV")
{
}

TetVolField2QuadraticTetVolField::~TetVolField2QuadraticTetVolField(){
}

void TetVolField2QuadraticTetVolField::execute(){
}

void TetVolField2QuadraticTetVolField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace RobV


