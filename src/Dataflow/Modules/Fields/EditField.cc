/*
 *  EditField.cc:
 *
 *  Written by:
 *   moulding
 *   April 22, 2001
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

namespace Moulding {

using namespace SCIRun;

class PSECORESHARE EditField : public Module {
public:
  EditField(const string& id);

  virtual ~EditField();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" PSECORESHARE Module* make_EditField(const string& id) {
  return scinew EditField(id);
}

EditField::EditField(const string& id)
  : Module("EditField", id, Source, "Fields", "Moulding")
{
}

EditField::~EditField(){
}

void EditField::execute(){
}

void EditField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Moulding


