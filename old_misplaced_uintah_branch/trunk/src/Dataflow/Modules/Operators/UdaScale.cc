/*
 *  UdaScale.cc:
 *
 *  Written by:
 *   kuzimmer
 *   TODAY'S DATE HERE
 *
 */

#include <SCIRun/Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <SCIRun/Core/Geometry/Vector.h>
#include <SCIRun/Dataflow/GuiInterface/GuiVar.h> 
#include <Core/Datatypes/Archive.h>
#include <Core/DataArchive/DataArchive.h>
#include <Dataflow/Ports/ArchivePort.h>

namespace Uintah {

using namespace SCIRun;

class UdaScale : public Module {
public:
  UdaScale(GuiContext*);

  virtual ~UdaScale();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiDouble cell_scale_;
  double current_scale_;
};


DECLARE_MAKER(UdaScale)
UdaScale::UdaScale(GuiContext* ctx)
  : Module("UdaScale", ctx, Source, "Operators", "Uintah"),
    cell_scale_(get_ctx()->subVar("cell-scale"), 1.0),
    current_scale_(1.0)
{

}

UdaScale::~UdaScale(){
}

void
 UdaScale::execute(){
  ArchiveIPort *in =  (ArchiveIPort *) get_iport("Data Archive");
  ArchiveOPort *out = (ArchiveOPort *) get_oport("Data Archive");

  ArchiveHandle handle;

  if (!(in->get(handle) && handle.get_rep())) {
    warning("Input Archive is empty.");
    return;
  }

  double s = cell_scale_.get();
  if( s != current_scale_ ){
    DataArchiveHandle archive = handle->getDataArchive();
    Vector cell_scale( s, s, s );
    archive->setCellScale( cell_scale );
    handle->generation = handle->compute_new_generation();
    current_scale_ = s;
  }
  out->send(handle);
  
}

void
 UdaScale::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Uintah


