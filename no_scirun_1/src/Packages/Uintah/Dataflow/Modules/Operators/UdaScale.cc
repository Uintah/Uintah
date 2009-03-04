/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


/*
 *  UdaScale.cc:
 *
 *  Written by:
 *   kuzimmer
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Vector.h>
#include <Core/GuiInterface/GuiVar.h> 
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>

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


