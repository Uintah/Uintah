/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
 *  ChooseImage.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Containers/StringUtil.h>

#include <Insight/Dataflow/Ports/ITKDatatypePort.h>

namespace Insight {

using namespace SCIRun;

class ChooseImage : public Module {
private:
  GuiInt port_index_;
  GuiInt usefirstvalid_;
public:
  ChooseImage(GuiContext*);

  virtual ~ChooseImage();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ChooseImage)
ChooseImage::ChooseImage(GuiContext* ctx)
  : Module("ChooseImage", ctx, Source, "DataIO", "Insight"),
    port_index_(ctx->subVar("port-index")),
    usefirstvalid_(ctx->subVar("usefirstvalid"))
{
}

ChooseImage::~ChooseImage(){
}

void ChooseImage::execute(){
  ITKDatatypeOPort *oimg = (ITKDatatypeOPort *)get_oport("OutputImage");
  if (!oimg) {
    error("Unable to initialize oport 'OututImage'.");
    return;
  }

  update_state(NeedData);

  port_range_type range = get_iports("InputImage");
  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;

  int usefirstvalid = usefirstvalid_.get();

  ITKDatatypeIPort *ifield = 0;
  ITKDatatypeHandle field;
  
  if (usefirstvalid) {
    // iterate over the connections and use the
    // first valid field
    int idx = 0;
    bool found_valid = false;
    while (pi != range.second) {
      ifield = (ITKDatatypeIPort *)get_iport(idx);
      if (ifield->get(field) && field != 0) {
	found_valid = true;
	break;
      }
      ++idx;
      ++pi;
    }
    if (!found_valid) {
      error("Didn't find any valid images\n");
      return;
    }
  } else {
    // use the index specified
    int idx=port_index_.get();
    if (idx<0) { error("Can't choose a negative port"); return; }
    while (pi != range.second && idx != 0) { ++pi ; idx--; }
    int port_number=pi->second;
    if (pi == range.second || ++pi == range.second) { 
      error("Selected port index out of range"); return; 
    }

    ifield = (ITKDatatypeIPort *)get_iport(port_number);
    if (!ifield) {
      error("Unable to initialize iport '" + to_string(port_number) + "'.");
      return;
    }
    ifield->get(field);
  }
  
  oimg->send(field);
}

void ChooseImage::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


