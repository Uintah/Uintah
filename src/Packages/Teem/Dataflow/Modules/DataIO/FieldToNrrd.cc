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
 *  FieldToNrrd.cc:  Convert a Nrrd to a Field
 *
 *  Written by:
 *   David Weinstein
 *   School of Computing
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Teem/Dataflow/Modules/DataIO/ConvertToNrrd.h>
#include <iostream>
#include <utility>

using std::endl;
using std::pair;

namespace SCITeem {

using namespace SCIRun;

class FieldToNrrd : public Module {
public:
  FieldToNrrd(GuiContext *ctx);
  virtual ~FieldToNrrd();
  virtual void execute();
private:
  FieldIPort  *ifield_;
  NrrdOPort   *onrrd_;
  
  GuiString    label_;
  int          ifield_generation_;
  NrrdDataHandle onrrd_handle_;
};

} // end namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(FieldToNrrd)


FieldToNrrd::FieldToNrrd(GuiContext *ctx):
  Module("FieldToNrrd", ctx, Filter, "DataIO", "Teem"),
  label_(ctx->subVar("label")),
  ifield_generation_(-1),
  onrrd_handle_(0)
{
}

FieldToNrrd::~FieldToNrrd()
{
}

void FieldToNrrd::execute()
{
  ifield_ = (FieldIPort *)get_iport("Field");
  onrrd_ = (NrrdOPort *)get_oport("Nrrd");
  
  if (!ifield_) {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  
  if (!onrrd_) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }
  
  FieldHandle field_handle; 
  if (!ifield_->get(field_handle))
    return;
  
  if (ifield_generation_ != field_handle->generation) {
    ifield_generation_ = field_handle->generation;
    
    const TypeDescription *td = field_handle->get_type_description();
    CompileInfoHandle ci = ConvertToNrrdBase::get_compile_info(td);
    Handle<ConvertToNrrdBase> algo;
    if (!module_dynamic_compile(ci, algo)) return;  
    
    label_.reset();
    string lab = label_.get();
    onrrd_handle_ = algo->convert_to_nrrd(field_handle, lab);
  }
  
  onrrd_->send(onrrd_handle_);
}

