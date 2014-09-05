/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
};

} // end namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(FieldToNrrd)


FieldToNrrd::FieldToNrrd(GuiContext *ctx):
  Module("FieldToNrrd", ctx, Filter, "DataIO", "Teem"),
  label_(ctx->subVar("label"))
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

  const TypeDescription *td = field_handle->get_type_description();
  CompileInfoHandle ci = ConvertToNrrdBase::get_compile_info(td);
  Handle<ConvertToNrrdBase> algo;
  if (!module_dynamic_compile(ci, algo)) return;  
  
  label_.reset();
  string lab = label_.get();
  NrrdDataHandle onrrd_handle = algo->convert_to_nrrd(field_handle, lab);

  onrrd_handle->set_orig_field(field_handle);

  onrrd_->send(onrrd_handle);
}

