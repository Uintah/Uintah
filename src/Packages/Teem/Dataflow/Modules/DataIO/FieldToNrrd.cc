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
  FieldIPort* ifield;
  NrrdOPort* onrrd;
public:
  FieldToNrrd(GuiContext *ctx);
  virtual ~FieldToNrrd();
  virtual void execute();
};

} // end namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(FieldToNrrd)


FieldToNrrd::FieldToNrrd(GuiContext *ctx):
  Module("FieldToNrrd", ctx, Filter, "DataIO", "Teem")
{
}

FieldToNrrd::~FieldToNrrd()
{
}

#define COPY_INTO_NRRD_FROM_FIELD(type, Type) \
    LatVolField<type> *f = \
      dynamic_cast<LatVolField<type>*>(field); \
    lvm = f->get_typed_mesh(); \
    nx = f->fdata().dim3(); \
    ny = f->fdata().dim2(); \
    nz = f->fdata().dim1(); \
    type *data=new type[nx*ny*nz]; \
    type *p=&(data[0]); \
    for (int k=0; k<nz; k++) \
      for (int j=0; j<ny; j++) \
	for (int i=0; i<nx; i++) \
	  *p++=f->fdata()(k,j,i); \
    nrrdWrap(nout->nrrd, data, nrrdType##Type, 3, nx, ny, nz); \
    if (f->data_at() == Field::NODE) \
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, \
                  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode); \
    else \
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, \
                  nrrdCenterCell, nrrdCenterCell, nrrdCenterCell)

void FieldToNrrd::execute()
{
  ifield = (FieldIPort *)get_iport("Field");
  onrrd = (NrrdOPort *)get_oport("Nrrd");

  if (!ifield) {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  if (!onrrd) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }

  FieldHandle field_handle;
  if (!ifield->get(field_handle))
    return;

  const TypeDescription *td = field_handle->get_type_description();
  CompileInfoHandle ci = ConvertToNrrdBase::get_compile_info(td);
  Handle<ConvertToNrrdBase> algo;
  if (!module_dynamic_compile(ci, algo)) return;  

  NrrdDataHandle onrrd_handle = algo->convert_to_nrrd(field_handle);

  onrrd->send(onrrd_handle);
}

