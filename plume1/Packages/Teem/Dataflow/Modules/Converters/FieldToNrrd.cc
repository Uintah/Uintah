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
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Teem/Dataflow/Modules/Converters/ConvertToNrrd.h>
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
  NrrdOPort   *opoints_;
  NrrdOPort   *oconnect_;
  NrrdOPort   *odata_;
  
  GuiString    label_;
  int          ifield_generation_;
  NrrdDataHandle points_handle_, connect_handle_, data_handle_;
};

} // end namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(FieldToNrrd)


FieldToNrrd::FieldToNrrd(GuiContext *ctx):
  Module("FieldToNrrd", ctx, Filter, "Converters", "Teem"),
  label_(ctx->subVar("label")),
  ifield_generation_(-1),
  points_handle_(0),  connect_handle_(0),  data_handle_(0)
{
}

FieldToNrrd::~FieldToNrrd()
{
}

void FieldToNrrd::execute()
{
  ifield_ = (FieldIPort *)get_iport("Field");
  opoints_ = (NrrdOPort *)get_oport("Points");
  oconnect_ = (NrrdOPort *)get_oport("Connections");
  odata_ = (NrrdOPort *)get_oport("Data");
  
  FieldHandle field_handle; 
  if (!(ifield_->get(field_handle) && field_handle.get_rep())) {
    return;
  }

  // Just data for lattices, data and points for structured, all for rest.
  bool compute_points_p = true;
  bool compute_connects_p = true;
  bool compute_data_p = true;

  if (!field_handle->mesh()->is_editable())
  {
    if (oconnect_->nconnections())
    {
      warning("Not computing connections for non-editable mesh type.");
    }
    else
    {
      remark("Not computing connections for non-editable mesh type.");
    }
    compute_connects_p = false;
  }
  
  const string meshstr =
    field_handle->get_type_description(0)->get_name().substr(0, 6);
  if (!(field_handle->mesh()->is_editable() || meshstr == "Struct"))
  {
    if (opoints_->nconnections())
    {
      warning("Not computing points for strict lattice.");
    }
    else
    {
      remark("Not computing points for strict lattice.");
    }
    compute_points_p = false;
  }
  
  if (field_handle->basis_order() == -1)
  {
    if (odata_->nconnections())
    {
      warning("No data in input field.");
    }
    else
    {
      remark("No data in input field.");
    }
    compute_data_p = false;
  }

  if (ifield_generation_ != field_handle->generation)
  {
    ifield_generation_ = field_handle->generation;
    points_handle_ = connect_handle_ = data_handle_ = 0;
    
    const TypeDescription *td = field_handle->get_type_description();
    CompileInfoHandle ci = ConvertToNrrdBase::get_compile_info(td);
    Handle<ConvertToNrrdBase> algo;
    if (!module_dynamic_compile(ci, algo)) return;  
    
    label_.reset();
    algo->convert_to_nrrd(field_handle, points_handle_, 
			  connect_handle_, data_handle_,
			  compute_points_p, compute_connects_p,
			  compute_data_p, label_.get());
  }
  
  // set the Nrrd names and send them
  string property;
  string nrrd_name = "Unknown";
  if (field_handle->get_property( "name", property ) && property != "Unknown") 
    nrrd_name = property;
  if (points_handle_ != 0) {
    opoints_->send(points_handle_);
    points_handle_->set_property("Name", nrrd_name + "-Points", false);
  }
  if (connect_handle_ != 0) {
    oconnect_->send(connect_handle_);
    connect_handle_->set_property("Name", nrrd_name + "-Connectivity", false);
  }
  if (data_handle_ != 0) {
    odata_->send(data_handle_);
    data_handle_->set_property("Name", nrrd_name + "-Data", false);
  }

}

