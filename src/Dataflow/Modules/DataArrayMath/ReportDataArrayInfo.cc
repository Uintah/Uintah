/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <map>
#include <iostream>

namespace SCIRun {

class ReportDataArrayInfo : public Module {
public:
  ReportDataArrayInfo(GuiContext*);

  virtual ~ReportDataArrayInfo();

  virtual void execute();

private:
  GuiString gui_matrixname_;
  GuiString gui_generation_;
  GuiString gui_typename_;
  GuiString gui_elements_;

  int generation_;

  void clear_vals();
  void update_input_attributes(MatrixHandle);

};


DECLARE_MAKER(ReportDataArrayInfo)
ReportDataArrayInfo::ReportDataArrayInfo(GuiContext* ctx)
  : Module("ReportDataArrayInfo", ctx, Source, "DataArrayMath", "SCIRun"),
    gui_matrixname_(get_ctx()->subVar("matrixname", false)),
    gui_generation_(get_ctx()->subVar("generation", false)),
    gui_typename_(get_ctx()->subVar("typename", false)),
    gui_elements_(get_ctx()->subVar("elements", false)),
    generation_(-1)
{
  gui_matrixname_.set("---");
  gui_generation_.set("---");
  gui_typename_.set("---");
  gui_elements_.set("---");
}


ReportDataArrayInfo::~ReportDataArrayInfo()
{
}


void
ReportDataArrayInfo::execute()
{
  // The input port (with data) is required.
  MatrixHandle mh;
  if (!get_input_handle("DataArray", mh))
  {
    clear_vals();
    generation_ = -1;
    return;
  }

  if (generation_ != mh.get_rep()->generation)
  {
    generation_ = mh.get_rep()->generation;
    update_input_attributes(mh);
  }
  
  MatrixHandle nrows = scinew DenseMatrix(1, 1);
  double* dataptr = nrows->get_data_pointer();
  dataptr[0] = mh->nrows();
  send_output_handle("NumElements", nrows);
}


void
ReportDataArrayInfo::clear_vals()
{
  gui_matrixname_.set("---");
  gui_generation_.set("---");
  gui_typename_.set("---");
  gui_elements_.set("---");
}


void
ReportDataArrayInfo::update_input_attributes(MatrixHandle m)
{
  string matrixname;
  if (m->get_property("name", matrixname))
  {
    gui_matrixname_.set(matrixname);
  }
  else
  {
    gui_matrixname_.set("--- Name Not Assigned ---");
  }

  gui_generation_.set(to_string(m->generation));

  // Set the typename.
  if (m->ncols() == 1)
  {
    gui_typename_.set("ScalarArray");
    gui_elements_.set(to_string(m->nrows()));  
  }
  else if (m->ncols() == 3)
  {
    gui_typename_.set("VectorArray");
    gui_elements_.set(to_string(m->nrows()));  
  }
  else if ((m->ncols() == 6)||(m->ncols() == 9))
  {
    gui_typename_.set("TensorArray");
    gui_elements_.set(to_string(m->nrows()));  
  }
  else
  {
    gui_typename_.set("--- invalid array type ---");
    gui_elements_.set(to_string(m->nrows()));  
  }
}

} // End namespace SCIRun


