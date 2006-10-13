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


//    File   : MatrixInfo.cc
//    Author : McKay Davis
//    Date   : July 2002

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Containers/Handle.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace SCIRun {

using std::endl;
using std::pair;

class MatrixInfo : public Module {
private:
  GuiString gui_matrixname_;
  GuiString gui_generation_;
  GuiString gui_typename_;
  GuiString gui_rows_;
  GuiString gui_cols_;
  GuiString gui_elements_;

  int              generation_;

  void clear_vals();
  void update_input_attributes(MatrixHandle);

public:
  MatrixInfo(GuiContext* ctx);
  virtual ~MatrixInfo();
  virtual void execute();
};

  DECLARE_MAKER(MatrixInfo)

MatrixInfo::MatrixInfo(GuiContext* ctx)
  : Module("MatrixInfo", ctx, Sink, "Math", "SCIRun"),
    gui_matrixname_(get_ctx()->subVar("matrixname", false), "---"),
    gui_generation_(get_ctx()->subVar("generation", false), "---"),
    gui_typename_(get_ctx()->subVar("typename", false), "---"),
    gui_rows_(get_ctx()->subVar("rows", false), "---"),
    gui_cols_(get_ctx()->subVar("cols", false), "---"),
    gui_elements_(get_ctx()->subVar("elements", false), "---"),
    generation_(-1)
{
}


MatrixInfo::~MatrixInfo()
{
}


void
MatrixInfo::clear_vals()
{
  gui_matrixname_.set("---");
  gui_generation_.set("---");
  gui_typename_.set("---");
  gui_rows_.set("---");
  gui_cols_.set("---");
  gui_elements_.set("---");
}


void
MatrixInfo::update_input_attributes(MatrixHandle m)
{
  const bool regressing = sci_getenv_p("SCI_REGRESSION_TESTING");

  string matrixname;
  if (!m->get_property("name", matrixname))
  {
    matrixname = "--- Name Not Assigned ---";
  }
  gui_matrixname_.set(matrixname);
  if (regressing) { remark("Name: " + matrixname); }

  gui_generation_.set(to_string(m->generation));
  if (regressing) { remark("Generation: " + to_string(m->generation)); }

  // Set the typename.
  string kind;
  string size;
  if (m->is_sparse())
  {
    kind = "SparseRowMatrix";
    size = to_string(m->sparse()->get_nnz());
  }
  else if (m->is_column())
  {
    kind = "ColumnMatrix";
    size = to_string(m->ncols() * m->nrows());
  }
  else if (m->is_dense_col_maj())
  {
    kind = "DenseColMajMatrix";
    size = to_string(m->ncols() * m->nrows());
  }
  else if (m->is_dense())
  {
    kind = "DenseMatrix";
    size = to_string(m->ncols() * m->nrows());
  }
  else
  {
    kind = "Unknown";
    size = "*";
  }
  gui_typename_.set(kind);
  gui_elements_.set(size);
  if (regressing)
  {
    remark("Type: " + kind);
  }

  gui_rows_.set(to_string(m->nrows()));
  gui_cols_.set(to_string(m->ncols()));
  if (regressing)
  {
    remark("Rows: " + to_string(m->nrows()));
    remark("Cols: " + to_string(m->ncols()));
    remark("Elems: " + size);
  }
}


void
MatrixInfo::execute()
{
  // The input port (with data) is required.
  MatrixHandle mh;
  if (!get_input_handle("Input", mh, false))
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
}


} // end SCIRun namespace


