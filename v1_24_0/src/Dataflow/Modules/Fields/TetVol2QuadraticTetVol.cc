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

//    File   : TetVol2QuadraticTetVol.cc
//    Author : Martin Cole
//    Date   : Wed Feb 27 09:01:36 2002

#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ConvertTet.h>
#include <Core/Containers/Handle.h>


namespace SCIRun {

class TetVol2QuadraticTetVol : public Module {
public:
  TetVol2QuadraticTetVol(GuiContext* ctx);

  virtual ~TetVol2QuadraticTetVol();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  void debug_tets(TetVolMeshHandle in, 
		  QuadraticTetVolMeshHandle out);
private:
  FieldIPort*              ifld_;
  FieldOPort*              ofld_;
};

DECLARE_MAKER(TetVol2QuadraticTetVol)
TetVol2QuadraticTetVol::TetVol2QuadraticTetVol(GuiContext* ctx)
  : Module("TetVol2QuadraticTetVol", ctx, Filter, "FieldsGeometry", "SCIRun")
{
}

TetVol2QuadraticTetVol::~TetVol2QuadraticTetVol(){
}

void TetVol2QuadraticTetVol::execute()
{
  ifld_ = (FieldIPort *)get_iport("InputTetField");
  FieldHandle fld_handle;
  
  ifld_->get(fld_handle);

  if(!fld_handle.get_rep())
  {
    warning("No Data in port 1 field.");
    return;
  }
  else if (fld_handle->mesh()->get_type_description()->get_name() !=
	   get_type_description((TetVolMesh *)0)->get_name())
  {
    error("Input must be a TetVol type.");
    return;
  }
  const TypeDescription *td = fld_handle->get_type_description();
  CompileInfoHandle ci = ConvertTetBase::get_compile_info(td);
  Handle<ConvertTetBase> algo;
  if (!module_dynamic_compile(ci, algo)) return;  

  FieldHandle ofld_handle = algo->convert_quadratic(fld_handle);

  // debug_tets(((TetVolField<double>*)fld_handle.get_rep())->get_typed_mesh(), 
  //   ((QuadraticTetVolField<double>*)ofld_handle.get_rep())->get_typed_mesh());

  ofld_ = (FieldOPort *)get_oport("OutputQuadraticTet");
  ofld_->send(ofld_handle);
}

void 
TetVol2QuadraticTetVol::debug_tets(TetVolMeshHandle in, 
				   QuadraticTetVolMeshHandle out)
{
  cout << "-------- Linear Tets -----------------\n";
  TetVolMesh::Cell::iterator iter, endit;
  in->begin(iter);
  in->end(endit);
  int tet = 1;
  while (iter != endit) {
    cout << "Tet num: " << tet++ << "\n";
    TetVolMesh::Cell::index_type idx = *iter;
    ++iter;
    TetVolMesh::Node::array_type n;
    in->get_nodes(n, idx);
    TetVolMesh::Node::array_type::iterator niter = n.begin();
    while (niter != n.end()) {
      cout << "Node: " << *niter + 1 << "\n";
      Point p;
      in->get_center(p, *niter);
      ++niter;
      cout << p << "\n\n";
    }
  }

  cout << "-------- Quadratic Tets -----------------" << "\n";
  QuadraticTetVolMesh::Cell::iterator qiter, qendit;
  out->begin(qiter);
  out->end(qendit);
  int qtet = 1;
  while (qiter != qendit) {
    cout << "Tet num: " << qtet++ << "\n";
    QuadraticTetVolMesh::Cell::index_type idx = *qiter;
    ++qiter;
    QuadraticTetVolMesh::Node::array_type n;
    out->get_nodes(n, idx);
    int nnum = 1;
    QuadraticTetVolMesh::Node::array_type::iterator niter = n.begin();
    while (niter != n.end()) {
      cout << "Node: " << nnum++ << "\n";
      Point p;
      out->get_center(p, *niter);
      ++niter;
      cout << p << "\n\n";
    }
  }




}

void TetVol2QuadraticTetVol::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


