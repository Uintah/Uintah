//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : ChangeTetVolScalars.cc
//    Author : Martin Cole
//    Date   : Mon Sep 11 11:22:14 2006


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/SimplePort.h>

#include <Core/Basis/TetLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>

#include <Dataflow/GuiInterface/GuiVar.h>

#include <iostream>

namespace SCIRun {
using namespace std;
using namespace SCIRun;

class ChangeTetVolScalars : public Module 
{
public:
  ChangeTetVolScalars(GuiContext*);
  virtual ~ChangeTetVolScalars();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
private:
  GuiDouble       newval_;
};


DECLARE_MAKER(ChangeTetVolScalars)
ChangeTetVolScalars::ChangeTetVolScalars(GuiContext* ctx) : 
  Module("ChangeTetVolScalars", ctx, Source, "Examples", "SCIRun"),
  newval_(get_ctx()->subVar("newval"), 1.0)
{
}

ChangeTetVolScalars::~ChangeTetVolScalars()
{
}

void
ChangeTetVolScalars::execute()
{
  typedef TetVolMesh<TetLinearLgn<Point> >    TVMesh;
  typedef TetLinearLgn<double>                DataBasis;
  typedef GenericField<TVMesh, DataBasis, vector<double> > TVField;  

  FieldHandle field_handle;
  if (! get_input_handle("InField", field_handle, true)) {
    error("ChangeTetVolScalars must have a SCIRun::Field as input to continue.");
    return;
  }

  // Must detach since we will be altering the input field.
  field_handle.detach();

  TVField *in = dynamic_cast<TVField*>(field_handle.get_rep());

  if (in == 0) {
    error("This Module only accepts Linear TetVol Fields with double data.");
    return;
  }
    
  newval_.reset();

  TVField::mesh_handle_type mh = in->get_typed_mesh();
  TVMesh::Node::iterator iter;
  TVMesh::Node::iterator end;
  mh->begin(iter);
  mh->end(end);

  while (iter != end) {
    TVMesh::Node::index_type ni = *iter;
    Point node;
    mh->get_center(node, ni);
    TVField::value_type val;
    in->value(val, ni);
    cerr << "at point: " << node << "the input value is: " << val << endl;

    // Set the value to be the value from the gui;
    in->set_value(newval_.get(), ni);
    ++iter;
  }

  send_output_handle("OutField", field_handle);
}

void
ChangeTetVolScalars::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


