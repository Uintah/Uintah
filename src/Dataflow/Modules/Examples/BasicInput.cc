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
//    File   : BasicInput.cc
//    Author : Martin Cole
//    Date   : Mon Sep 11 11:22:14 2006


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/SimplePort.h>
#include <iostream>

namespace SCIRun {
using namespace std;
using namespace SCIRun;

class BasicInput : public Module 
{
public:
  BasicInput(GuiContext*);
  virtual ~BasicInput();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(BasicInput)
BasicInput::BasicInput(GuiContext* ctx) : 
  Module("BasicInput", ctx, Source, "Examples", "SCIRun")
{
}

BasicInput::~BasicInput()
{
}

// Print Hello World! when we execute.
void
BasicInput::execute()
{
  FieldHandle field_handle;
  if (! get_input_handle("FieldIn", field_handle, true)) {
    error("BasicInput must have a SCIRun::Field as input to continue.");
  }
  cerr << "Hello World!" << endl;
}

void
BasicInput::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


