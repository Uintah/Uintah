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

#include <Core/Algorithms/Math/MathAlgo.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class CuthillMcKee : public Module {
public:
  CuthillMcKee(GuiContext*);
  virtual void execute();

};


DECLARE_MAKER(CuthillMcKee)
CuthillMcKee::CuthillMcKee(GuiContext* ctx)
  : Module("CuthillMcKee", ctx, Source, "Math", "ModelCreation")
{
}


void CuthillMcKee::execute()
{
  MatrixHandle Mat, Mapping, InverseMapping;
  if(!(get_input_handle("Matrix",Mat,true))) return;

  if (inputs_changed_ || !oport_cached("Matrix") || !oport_cached("Mapping") || !oport_cached("InverseMapping"))
  {
    SCIRunAlgo::MathAlgo algo(this);
    if(!(algo.CuthillmcKee(Mat,Mat,InverseMapping,true))) return;
    
    Mapping = InverseMapping->transpose();
    
    send_output_handle("Matrix",Mat,true);
    send_output_handle("Mapping",Mapping,true);
    send_output_handle("InverseMapping",InverseMapping,true);
  }
}


} // End namespace ModelCreation


