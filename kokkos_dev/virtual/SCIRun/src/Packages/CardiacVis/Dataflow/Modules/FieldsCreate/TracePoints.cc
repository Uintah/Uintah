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

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Packages/CardiacVis/Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace CardiacVis {

using namespace SCIRun;

class TracePoints : public Module {
public:
  TracePoints(GuiContext*);
  virtual void execute();

private:
  GuiDouble  guitol_;
  
  FieldHandle output_;    
  int field_generation_;
  int time_generation_;

};


DECLARE_MAKER(TracePoints)
TracePoints::TracePoints(GuiContext* ctx)
  : Module("TracePoints", ctx, Source, "FieldsCreate", "CardiacVis"),
  guitol_(ctx->subVar("tol")),
  output_(0),
  field_generation_(-1),
  time_generation_(-1)
{
}

void
TracePoints::execute()
{
  FieldHandle Input,Output;
  MatrixHandle Time;
  
  if(!(get_input_handle("Singularity Points",Input,true))) return;
  if(!(get_input_handle("Time",Time,true))) return;

  // Make sure we synchronous
  if ((field_generation_ != Input->generation)&&(time_generation_  != Time->generation))
  {
    field_generation_ = Input->generation;
    time_generation_  = Time->generation;

    SCIRunAlgo::ConverterAlgo calgo(this);
    CardiacVis::FieldsAlgo algo(this);
    
    double time;
    double tol;
    
    tol = guitol_.get();
    calgo.MatrixToDouble(Time,time);
    algo.TracePoints(Input,output_,output_,time,tol);

    Output = output_;
    send_output_handle("Singularity Lines",Output,false);
  }
}

} // End namespace CardiacVis


