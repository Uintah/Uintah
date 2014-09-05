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
 *  TriSurfPhaseFilter.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Ports/FieldPort.h>

#include <Core/Datatypes/Field.h>
#include <Packages/CardiacVis/Core/Algorithms/Fields/FieldsAlgo.h>

namespace CardiacVis {

using namespace SCIRun;

class TriSurfPhaseFilter : public Module {
  public:
    TriSurfPhaseFilter(GuiContext*);
    virtual void execute();   
};


DECLARE_MAKER(TriSurfPhaseFilter)
TriSurfPhaseFilter::TriSurfPhaseFilter(GuiContext* ctx)
  : Module("TriSurfPhaseFilter", ctx, Source, "FieldsCreate", "CardiacVis")
{
}

void TriSurfPhaseFilter::execute()
{
  FieldHandle input, output;
  FieldHandle phaseline, phasepoint;

  if (!(get_input_handle("PhaseField",input,true))) return;
  
  CardiacVis::FieldsAlgo algo(this);
  
  if(!(algo.TriSurfPhaseFilter(input,output,phaseline,phasepoint))) return;
 
  send_output_handle("PhaseField",output,true);
  send_output_handle("PhaseLine",phaseline,true);
  send_output_handle("PhasePoint",phasepoint,true);

}

} // End namespace CardiacVis


