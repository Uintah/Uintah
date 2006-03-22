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

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/FieldPort.h>

namespace CardioWave {

using namespace SCIRun;

class DMDCreateDomain : public Module {
public:
  DMDCreateDomain(GuiContext*);

  virtual void execute();

};


DECLARE_MAKER(DMDCreateDomain)

DMDCreateDomain::DMDCreateDomain(GuiContext* ctx)
  : Module("DMDCreateDomain", ctx, Source, "DiscreteMultiDomain", "CardioWave")
{
}


void DMDCreateDomain::execute()
{
  FieldIPort* conductivityport = dynamic_cast<FieldIPort*>(get_input_port(0));
  if (conductivityport == 0)
  {
    error("Could not find Conductivity port");
    return;
  }

  FieldIPort* elementport = dynamic_cast<FieldIPort*>(get_input_port(1));
  if (elementport == 0)
  {
    error("Could not find Element Type port");
    return;
  }

  FieldHandle Conductivity;
  FieldHandle Elementtype;
  
  conductivityport->get(Conductivity);
  elementport->get(Elementtype);
  
  if (Elementtype.get_rep() == 0)
  {
    error("Element Type field is empty");
    return;
  }

  if (Conductivity.get_rep() == 0)
  {
    error("Conductivity field is empty");
    return;
  }

  BundleHandle output = scinew Bundle();
  output->setField("conductivity",Conductivity);
  output->setField("elementtype",Elementtype);

  BundleOPort* output_port = dynamic_cast<BundleOPort*>(get_output_port(0));
  if (output_port)
  {
    output_port->send(output);
  }
}

} // End namespace CardioWave


