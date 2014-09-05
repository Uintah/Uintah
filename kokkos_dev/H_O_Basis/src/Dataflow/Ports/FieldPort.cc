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


// FieldPort.cc
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group


#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

extern "C" {
IPort* make_FieldIPort(Module* module, const string& name) {
  return scinew SimpleIPort<FieldHandle>(module,name);
}
OPort* make_FieldOPort(Module* module, const string& name) {
  return scinew SimpleOPort<FieldHandle>(module,name);
}
}

template<> string SimpleIPort<FieldHandle>::port_type_("Field");
template<> string SimpleIPort<FieldHandle>::port_color_("yellow");


//! Specialization for field ports.
//! Field ports must only send const fields i.e. frozen fields.
template<>
void SimpleOPort<FieldHandle>::send(const FieldHandle& data)
{
  if (data.get_rep() && (! data->is_frozen()))
    data->freeze();

  do_send(data, SEND_NORMAL);
}

template<>
void SimpleOPort<FieldHandle>::send_intermediate(const FieldHandle& data)
{
  if (data.get_rep() && (! data->is_frozen()))
    data->freeze();

  do_send(data, SEND_INTERMEDIATE);
}

} // End namespace SCIRun

