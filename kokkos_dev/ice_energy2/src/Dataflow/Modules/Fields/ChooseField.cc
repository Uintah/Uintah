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
 *  ChooseField.cc: Choose one input field to be passed downstream
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/ChooseModule.h>
#include <Dataflow/Network/Ports/FieldPort.h>

namespace SCIRun {

class ChooseField : public ChooseModule< FieldHandle > {
public:
  ChooseField(GuiContext* ctx);

  virtual string get_names( std::vector<FieldHandle>& handles );

};

DECLARE_MAKER(ChooseField)
ChooseField::ChooseField(GuiContext* ctx)
  : ChooseModule< FieldHandle >("ChooseField", ctx, Filter,
				"FieldsOther", "SCIRun", "Field")
{
}

string
ChooseField::get_names( std::vector<FieldHandle>& handles )
{
  unsigned int idx = 0;
  string field_names("");
  string prop;
  while( idx < handles.size() ){
    if( handles[idx].get_rep() ){
      if( (handles[idx]->get_property("name", prop )) ) {
        field_names += prop + " ";
      } else { // use the port index to specify the name
        field_names += "--port_" + to_string(idx) + "-- ";
      }
    } else {
      field_names += "invalid_field ";
    }
    ++idx;
  }
  return field_names;
}
  


} // End namespace SCIRun

