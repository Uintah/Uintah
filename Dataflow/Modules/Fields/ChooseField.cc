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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class ChooseField : public Module {

public:
  ChooseField(GuiContext* ctx);
  virtual ~ChooseField();
  virtual void execute();

private:
  GuiInt gUseFirstValid_;
  GuiInt gPortIndex_;

  FieldHandle fHandle_;
};

DECLARE_MAKER(ChooseField)
ChooseField::ChooseField(GuiContext* ctx)
  : Module("ChooseField", ctx, Filter, "FieldsOther", "SCIRun"),
    gUseFirstValid_(ctx->subVar("use-first-valid"), 1),
    gPortIndex_(ctx->subVar("port-index"), 0),
    fHandle_(0)
{
}

ChooseField::~ChooseField()
{
}

void
ChooseField::execute()
{
  pre_execute();

  std::vector<FieldHandle> fHandles;

  if( !getDynamicIHandle( "Field", fHandles, false ) ) return;

  // Check to see if any values have changed via a matrix or user.
  if( !fHandle_.get_rep() ||
      gUseFirstValid_.changed( true ) ||

      (gUseFirstValid_.get() == 1 ) ||      
      (gUseFirstValid_.get() == 0 &&  gPortIndex_.changed( true )) ||

      execute_error_ ) {

    execute_error_ = false;
  
    // use the first valid field
    if (gUseFirstValid_.get()) {

      unsigned int idx = 0;
      while( idx < fHandles.size() && !fHandles[idx].get_rep() ) idx++;

      if( idx < fHandles.size() && fHandles[idx].get_rep() ) {
	fHandle_ = fHandles[idx];

	gPortIndex_.set( idx );

	reset_vars();

      } else {
	error("Did not find any valid fields.");

	execute_error_ = true;
	return;
      }

    } else {
      // use the index specified
      int idx = gPortIndex_.get();

      if ( 0 <= idx && idx < (int) fHandles.size() ) {
	if( fHandles[idx].get_rep() ) {
	  fHandle_ = fHandles[idx];

	} else {
	  error( "Port " + to_string(idx) + " did not contain a valid field.");
	  execute_error_ = true;
	  return;
	}

      } else {
	error("Selected port index out of range.");
	execute_error_ = true;
	return;
      }
    }
  }

  // Send the data downstream
  setOHandle( "Field",  fHandle_, true );

  post_execute();
}

} // End namespace SCIRun

