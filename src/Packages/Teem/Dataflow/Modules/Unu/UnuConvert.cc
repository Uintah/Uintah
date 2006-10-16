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


/*
 *  UnuConvert: Convert between C types
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {
using namespace SCIRun;

class UnuConvert : public Module {
  GuiInt type_;
  int last_type_;
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  UnuConvert(SCIRun::GuiContext *ctx);
  virtual ~UnuConvert();
  virtual void execute();
};

} //end namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(UnuConvert)

UnuConvert::UnuConvert(SCIRun::GuiContext *ctx)
  : Module("UnuConvert", ctx, Filter, "UnuAtoM", "Teem"), 
  type_(get_ctx()->subVar("type")),
  last_type_(0), last_generation_(-1), last_nrrdH_(0)
{
}


UnuConvert::~UnuConvert()
{
}


void 
UnuConvert::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrdH;
  if (!get_input_handle("Nrrd", nrrdH)) return;

  reset_vars();

  int type = type_.get();
  if (last_generation_ == nrrdH->generation &&
      last_type_ == type &&
      last_nrrdH_.get_rep())
  {
    send_output_handle("Nrrd", last_nrrdH_, true);
    return;
  }

  last_generation_ = nrrdH->generation;
  last_type_ = type;

  Nrrd *nin = nrrdH->nrrd_;
  Nrrd *nout = nrrdNew();
  string name, cname;
  get_nrrd_compile_type(type, name, cname);
  remark("New type is " + name + ".");

  if (nrrdConvert(nout, nin, type)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble resampling: ") + err);
    msg_stream_ << "  input Nrrd: nin->dim="<<nin->dim<<"\n";
    free(err);
  }

  last_nrrdH_ = scinew NrrdData(nout);
  send_output_handle("Nrrd", last_nrrdH_, true);
}

