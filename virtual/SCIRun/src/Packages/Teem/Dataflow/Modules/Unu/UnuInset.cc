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
 *  UnuInset.cc Replace a sub-region with a different nrrd. This is functionally
 *  the opposite of "crop".
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuInset : public Module {
public:
  UnuInset(GuiContext*);

  virtual ~UnuInset();

  virtual void execute();

  GuiString       mins_;
};


DECLARE_MAKER(UnuInset)
UnuInset::UnuInset(GuiContext* ctx)
  : Module("UnuInset", ctx, Source, "UnuAtoM", "Teem"),
    mins_(get_ctx()->subVar("mins"), "0")
{
}


UnuInset::~UnuInset()
{
}


void
UnuInset::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("InputNrrd", nrrd_handle)) return;

  NrrdDataHandle sub_handle;
  if (!get_input_handle("SubRegionNrrd", sub_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *sub = sub_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  // Determine the number of mins given
  string mins = mins_.get();
  unsigned int minsLen = 0;
  char ch;
  int i=0, start=0;
  bool inword = false;
  while (i < (int)mins.length())
  {
    ch = mins[i];
    if(isspace(ch))
    {
      if (inword)
      {
	minsLen++;
	inword = false;
      }
    }
    else if (i == (int)mins.length()-1)
    {
      minsLen++;
      inword = false;
    }
    else
    {
      if(!inword) 
	inword = true;
    }
    i++;
  }

  if (minsLen != nin->dim)
  {
    error("min coords " + to_string(minsLen) +
          " != nrrd dim " + to_string(nin->dim));
    return;
  }

  size_t *min = new size_t[nin->dim];
  // Size/samples
  i=0, start=0;
  int which = 0, end=0, counter=0;
  inword = false;
  while (i < (int)mins.length())
  {
    ch = mins[i];
    if (isspace(ch))
    {
      if (inword)
      {
	end = i;
	min[counter] = (atoi(mins.substr(start,end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
    }
    else if (i == (int)mins.length()-1)
    {
      if (!inword)
      {
	start = i;
      }
      end = i+1;
      min[counter] = (atoi(mins.substr(start,end-start).c_str()));
      which++;
      counter++;
      inword = false;
    }
    else
    {
      if(!inword)
      {
	start = i;
	inword = true;
      }
    }
    i++;
  }

  if (nrrdInset(nout, nin, sub, min))
  {
    char *err = biffGetDone(NRRD);
    error(string("Error Insetting nrrd: ") + err);
    free(err);
  }

  delete min;

  NrrdDataHandle out(scinew NrrdData(nout));

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  // Copy the axis kinds
  for (unsigned int i=0; i<nin->dim; i++)
  {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  send_output_handle("OutputNrrd", out);
}


} // End namespace Teem


