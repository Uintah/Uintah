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

//    File   : TendBmat.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <teem/ten.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <cstdio>

namespace SCITeem {

using namespace SCIRun;

class TendBmat : public Module {
public:
  TendBmat(SCIRun::GuiContext *ctx);
  virtual ~TendBmat();
  virtual void execute();
  virtual void presave();

private:
  // Create a memory for a new nrrd, that is arranged 3 x n;
  bool extract_gradients(vector<double> &);

  GuiString    gradient_list_;
};

DECLARE_MAKER(TendBmat)

TendBmat::TendBmat(SCIRun::GuiContext *ctx) : 
  Module("TendBmat", ctx, Filter, "Tend", "Teem"), 
  gradient_list_(get_ctx()->subVar("gradient_list"), "")
{
}


TendBmat::~TendBmat()
{
}


// Create a memory for a new nrrd, that is arranged 3 x n;
bool
TendBmat::extract_gradients(vector<double> &d)
{
  get_gui()->execute(get_id() + " update_text"); // make gradient_list current
  istringstream str(gradient_list_.get().c_str());
  while (true)
  {
    double tmp;
    str >> tmp;
    if (!str.eof() && !str.fail()) {
      d.push_back(tmp);
    }
    else {
      break;
    }
  }
  if (d.size() % 3 != 0) {
    error("Error: Number of input values must be divisible by 3");
    return false;
  }
  return true;
}


void 
TendBmat::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  NrrdIPort *inrrd = (NrrdIPort *)get_iport("nin");
  bool we_own_the_data;
  vector<double> *mat=0;
  Nrrd *nin;
  if (inrrd->get(nrrd_handle) && nrrd_handle.get_rep())
  {
    we_own_the_data = false;
    nin = nrrd_handle->nrrd_;
  }
  else
  {
    we_own_the_data = true;
    mat = new vector<double>;
    if (! extract_gradients(*mat))
    {
      error("Please adjust your input in the gui to represent a 3 x N set.");
      return;
    }
    nin = nrrdNew();
    size_t size[2] = {3, (*mat).size()/3};
    nrrdWrap_nva(nin, &(*mat)[0], nrrdTypeDouble, 2, size);
  }

  Nrrd *nout = nrrdNew();
  if (tenBMatrixCalc(nout, nin)) {
    char *err = biffGetDone(TEN);
    error(string("Error making aniso volume: ") + err);
    free(err);
    return;
  }
  
  nout->axis[0].label = airStrdup("tensor components");
  nout->axis[1].label = airStrdup("n");

  if (we_own_the_data)
  {
    nrrdNix(nin);
    delete mat;
  }

  NrrdDataHandle ntmp(scinew NrrdData(nout));

  send_output_handle("nout", ntmp);
}


void
TendBmat::presave()
{
  get_gui()->execute(get_id() + " update_text"); // make gradient_list current
}


} // End namespace SCITeem
