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
 *  SegFieldToLatVol.cc: Erosion/dilation/open/close/absorption
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/BioPSE/Core/Datatypes/SegLatVolField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class SegFieldToLatVol : public Module
{
  GuiString latVolData_;
public:
  SegFieldToLatVol(GuiContext* ctx);
  virtual ~SegFieldToLatVol();
  virtual void execute();
};


DECLARE_MAKER(SegFieldToLatVol)

SegFieldToLatVol::SegFieldToLatVol(GuiContext* ctx)
  : Module("SegFieldToLatVol", ctx, Filter, "Modeling", "BioPSE"), 
    latVolData_(ctx->subVar("lat_vol_data"))
{
}



SegFieldToLatVol::~SegFieldToLatVol()
{
}

void
SegFieldToLatVol::execute()
{
  // Make sure the ports exist.
  FieldIPort *ifp = (FieldIPort *)get_iport("SegField");
  FieldOPort *ofp = (FieldOPort *)get_oport("LatVolField");

  // Make sure the input data exists.
  FieldHandle ifieldH;
  if (!ifp->get(ifieldH) || !ifieldH.get_rep()) {
    error("No input data");
    return;
  }
  SegLatVolField *slvf = dynamic_cast<SegLatVolField *>(ifieldH.get_rep());
  if (!slvf) {
    error("Input field was not a SegLatVolField");
    return;
  }

  // send a LatVolField with the user's chosen data as output
  LatVolField<int> *lvf = new LatVolField<int>(slvf->get_typed_mesh(), 0);
  if (latVolData_.get() == "componentMatl") {
    for (int i=0; i<lvf->fdata().dim1(); i++)
      for (int j=0; j<lvf->fdata().dim2(); j++)
	for (int k=0; k<lvf->fdata().dim3(); k++)
	  lvf->fdata()(i,j,k)=slvf->compMatl(slvf->fdata()(i,j,k));
  } else if (latVolData_.get() == "componentIdx") {
    for (int i=0; i<lvf->fdata().dim1(); i++)
      for (int j=0; j<lvf->fdata().dim2(); j++)
	for (int k=0; k<lvf->fdata().dim3(); k++)
	  lvf->fdata()(i,j,k)=slvf->fdata()(i,j,k);
  } else if (latVolData_.get() == "componentSize") {
    for (int i=0; i<lvf->fdata().dim1(); i++)
      for (int j=0; j<lvf->fdata().dim2(); j++)
	for (int k=0; k<lvf->fdata().dim3(); k++)
	  lvf->fdata()(i,j,k)=slvf->compSize(slvf->fdata()(i,j,k));
  } else {
    error("Unknown latVolData value");
    return;
  }
  FieldHandle lvfH(lvf);
  ofp->send(lvfH);
}
} // End namespace BioPSE

