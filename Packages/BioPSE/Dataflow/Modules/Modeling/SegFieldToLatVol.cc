/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
  // make sure the ports exist
  FieldIPort *ifp = (FieldIPort *)get_iport("SegField");
  FieldHandle ifieldH;
  if (!ifp) {
    error("Unable to initialize iport 'SegField'.");
    return;
  }
  FieldOPort *ofp = (FieldOPort *)get_oport("LatVolField");
  if (!ofp) {
    error("Unable to initialize oport 'LatVolField'.");
    return;
  }

  // make sure the input data exists
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
  LatVolField<int> *lvf = new LatVolField<int>(slvf->get_typed_mesh(), Field::CELL);
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

