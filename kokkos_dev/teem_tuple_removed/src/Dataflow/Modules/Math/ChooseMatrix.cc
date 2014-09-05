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
 *  ChooseMatrix.cc: Choose one input matrix to be passed downstream
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
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class PSECORESHARE ChooseMatrix : public Module {
private:
  GuiInt port_index_;
public:
  ChooseMatrix(GuiContext* ctx);
  virtual ~ChooseMatrix();
  virtual void execute();
};

DECLARE_MAKER(ChooseMatrix)
ChooseMatrix::ChooseMatrix(GuiContext* ctx)
  : Module("ChooseMatrix", ctx, Filter, "Math", "SCIRun"),
    port_index_(ctx->subVar("port-index"))
{
}

ChooseMatrix::~ChooseMatrix()
{
}

void
ChooseMatrix::execute()
{
  MatrixOPort *ofld = (MatrixOPort *)get_oport("Matrix");
  if (!ofld) {
    error("Unable to initialize oport 'Matrix'.");
    return;
  }

  port_range_type range = get_iports("Matrix");
  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;
  int idx=port_index_.get();
  if (idx<0) { error("Can't choose a negative port"); return; }
  while (pi != range.second && idx != 0) { ++pi ; idx--; }
  int port_number=pi->second;
  if (pi == range.second || ++pi == range.second) { 
    error("Selected port index out of range"); return; 
  }

  MatrixIPort *imatrix = (MatrixIPort *)get_iport(port_number);
  if (!imatrix) {
    error("Unable to initialize iport '" + to_string(port_number) + "'.");
    return;
  }
  MatrixHandle matrix;
  imatrix->get(matrix);
  ofld->send(matrix);
}

} // End namespace SCIRun

