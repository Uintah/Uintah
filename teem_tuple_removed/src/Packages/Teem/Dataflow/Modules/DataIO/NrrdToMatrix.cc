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
 *  NrrdToMatrix.cc: Converts Nrrd(s) to a SCIRun Matrix.  It may convert
 *                   it to a ColumnMatrix, DenseMatrix, or SparseMatrix
 *                   depending on which ports are connected.
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

namespace Teem {

using namespace SCIRun;

class PSECORESHARE NrrdToMatrix : public Module {
public:
  NrrdToMatrix(GuiContext*);

  virtual ~NrrdToMatrix();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(NrrdToMatrix)
NrrdToMatrix::NrrdToMatrix(GuiContext* ctx)
  : Module("NrrdToMatrix", ctx, Source, "DataIO", "Teem")
{
}

NrrdToMatrix::~NrrdToMatrix(){
}

void
 NrrdToMatrix::execute(){
}

void
 NrrdToMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


