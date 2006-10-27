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


#include <Packages/CardioWaveInterface/Core/TissueModels/RegularBundle.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace CardioWave {

using namespace SCIRun;

class CreateRegularBundle : public Module {
public:
  CreateRegularBundle(GuiContext*);
  virtual void execute();

private:
  GuiInt cellsx_;
  GuiInt cellsy_;
  GuiInt cellsz_;
  GuiInt elemsx_ics_;
  GuiInt elemsx_ecs_;
  GuiInt elemsy_ics_;
  GuiInt elemsy_ecs_;
  GuiInt elemsz_;
  GuiInt bath_start_;
  GuiInt bath_end_;

  GuiDouble cell_length_;
  GuiDouble cell_crosssection_;
  GuiDouble ics_vol_frac_;

  GuiInt connection_x_;
  GuiInt connection_y_;
  GuiInt disable_center_;
};


DECLARE_MAKER(CreateRegularBundle)
CreateRegularBundle::CreateRegularBundle(GuiContext* ctx)
  : Module("CreateRegularBundle", ctx, Source, "TissueModel", "CardioWave"),
  cellsx_(ctx->subVar("cells-x")),
  cellsy_(ctx->subVar("cells-y")),
  cellsz_(ctx->subVar("cells-z")),
  elemsx_ics_(ctx->subVar("elems-x-ics")),
  elemsx_ecs_(ctx->subVar("elems-x-ecs")),
  elemsy_ics_(ctx->subVar("elems-y-ics")),
  elemsy_ecs_(ctx->subVar("elems-y-ecs")),
  elemsz_(ctx->subVar("elems-z")),
  bath_start_(ctx->subVar("bath-start")),
  bath_end_(ctx->subVar("bath-start")),
  cell_length_(ctx->subVar("cell-length")),
  cell_crosssection_(ctx->subVar("cell-crosssection")),
  ics_vol_frac_(ctx->subVar("ics-vol-frac")),
  connection_x_(ctx->subVar("lateral-connection-x")),
  connection_y_(ctx->subVar("lateral-connection-y")),
  disable_center_(ctx->subVar("disable-center"))
{
}

void CreateRegularBundle::execute()
{
  FieldHandle Output;

  TissueModel_RegularBundle Tissue(this, cellsx_.get(),
				   cellsy_.get(),cellsz_.get());
  Tissue.set_numelems_x_ics(elemsx_ics_.get());
  Tissue.set_numelems_y_ics(elemsy_ics_.get());
  Tissue.set_numelems_x_ecs(elemsx_ecs_.get());
  Tissue.set_numelems_y_ecs(elemsy_ecs_.get());
  Tissue.set_numelems_z(elemsz_.get());
  Tissue.set_numelems_bath_start(bath_start_.get());
  Tissue.set_numelems_bath_end(bath_end_.get());
  Tissue.set_connection_x(connection_x_.get());
  Tissue.set_connection_y(connection_y_.get());

  Tissue.set_cell_length(cell_length_.get());
  Tissue.set_cell_crosssection(cell_crosssection_.get());
  Tissue.set_ics_vol_frac(ics_vol_frac_.get());
  Tissue.set_disable_center(disable_center_.get());
  
  if(!(Tissue.create_mesh(Output))) return;

  send_output_handle("TissueModel",Output,false);
}

} // End namespace CardioWave


