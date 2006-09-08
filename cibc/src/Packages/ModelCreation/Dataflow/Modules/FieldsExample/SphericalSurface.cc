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

#include <Dataflow/Network/Module.h>

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <Packages/ModelCreation/Core/Fields/ExampleFields.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>

namespace ModelCreation {

using namespace SCIRun;

class SphericalSurface : public Module {
public:
  SphericalSurface(GuiContext*);
  virtual void execute();
  
private:
  GuiDouble guidiscretization_;
  GuiDouble guiradius_;

};


DECLARE_MAKER(SphericalSurface)
SphericalSurface::SphericalSurface(GuiContext* ctx)
  : Module("SphericalSurface", ctx, Source, "FieldsExample", "ModelCreation"),
    guidiscretization_(get_ctx()->subVar("discretization")),
    guiradius_(get_ctx()->subVar("radius"))    
{
}

void
SphericalSurface::execute()
{
  SCIRunAlgo::ConverterAlgo mc(this);
  
  MatrixHandle disc;
  if (!get_input_handle("Discretization", disc, false))
  {
    mc.DoubleToMatrix(guidiscretization_.get(), disc);
  }
  
  MatrixHandle radius;
  if (!get_input_handle("Radius", radius, false))
  {
    mc.DoubleToMatrix(guiradius_.get(), radius);
  }

  ExampleFields sphere_algo(this);
  SCIRunAlgo::FieldsAlgo algo(this);
  FieldHandle ofield;
  if (sphere_algo.SphericalSurface(ofield, disc))
  {
    Transform TF;
    double r;
    mc.MatrixToDouble(radius,r);
    TF.pre_scale(Vector(r,r,r));
    algo.TransformField(ofield,ofield,TF);
    send_output_handle("Field", ofield);
  }
}


} // End namespace ModelCreation


