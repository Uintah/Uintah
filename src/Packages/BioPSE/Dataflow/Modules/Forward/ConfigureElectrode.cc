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
 *  ConfigureElectrode: Insert an electrode into a finite element mesh
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;
typedef CurveMesh<CrvLinearLgn<Point> >                     CMesh;
typedef CrvLinearLgn<double>                                CBasis;
typedef GenericField<CMesh, CBasis, vector<double> >        CField;  
class ConfigureElectrode : public Module {
  GuiString active_;
  GuiDouble voltage_;
public:
  ConfigureElectrode(GuiContext *context);
  virtual ~ConfigureElectrode();
  virtual void execute();
};

DECLARE_MAKER(ConfigureElectrode)

ConfigureElectrode::ConfigureElectrode(GuiContext *context)
  : Module("ConfigureElectrode", context, Filter, "Forward", "BioPSE"),
    active_(context->subVar("active")), voltage_(context->subVar("voltage"))
{
}

ConfigureElectrode::~ConfigureElectrode()
{
}


void
ConfigureElectrode::execute()
{
  FieldHandle ielecH;
  if (!get_input_handle("Electrode", ielecH)) return;

  CField *elecFld = dynamic_cast<CField*>(ielecH.get_rep());
  if (!elecFld) {
    error("Input electrode wasn't a CurveField<double>.");
    return;
  }

  double voltage = voltage_.get();
  string active = active_.get();

  CMesh::Node::iterator ni;
  elecFld->get_typed_mesh()->begin(ni);
  elecFld->fdata()[*ni]=voltage;

  elecFld->set_property("active_side", active, false);

  FieldHandle efield(elecFld);
  send_output_handle("Electrode", efield);
}

} // End namespace BioPSE
