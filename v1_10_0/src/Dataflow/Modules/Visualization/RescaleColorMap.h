#ifndef SCIRun_RescaleColorMap_H
#define SCIRun_RescaleColorMap_H
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

#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/FieldPort.h>

namespace SCIRun {

class RescaleColorMap : public Module {
public:

  //! Constructor taking [in] id as an identifier
  RescaleColorMap(GuiContext* ctx);

  virtual ~RescaleColorMap();
  virtual void execute();

protected:
  bool success_;

private:
  GuiInt isFixed;
  GuiDouble min;
  GuiDouble max;
  GuiInt makeSymmetric;
  pair<double,double> minmax_;
};

} // End namespace SCIRun

#endif
