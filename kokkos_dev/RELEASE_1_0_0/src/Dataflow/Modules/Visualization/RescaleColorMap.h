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
#include <Core/Containers/Array1.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/FieldPort.h>

namespace SCIRun {

class RescaleColorMap : public Module {
  ColorMapOPort* omap;
  Array1<FieldIPort*> fieldports;
  ColorMapIPort* imap;


public:

        // GROUP:  Constructors:
        ///////////////////////////
        // Constructs an instance of class RescaleColorMap
        // Constructor taking
        //    [in] id as an identifier
  RescaleColorMap(const clString& id);

        // GROUP:  Destructor:
        ///////////////////////////
        // Destructor
  virtual ~RescaleColorMap();

  virtual void get_minmax(FieldHandle f);

  template<class F> void dispatch_minmax(F *f) 
  { success_ = field_minmax(*f, minmax_); }

        // GROUP:  Access functions:
        ///////////////////////////
        // execute() - execution scheduled by scheduler
  virtual void execute();
        ///////////////////////////

  virtual void connection(ConnectionMode mode, int which_port, int);

protected:
  bool success_;

private:
  GuiInt isFixed;
  GuiDouble min;
  GuiDouble max;
  pair<double,double> minmax_;
};

} // End namespace SCIRun

#endif
