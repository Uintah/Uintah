
#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Ports/ColorMapPort.h>

namespace SCIRun {


/**************************************
CLASS
   RescaleColorMap

   A module that can scale the colormap values to fit the data
   or express a fixed data range.

GENERAL INFORMATION
   RescaleColorMap.h
   Written by:

     Kurt Zimmerman<br>
     Department of Computer Science<br>
     University of Utah<br>
     June 1999

     Copyright (C) 1998 SCI Group

KEYWORDS
   ColorMap, Transfer Function

DESCRIPTION
   This module takes a color map and some data or vector field
   and scales the map to fit that field.

****************************************/

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


        // GROUP:  Access functions:
        ///////////////////////////
        // execute() - execution scheduled by scheduler
  virtual void execute();
        ///////////////////////////
  virtual void connection(ConnectionMode mode, int which_port, int);

private:
  GuiInt isFixed;
  GuiDouble min;
  GuiDouble max;
};

} // End namespace SCIRun
