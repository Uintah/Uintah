
#include <PSECore/Dataflow/Module.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Containers/Array1.h>
#include <PSECore/CommonDatatypes/ColorMapPort.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;

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
  Array1<ScalarFieldIPort*> fieldports;
  ColorMapIPort* imap;


public:

        // GROUP:  Constructors:
        ///////////////////////////
        //
        // Constructs an instance of class RescaleColorMap
        //
        // Constructor taking
        //    [in] id as an identifier
        //
  RescaleColorMap(const clString& id);

        // GROUP:  Destructor:
        ///////////////////////////
        // Destructor
  virtual ~RescaleColorMap();


        // GROUP:  Access functions:
        ///////////////////////////
        //
        // execute() - execution scheduled by scheduler
  virtual void execute();
        ///////////////////////////
  virtual void connection(ConnectionMode mode, int which_port, int);

private:
  TCLint isFixed;
  TCLdouble min;
  TCLdouble max;
};

} // End namespace Modules
} // End namespace PSECommon
