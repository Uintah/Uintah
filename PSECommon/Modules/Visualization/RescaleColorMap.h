/************************************** 
CLASS 
   RescaleColormap

   A module that can scale the colormap values to fit the data
   or express a fixed data range.

GENERAL INFORMATION 
   RescaleColormap.h
   Written by: 

     Kurt Zimmerman
     Department of Computer Science 
     University of Utah 
     June 1999

     Copyright (C) 1998 SCI Group

KEYWORDS 
   Colormap, Transfer Function

DESCRIPTION 
   This module takes a color map and some data or vector field
   and scales the map to fit that field.

****************************************/ 

#include <Dataflow/Module.h>
#include <TclInterface/TCLvar.h>
#include <Containers/Array1.h>
#include <CommonDatatypes/ColorMapPort.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;

class RescaleColorMap : public Module {
  ColorMapOPort* omap;
  Array1<ScalarFieldIPort*> fieldports;
  ColorMapIPort* imap;


public:
  RescaleColorMap(const clString& id);
  RescaleColorMap(const RescaleColorMap&, int deep);
  virtual ~RescaleColorMap();
  virtual Module* clone(int deep);
  virtual void execute();
  virtual void connection(ConnectionMode mode, int which_port, int);

private:
  TCLint isFixed;
  TCLdouble min;
  TCLdouble max;
};

} // End namespace Modules
} // End namespace PSECommon
