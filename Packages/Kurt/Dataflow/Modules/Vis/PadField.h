#ifndef PADFIELD_H
#define PADFIELD_H
/*
 * PadField.cc
 *
 * Simple interface to volume rendering stuff
 */


#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>

namespace Kurt {
using namespace SCIRun;



class PadField : public Module {

public:
  PadField( const clString& id);

  virtual ~PadField();
  virtual void execute();
  //void tcl_command( TCLArgs&, void* );

private:
  ScalarFieldIPort *inscalarfield;
  ScalarFieldOPort *outscalarfield;


  TCLint pad_mode;
  TCLint xpad;
  TCLint ypad;
  TCLint zpad;
};
} // End namespace Kurt


#endif
