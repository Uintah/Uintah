#ifndef PADFIELD_H
#define PADFIELD_H
/*
 * PadField.cc
 *
 * Simple interface to volume rendering stuff
 */


#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace Kurt {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;



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

} // namespace Modules
} // namespace Uintah

#endif
