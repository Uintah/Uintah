//static char *id="@(#) $Id$";

/* ElapsedTime.cc
   Display elapsed wall clock time

   Philip Sutton
   October 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#include <SCICore/Util/NotFinished.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <PSECore/Dataflow/Module.h>

#include <unistd.h>
#include <limits.h>
#include <values.h>

#include "Clock.h"

namespace Phil {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace SCICore::TclInterface;

class ElapsedTime : public Module {
public:
  ElapsedTime(const clString& id);
  virtual ~ElapsedTime();
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);
protected:
private:
  TCLint min, sec, hsec;
  TCLint stop;
  int _min, _sec, _hsec;

}; // class ElapsedTime

extern "C" Module* make_ElapsedTime(const clString& id){
  return new ElapsedTime(id);
}

ElapsedTime::ElapsedTime(const clString& id)
  : Module("ElapsedTime",id,Filter), min("min",id,this), sec("sec",id,this),
  hsec("hsec",id,this), stop("stop",id,this) {
  _min = 0;
  _sec = 0;
  _hsec = 0;
  min.set(0);
  sec.set(0);
  hsec.set(0);
  stop.set(0);
}

ElapsedTime::~ElapsedTime() {
}

void 
ElapsedTime::execute() {
  _min = 0;
  _sec = 0;
  _hsec = 0;
  min.set(0);
  sec.set(0);
  hsec.set(0);
  stop.set(0);
  stop.reset();
  iotimer_t t0, t1;
  iotimer_t lasttime, diff;

  init_clock();

  // nap length
  long int n = (long int) (.001 * (float)CLK_TCK);

  t0 = read_time();
  lasttime = t0;
  for( ;; ) {

    // sleep until 0.01s have gone by
    sginap(n);
    t1 = read_time();
    diff = t1 - lasttime;
    if( (diff*(cycleval*1.0)*1E-12) < 0.01 ) {
      continue;
    }
    lasttime = t1;

    // update the hundredths and seconds
    _hsec += (diff*(cycleval*1.0)*1E-12) * 100;
    while( _hsec >= 100 ) {
      _sec++;
      _hsec -= 100;
    }

    // update the seconds and minutes
    while( _sec >= 60 ) {
      _min++;
      _sec -= 60;
    }

    // set everything in the GUI
    hsec.set(_hsec);
    sec.set(_sec);
    min.set(_min);

    TCL::execute( id + " update_elapsed_time");
    stop.reset();
    if( stop.get() == 1 )
      break;
  }
  stop.set(0);
  stop.reset();
}

void 
ElapsedTime::tcl_command(TCLArgs& args, void* userdata) {
  if( args[1] == "getVars" ) {
    char var[80];
    sprintf(var,"%d",_min);
    clString svar = clString( var );
    clString result = svar;

    sprintf(var,"%d",_sec);
    svar = clString( var );
    result += clString( " " + svar );

    sprintf(var,"%d",_hsec);
    svar = clString( var );
    result += clString( " " + svar );

    args.result( result );
  } else {
    // message not for us - propagate up
    Module::tcl_command( args, userdata );
  }
}

} // end namespace Modules
} // end namespace Phil

//
// $Log$
// Revision 1.3  2000/03/17 09:28:10  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  2000/02/04 22:16:48  psutton
// fixed ID problem
//
// Revision 1.1  2000/02/04 21:16:47  psutton
// initial revision
//
//
