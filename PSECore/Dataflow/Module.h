
/*
 *  Module.h: Base class for modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Module_h
#define SCI_project_Module_h 1

#ifdef _WIN32
#pragma warning(disable:4355 4786)
#endif

#include <PSECore/share/share.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Util/Timer.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Mailbox.h>
#include <SCICore/Geom/Pickable.h>

namespace SCICore {
  namespace Geometry {
    class Vector;
  }
  namespace GeomSpace {
    class GeomPick;
  }
}

namespace PSECore {
  namespace Comm {
    class MessageBase;
  }
}
namespace PSECommon {
  namespace Modules {
    class Roe;
  }
}

namespace PSECore {
namespace Dataflow {

using SCICore::TclInterface::TCL;
using SCICore::TclInterface::TCLArgs;
using SCICore::TclInterface::TCLstring;
using SCICore::GeomSpace::GeomPick;
using SCICore::GeomSpace::BState;
using SCICore::GeomSpace::Pickable;
using SCICore::Geometry::Vector;
using SCICore::Containers::clString;
using SCICore::Containers::Array1;

using PSECore::Comm::MessageBase;
using PSECommon::Modules::Roe;

class Connection;
class Network;
class NetworkEditor;
class OPort;
class IPort;

class PSECORESHARE Module : public TCL, public Pickable {
    /*
     * This exists to trip people up that still have clone and
     * copy constructors in the modules - they shoudl be removed.
     */
    Module(const Module&);

    // Added by Mohamed Dekhil for the CSAFE project
  TCLstring notes ;
public:
    enum State {
	NeedData,
	JustStarted,
	Executing,
	Completed
    };
public:
    int show_status;

    friend class ModuleHelper;
    virtual void do_execute();
    virtual void execute()=0;

    State state;
    Array1<OPort*> oports;
    Array1<IPort*> iports;
    ModuleHelper* helper;
    int have_own_dispatch;

    double progress;
    CPUTimer timer;
    //WallClockTimer timer;

public:
    enum ConnectionMode {
	Connected,
	Disconnected
    };
    enum SchedClass {
	Sink,
	Source,
	Filter,
	Iterator,
	SalmonSpecial
    };
    Module(const clString& name, const clString& id, SchedClass);
    virtual ~Module();

    /*
     * This exists to trip people up that still have clone and
     * copy constructors in the modules - they shoudl be removed.
     */
    virtual int clone(int deep);

    SCICore::Thread::Mailbox<MessageBase*> mailbox;

    inline State get_state(){ return state;}
    inline double get_progress(){ return progress;}

    void get_position(int& x, int& y);

    // Callbacks
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void geom_pick(GeomPick*, void*, int);
    virtual void geom_pick(GeomPick*, void*);
    virtual void geom_pick(GeomPick*, Roe*, int, const BState& bs);
    virtual void geom_release(GeomPick*, int, const BState& bs);
    virtual void geom_release(GeomPick*, void*, int);
    virtual void geom_release(GeomPick*, void*);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, void*, int);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, void*);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, 
			    int, const BState&);
    virtual void geom_moved(GeomPick*, int, double, const Vector&, 
			    const BState&, int);
    virtual void widget_moved(int);
    virtual void widget_moved2(int last, void *) {
	widget_moved(last);
    }
    // Port manipulations
    void add_iport(IPort*);
    void add_oport(OPort*);
    void remove_iport(int);
    void remove_oport(int);
    void rename_iport(int, const clString&);
    void rename_oport(int, const clString&);
    virtual void reconfigure_iports();
    virtual void reconfigure_oports();

    // Used by Module subclasses
    void error(const clString&);
    void update_state(State);
    void update_progress(double);
    void update_progress(double, Timer &);
    void update_progress(int, int);
    void update_progress(int, int, Timer &);
    void want_to_execute();

    // User Interface information
    NetworkEditor* netedit;
    Network* network;
    clString name;
    clString categoryName;   
    clString packageName;    
    clString moduleName;  
    int abort_flag;
public:
    int niports();
    int noports();
    IPort* iport(int);
    OPort* oport(int);
    void multisend(OPort*, OPort* =0);
    void set_context(NetworkEditor*, Network*);

    int need_execute;
    SchedClass sched_class;
    // virtual int should_execute();

    clString id;
    int handle; 	// mm-skeleton and remote share the same handle
    bool remote;        // mm-is this a remote module?  not used.
    bool skeleton;	// mm-is this a skeleton module?
    bool isSkeleton()	{ return skeleton; }
    void tcl_command(TCLArgs&, void*);

};

typedef Module* (*ModuleMaker)(const clString& id);

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.9  1999/11/17 23:14:30  moulding
// added a warning pragma to quiet the vc++ compiler (debug name truncation)
//
// Revision 1.8  1999/11/10 23:24:31  dmw
// added show_status flag to module interface -- if you turn it off, the timer and port lights won't update
//
// Revision 1.7  1999/08/30 18:47:52  kuzimmer
// Modified so that dataflow scripts can be read and written properly
//
// Revision 1.6  1999/08/28 17:54:29  sparker
// Integrated new Thread library
//
// Revision 1.5  1999/08/26 23:59:07  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.4  1999/08/18 20:20:18  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.3  1999/08/17 06:38:22  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:58  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 22:02:42  dav
// added back .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif /* SCI_project_Module_h */
