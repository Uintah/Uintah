
/*
 *  Port.h: Classes for module ports
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Port_h
#define SCI_project_Port_h 1

#include <PSECore/share/share.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Comm/MessageBase.h>

namespace PSECore {
namespace Dataflow {

using SCICore::Containers::clString;
using SCICore::Containers::Array1;

class Connection;
class Module;

class PSECORESHARE Port {
    clString type_name;
    clString portname;
    clString colorname;
    int protocols;
    int u_proto;
protected:
    Module* module;
    int which_port;
    Array1<Connection*> connections;
    int xlight, ylight;
    enum PortState {
	Off,
	Resetting,
	Finishing,
	On
    };
    PortState portstate;
public:
    Port(Module*, const clString&, const clString&,
	 const clString&, int protocols);
    virtual ~Port();
    void set_port(int which_port);
    int using_protocol();
    int nconnections();
    Connection* connection(int);
    Module* get_module();
    int get_which_port();
    void set_which_port(int);
    virtual void attach(Connection*);
    virtual void detach(Connection*);
    virtual void reset()=0;
    virtual void finish()=0;
    clString get_typename();
    clString get_portname();
    clString get_colorname();
};

class PSECORESHARE IPort : public Port {
    void update_light();
protected:
    IPort(Module*, const clString&, const clString&,
	  const clString&, int protocols);
    void turn_on(PortState st=On);
    void turn_off();
public:
    virtual ~IPort();
};

class PSECORESHARE OPort : public Port {    
    void update_light();
protected:
    OPort(Module*, const clString&, const clString&,
	  const clString&, int protocols);
    void turn_on(PortState st=On);
    void turn_off();
public:
    virtual ~OPort();
    virtual int have_data()=0;
    virtual void resend(Connection*)=0;
};

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/26 23:59:08  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.2  1999/08/17 06:38:25  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:00  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 22:02:44  dav
// added back .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif /* SCI_project_Port_h */
