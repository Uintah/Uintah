
/*
 *  GeometryPort.h: Handle to the Geometry Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_GeometryPort_h
#define SCI_project_GeometryPort_h 1

#include <SCICore/share/share.h>
#include <PSECore/Dataflow/Port.h>
#include <SCICore/Multitask/ITC.h>

namespace SCICore {
  namespace Containers {
    class clString;
  }
  namespace GeomSpace {
    class GeomObj;
    class View;
  }
  namespace Multitask {
    class Mutex;
  }
  namespace Datatypes {
    class ColorImage;
    class DepthImage;
  }
}

namespace PSECore {
namespace Datatypes {

using PSECore::Dataflow::IPort;
using PSECore::Dataflow::OPort;
using PSECore::Dataflow::Module;
using PSECore::Dataflow::Connection;
using PSECore::Comm::MessageBase;

using SCICore::Containers::clString;
using SCICore::Multitask::Mutex;
using SCICore::Multitask::Mailbox;
using SCICore::Multitask::CrowdMonitor;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::View;
using SCICore::Datatypes::ColorImage;
using SCICore::Datatypes::DepthImage;

class GeometryComm;

typedef int GeomID;

class SCICORESHARE GeometryIPort : public IPort {
public:
    enum Protocol {
	Atomic=0x01
    };

protected:
    friend class GeometryOPort;
public:
    GeometryIPort(Module*, const clString& name, int protocol);
    virtual ~GeometryIPort();

    virtual void reset();
    virtual void finish();
};

struct GeometryData {
    ColorImage* colorbuffer;
    DepthImage* depthbuffer;

    View* view;
    int xres, yres;
    double znear, zfar;
    GeometryData();
    ~GeometryData();
    void Print();
};

#define GEOM_VIEW 1
#define GEOM_COLORBUFFER 2
#define GEOM_DEPTHBUFFER 4
#define GEOM_ALLDATA 7

class SCICORESHARE GeometryOPort : public OPort {
    GeometryIPort* in;
    int portid;
    GeomID serial;
    int dirty;
    int* busy_bit;
    Mutex* lock;

    GeometryComm* save_msgs;
    GeometryComm* save_msgs_tail;
    void save_msg(GeometryComm*);

    virtual void reset();
    virtual void finish();

    Mailbox<MessageBase*>* outbox;
    virtual void attach(Connection*);
public:
    GeometryOPort(Module*, const clString& name, int protocol);
    virtual ~GeometryOPort();

    GeomID addObj(GeomObj*, const clString& name, CrowdMonitor* lock=0);
    void delObj(GeomID, int del=1);
    void delAll();
    void flush();
    void flushViews();
    void flushViewsAndWait();

    void forward(GeometryComm* msg);

    int busy();

    virtual int have_data();
    virtual void resend(Connection*);

    int getNRoe();
    GeometryData* getData(int which_roe, int mask);
};

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:20  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:47  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:01  dav
// added back PSECore .h files
//
// Revision 1.2  1999/04/27 23:18:35  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif /* SCI_project_GeometryPort_h */
