
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

#include <Dataflow/Port.h>
#include <Multitask/ITC.h>

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
  namespace CoreDatatypes {
    class ColorImage;
    class DepthImage;
  }
}

namespace PSECommon {
namespace CommonDatatypes {

using PSECommon::Dataflow::IPort;
using PSECommon::Dataflow::OPort;
using PSECommon::Dataflow::Module;
using PSECommon::Dataflow::Connection;
using PSECommon::Comm::MessageBase;

using SCICore::Containers::clString;
using SCICore::Multitask::Mutex;
using SCICore::Multitask::Mailbox;
using SCICore::Multitask::CrowdMonitor;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::View;
using SCICore::CoreDatatypes::ColorImage;
using SCICore::CoreDatatypes::DepthImage;

class GeometryComm;

typedef int GeomID;

class GeometryIPort : public IPort {
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

class GeometryOPort : public OPort {
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

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:47  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:01  dav
// added back PSECommon .h files
//
// Revision 1.2  1999/04/27 23:18:35  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif /* SCI_project_GeometryPort_h */
