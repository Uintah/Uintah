
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

#include <PSECore/share/share.h>
#include <PSECore/Dataflow/Port.h>

namespace SCICore {
  namespace Containers {
    class clString;
  }
  namespace GeomSpace {
    class GeomObj;
    class View;
  }
  namespace Thread {
      class CrowdMonitor;
      class Mutex;
      template<class T> class Mailbox;
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
using SCICore::Thread::Mutex;
using SCICore::Thread::Mailbox;
using SCICore::Thread::CrowdMonitor;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::View;
using SCICore::Datatypes::ColorImage;
using SCICore::Datatypes::DepthImage;

class GeometryComm;

typedef int GeomID;

class PSECORESHARE GeometryIPort : public IPort {
public:
    enum Protocol {
	Atomic=0x01
    };

protected:
    friend class GeometryOPort;
public:
    GeometryIPort(Module*, const clString& name, int protocol=GeometryIPort::Atomic);
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

#define GEOM_VIEW		1
#define GEOM_COLORBUFFER	2
#define GEOM_DEPTHBUFFER	4
#define GEOM_ALLDATA		(GEOM_VIEW|GEOM_COLORBUFFER|GEOM_DEPTHBUFFER)
#define GEOM_TRIANGLES		8

class PSECORESHARE GeometryOPort : public OPort {
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
    GeometryOPort(Module*, const clString& name, int protocol=GeometryIPort::Atomic);
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
    void setView(int which_roe, View view);
};

} // End namespace Datatypes
} // End namespace PSECore


#endif /* SCI_project_GeometryPort_h */
