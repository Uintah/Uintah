/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Dataflow/share/share.h>
#include <Dataflow/Network/Port.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/Thread/Mailbox.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Light.h>
#include <string>
#include <vector>
#include <list>

namespace SCIRun {

using namespace std;

class View;
class CrowdMonitor;
class Mutex;
class ColorImage;
class DepthImage;
class GeometryComm;

typedef int GeomID;
typedef short LightID;

class PSECORESHARE GeometryIPort : public IPort {
public:
  GeometryIPort(Module*, const string& name);
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
#ifdef HAVE_COLLAB_VIS
  // CollabVis code begin
  double modelview[16];
  double projection[16];
  int viewport[4];
  
  ~GeometryData() {}
  // CollabVis code end
#else
  ~GeometryData();
#endif
  
  GeometryData();
  void Print();
};


#define GEOM_VIEW		1
#define GEOM_COLORBUFFER	2
#define GEOM_DEPTHBUFFER	4
#define GEOM_ALLDATA		(GEOM_VIEW|GEOM_COLORBUFFER|GEOM_DEPTHBUFFER)
#define GEOM_TRIANGLES		8
// CollabVis code begin
#define GEOM_MATRICES           16
// CollabVis code end

  
class PSECORESHARE GeometryOPort : public OPort {
private:

  GeomID serial_;
  LightID lserial_;
  bool dirty_;

  std::list<GeometryComm* > saved_msgs_;

  vector<int> portid_;
  vector<Mailbox<MessageBase*>*> outbox_;

  void save_msg(GeometryComm*);

public:
  GeometryOPort(Module*, const string& name);
  virtual ~GeometryOPort();

  GeomID addObj(GeomHandle, const string& name, CrowdMonitor* lock=0);
  LightID addLight(LightHandle, const string& name, CrowdMonitor* lock=0);
  void delObj(GeomID, int del=1);
  void delLight(LightID, int del = 1);
  void delAll();
  void flush();
  void flushViews();
  void flushViewsAndWait();

  void forward(GeometryComm* msg);
  bool direct_forward(GeometryComm* msg);

  virtual void reset();
  virtual void finish();

  virtual void attach(Connection*);
  virtual void detach(Connection*);

  virtual bool have_data();
  virtual void resend(Connection*);

  int getNViewers();
  int getNViewWindows(int viewer);
  GeometryData* getData(int which_viewer, int which_viewwindow, int mask);
  void setView(int which_viewer, int which_viewwindow, View view);
};

} // End namespace SCIRun


#endif /* SCI_project_GeometryPort_h */
