/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

class GeometryIPort : public IPort {
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

  
class GeometryOPort : public OPort {
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
  virtual void synchronize();

  virtual void attach(Connection *c);
  virtual void detach(Connection *c, bool blocked);

  virtual bool have_data();
  virtual void resend(Connection*);

  int getNViewers();
  int getNViewWindows(int viewer);
  GeometryData* getData(int which_viewer, int which_viewwindow, int mask);
  void setView(int which_viewer, int which_viewwindow, View view);
};

} // End namespace SCIRun


#endif /* SCI_project_GeometryPort_h */
