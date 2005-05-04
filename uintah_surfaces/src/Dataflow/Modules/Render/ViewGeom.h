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


#ifndef SCI_Render_ViewGeom_h
#define SCI_Render_ViewGeom_h 1

/*
 *  ViewGeom.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Geom/IndexedGroup.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geometry/BBox.h>
#include <Core/Persistent/Persistent.h>
#include <Dataflow/Network/Module.h>


namespace SCIRun {

class CrowdMonitor;
class GeometryComm;

/* this is basicaly a indexed group that also has some simple message
 * stuff
 */	

class GeomViewerPort: public GeomIndexedGroup {
  GeometryComm* msg_head;
  GeometryComm* msg_tail;

  int portno;
    
public:
  friend class Viewer;
  GeomViewerPort(int);
  virtual ~GeomViewerPort();

  GeometryComm* getHead(void) { return msg_head; }
};

/*
 * for items in a scene - has name (for roes to lookup)
 * a lock and a geomobj (to call)
 */

class GeomViewerItem: public GeomContainer {
private:  
  string name_;
  CrowdMonitor *crowd_lock_;

  GeomViewerItem();
  static Persistent *maker();

public:
  friend class ViewWindow;
  GeomViewerItem(GeomHandle,const string&, CrowdMonitor* lock);

  virtual GeomObj* clone();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
    
  string& getString(void) { return name_;}
};

} // End namespace SCIRun


#endif
