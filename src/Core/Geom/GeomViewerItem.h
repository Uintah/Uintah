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


#ifndef SCI_Core_Geom_GeomViewerItem_h
#define SCI_Core_Geom_GeomViewerItem_h 1

/*
 *  GeomViewerItem.h:
 *
 *   Department of Computer Science
 *   University of Utah
 *   Date: November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Core/Geom/GeomContainer.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geometry/BBox.h>
#include <Core/Persistent/Persistent.h>
#include <Dataflow/Network/Module.h>

#include <Core/Geom/share.h>

namespace SCIRun {

class CrowdMonitor;

class SCISHARE GeomViewerItem: public GeomContainer {
private:  
  string name_;
  CrowdMonitor *crowd_lock_;

  GeomViewerItem();
  static Persistent *maker();

public:
  GeomViewerItem(GeomHandle,const string&, CrowdMonitor* lock);

  CrowdMonitor *crowd_lock() { return crowd_lock_; }

  virtual GeomObj* clone();

  virtual void draw(DrawInfoOpenGL*, Material*, double time);
  virtual void fbpick_draw(DrawInfoOpenGL*, Material*, double time);
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
    
  string& getString(void) { return name_;}
};

} // End namespace SCIRun


#endif
