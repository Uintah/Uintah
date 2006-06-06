//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : SceneGraphEvent.h
//    Author : Martin Cole
//    Date   : Tue Jun  6 10:08:25 2006

#if !defined(SceneGraphEvent_h)
#define SceneGraphEvent_h

#include <Core/Events/BaseEvent.h>
#include <Core/Geom/GeomObj.h>
#include <string>


namespace SCIRun {

using namespace std;

class SceneGraphEvent : public BaseEvent 
{
public:
  SceneGraphEvent(GeomHandle o, string n, 
		  const string &target = "",
		  unsigned int time = 0);
  virtual ~SceneGraphEvent();
  
  virtual bool          is_scene_graph_event() { return true; }

  //! Accessors
  GeomHandle          get_geom_obj() const { return obj_; }
  string              get_geom_obj_name() const { return name_; }
  int                 get_scene_graph_id() const { return sg_id_; }

  //! Mutators
  void                  set_geom_obj(GeomHandle obj) { obj_ = obj; } 
  void                  set_geom_obj_name(GeomHandle obj) { obj_ = obj; } 
  void                  set_scene_graph_id(int id) { sg_id_ = id; }
private:
  GeomHandle          obj_;
  string              name_;
  int                 sg_id_;
};


} // namespace SCIRun

#endif // SceneGraphEvent_h
