//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
//    File   : SelectionSetTool.h
//    Author : Martin Cole
//    Date   : Mon Sep 18 10:02:37 2006

#if !defined(SelectionSetTool_h)
#define SelectionSetTool_h

#include <Core/Events/Tools/ViewToolInterface.h>
#include <Core/Events/Tools/BaseTool.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/GeomObj.h>
#include <vector>
#include <set>

namespace SCIRun {

using std::vector;
using std::set;

class RenderParams;

class SSTInterface
{
public:
  SSTInterface() {}
  virtual ~SSTInterface() {}
  virtual void set_selection_set_visible(bool) = 0;
  virtual set<unsigned int> &get_selection_set() = 0;
  virtual void set_selection_geom(GeomHandle) = 0;
  virtual void add_selection(unsigned int idx) = 0;
  virtual void remove_selection(unsigned int idx) = 0;
};

class SelectionSetTool : public BaseTool
{
public:
  enum selection_mode_e {
    NODES_E,
    EDGES_E,
    FACES_E,
    CELLS_E
  };
  SelectionSetTool(string name, SSTInterface*); 
  ~SelectionSetTool();

  propagation_state_e process_event(event_handle_t e);
  void delete_faces();
  void render_selection_set();

  void add_selection(unsigned int idx) {
    ssti_->add_selection(idx);
  }

  void remove_selection(unsigned int idx) {
    ssti_->remove_selection(idx);
  }
  
  void set_selection_mode(selection_mode_e m) { mode_ = m; }
  selection_mode_e get_selection_mode() { return mode_; }

private:
  selection_mode_e        mode_;
  FieldHandle             sel_fld_;
  unsigned int            sel_fld_id_;
  SSTInterface           *ssti_;
  RenderParams           *params_;
};

} // namespace SCIRun

#endif //FrameBufferPickTool_h
