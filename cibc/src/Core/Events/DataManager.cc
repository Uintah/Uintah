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
//    File   : DataManager.cc
//    Author : Martin Cole
//    Date   : Tue Sep 12 10:35:09 2006


#include <Core/Events/DataManager.h>
#include <Core/Events/SelectionTargetEvent.h>
#include <Core/Events/SceneGraphEvent.h>
#include <Core/Algorithms/Visualization/RenderField.h>

namespace SCIRun {

unsigned int DataManager::next_id_ = 1;

class DMCommandTool : public CommandTool
{
public:

  DMCommandTool(string name, DataManager *dm) :
    CommandTool(name),
    dm_(dm)
  {}

  virtual propagation_state_e issue_command(const string &cmmd, 
					    unsigned int time) 
  {
    if (cmmd == "selection field modified") {
      cerr << "selection field changed... reshow" << endl;
      unsigned int fid = dm_->get_selection_target();
      if (fid != 0) {
	dm_->show_field(fid);
      }
      return STOP_E;
    }
    return CONTINUE_E;
  }
private:
  DataManager           *dm_;
};

DataManager::DataManager() :
  ThrottledRunnable(10.),  // does not need to update very quickly
  lock_("DataManager lock"),
  tm_("DataManager tool manager"),
  events_(0)
{ 
  nrrds_.clear();
  mats_.clear();
  fields_.clear();

  events_ = EventManager::register_event_messages("DataManager");

  tool_handle_t dmct = new DMCommandTool("DataManager Command Tool", this);
  tm_.add_tool(dmct, 0);
}

DataManager::~DataManager() 
{
  nrrds_.clear();
  mats_.clear();
  fields_.clear();
}

bool
DataManager::iterate()
{
  event_handle_t ev;
  while (events_ && events_->tryReceive(ev)) {
    // Tools will set up the appropriate rendering state.
    QuitEvent *qe = dynamic_cast<QuitEvent*>(ev.get_rep());
    if (qe) {
      // this is the terminate signal, so return.
      return false;
    }
    tm_.propagate_event(ev);
  }

  return true;
}

FieldHandle  
DataManager::get_field(unsigned int id)
{
  return fields_[id];
}

MatrixHandle 
DataManager::get_matrix(unsigned int id)
{
  return mats_[id];
}

NrrdDataHandle   
DataManager::get_nrrd(unsigned int id)
{
  return nrrds_[id];
}

unsigned int 
DataManager::load_field(string fname)
{
  FieldHandle fld;
  Piostream* stream = auto_istream(fname);
  if (!stream) {
    cerr << "Couldn't open file: "<< fname << endl;
    return 0;
  }
  Pio(*stream, fld);

  lock_.lock();
  fields_[next_id_++] = fld;
  lock_.unlock();
  return next_id_ - 1;
}

unsigned int 
DataManager::load_matrix(string fname)
{
  return 0;
}

unsigned int 
DataManager::load_nrrd(string fname)
{
  return 0;
}


void 
DataManager::selection_target_changed(unsigned int fid)
{
  cerr << "the selection target id is: " << fid << endl;
  sel_fid_ = fid;
  FieldHandle fld = get_field(sel_fid_);

  SelectionTargetEvent *t = new SelectionTargetEvent();
  t->set_selection_target(fld);
  t->set_selection_id(fid);

  event_handle_t event = t;
  EventManager::add_event(event);
}  

// sends Scene Graph event with the rendered geometry.
bool         
DataManager::show_field(unsigned int fld_id)
{
  FieldHandle fld_handle = fields_[fld_id];
  static RenderParams p;
  p.defaults();
  p.faces_transparency_ = true;
  p.do_nodes_ = true;
  p.do_faces_ = true;
  p.do_edges_ = true;
  p.do_text_ = false;
  
  if (! render_field(fld_handle, p)) {
    cerr << "Error: render_field failed." << endl;
    return false;
  }

  // for now make this field the selection target as well.
  SelectionTargetEvent *ste = new SelectionTargetEvent();
  event_handle_t event = ste;
  ste->set_selection_target(fld_handle);
  EventManager::add_event(event);
   
  ostringstream str;
  str << "-" << fld_id;

  string fname;
  if (! fld_handle->get_property("name", fname)) {
    fname = "Field";
  }
  fname = fname + str.str();

  if (p.do_nodes_) 
  {
    GeomHandle gmat = scinew GeomMaterial(p.renderer_->node_switch_, 
					  p.def_material_);
    GeomHandle geom = scinew GeomSwitch(new GeomColorMap(gmat, 
							 p.color_map_));
    const char *name = p.nodes_transparency_ ? "Transparent Nodes" : "Nodes";
    SceneGraphEvent* sge = new SceneGraphEvent(geom, fname + name);
    event_handle_t event = sge;
    EventManager::add_event(event);
  }

  if (p.do_edges_) 
  { 
    GeomHandle gmat = scinew GeomMaterial(p.renderer_->edge_switch_, 
					  p.def_material_);
    GeomHandle geom = scinew GeomSwitch(new GeomColorMap(gmat, 
							 p.color_map_));
    const char *name = p.edges_transparency_ ? "Transparent Edges" : "Edges";
    SceneGraphEvent* sge = new SceneGraphEvent(geom, fname + name);
    event_handle_t event = sge;
    EventManager::add_event(event);
  }

  if (p.do_faces_)
  {
    GeomHandle gmat = scinew GeomMaterial(p.renderer_->face_switch_, 
					  p.def_material_);
    GeomHandle geom = scinew GeomSwitch(new GeomColorMap(gmat, 
							    p.color_map_));
    const char *name = p.faces_transparency_ ? "Transparent Faces" : "Faces";
    SceneGraphEvent* sge = new SceneGraphEvent(geom, fname + name);
    event_handle_t event = sge;
    EventManager::add_event(event);
  }
  if (p.do_text_) 
  {
    GeomHandle gmat = scinew GeomMaterial(p.text_geometry_, p.text_material_);
    GeomHandle geom = scinew GeomSwitch(new GeomColorMap(gmat, p.color_map_));
    const char *name = p.text_backface_cull_ ? "Culled Text Data":"Text Data";
    SceneGraphEvent* sge = new SceneGraphEvent(geom, fname + name);
    event_handle_t event = sge;
    EventManager::add_event(event);
  }

  return true;  
}

}

