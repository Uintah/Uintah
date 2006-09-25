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
//    File   : SelectionSetTool.cc
//    Author : Martin Cole
//    Date   : Mon Sep 18 10:04:12 2006

#include <Core/Events/Tools/SelectionSetTool.h>
#include <Core/Events/SelectionTargetEvent.h>
#include <Core/Events/EventManager.h>
#include <Core/Algorithms/Visualization/RenderField.h>
#include <Core/Algorithms/Fields/ClipAtIndeces.h>


#include <sstream>

namespace SCIRun {

SelectionSetTool::SelectionSetTool(string name, SSTInterface *ssti) :
  BaseTool(name),
  mode_(FACES_E),
  sel_fld_(0),
  sel_fld_id_(0),
  ssti_(ssti),
  params_(new RenderParams())
{
  params_->defaults();
  params_->def_material_ = new Material(Color(0.9, 0.1, 0.1));
  params_->text_material_ = new Material(Color(0.9, 0.1, 0.1));
}

SelectionSetTool::~SelectionSetTool() 
{}

BaseTool::propagation_state_e 
SelectionSetTool::process_event(event_handle_t e) 
{
  SelectionTargetEvent *st = 
    dynamic_cast<SelectionTargetEvent*>(e.get_rep());
  if (st) {
    sel_fld_ = st->get_selection_target();
    sel_fld_id_ = st->get_selection_id();
    return STOP_E;
  }
  return CONTINUE_E;
}

void 
SelectionSetTool::delete_faces() 
{
  typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;

  if (! sel_fld_.get_rep()) return;
  sel_fld_->lock.lock();
  // turn this call into a general algorithm, but for now assume trisurf.
  MeshHandle mb = sel_fld_->mesh();
  TSMesh *tsm = dynamic_cast<TSMesh *>(mb.get_rep());
  if (!tsm) {
    cerr << "Error:: not a TriSurf in SelectionSetTool::delete_faces" 
	 << endl;
  }
  set<unsigned int> &sfaces = ssti_->get_selection_set();


  vector<int> faces;
  set<unsigned int>::iterator si = sfaces.begin();
  while (si != sfaces.end()) {
    faces.push_back(*si++);
  }

  bool altered = false;
  // remove last index first.
  sort(faces.begin(), faces.end());
  vector<int>::reverse_iterator iter  = faces.rbegin();
  while (iter != faces.rend()) {
    int face = *iter++;
    altered |= tsm->remove_face(face);
    cout << "removed face " << face << endl;
  }
    

  //clear the selection set.
  sfaces.clear();
  ssti_->set_selection_set_visible(false);
  sel_fld_->lock.unlock();

  //notify the data manager that this model has changed.
  CommandEvent *c = new CommandEvent();
  c->set_command("selection field modified");
  event_handle_t event = c;
  EventManager::add_event(event);
}

void 
SelectionSetTool::render_selection_set() 
{
  //create a new field with the selection items in it;
  if (! sel_fld_.get_rep()) { return; }
  if (! ssti_->get_selection_set().size()) { 
    ssti_->set_selection_geom(GeomHandle(0));   
    return; 
  }

  params_->defaults(); // won't reset the colors we already set

  FieldHandle sel_vis;
  switch (mode_) {

  case NODES_E:
    params_->do_nodes_ = true;
    params_->do_text_ = true;
    params_->ns_ = 1.0;
    sel_vis = clip_nodes(sel_fld_, ssti_->get_selection_set());
    break;
  case EDGES_E:
    params_->do_edges_ = true;
    break;
  default:
  case FACES_E:
    params_->do_faces_ = true;
    sel_vis = clip_faces(sel_fld_, ssti_->get_selection_set());
    break;
  case CELLS_E:

    break;
  };
      

  if (!sel_vis.get_rep() || !render_field(sel_vis, *params_)) {
    cerr << "Error: render_field failed." << endl;
    return;
  }

  GeomHandle gmat;
  GeomHandle geom;
  string name;
  switch (mode_) {

  case NODES_E:
    {
      gmat = scinew GeomMaterial(params_->renderer_->node_switch_, 
				 params_->def_material_);
      geom = scinew GeomSwitch(scinew GeomColorMap(gmat, 
						   params_->color_map_));
      name = params_->nodes_transparency_ ? "Transparent Nodes" : "Nodes";

//       gmat = scinew GeomMaterial(params_->text_geometry_, params_->text_material_);
//       geom = scinew GeomSwitch(new GeomColorMap(gmat, params_->color_map_));
//       name = params_->text_backface_cull_ ? "Culled Text Data":"Text Data";
    }
    break;
  case EDGES_E:
    {
      gmat = scinew GeomMaterial(params_->renderer_->edge_switch_, 
				 params_->def_material_);
      geom = scinew GeomSwitch(scinew GeomColorMap(gmat, 
						   params_->color_map_));
      name = params_->edges_transparency_ ? "Transparent Edges" : "Edges";
    }
    break;
  default:
  case FACES_E:
    {
      gmat = scinew GeomMaterial(params_->renderer_->face_switch_, 
				 params_->def_material_);
      geom = scinew GeomSwitch(scinew GeomColorMap(gmat, 
						   params_->color_map_));
      name = params_->faces_transparency_ ? "Transparent Faces" : "Faces";
    }
    break;
  };
  ssti_->set_selection_geom(geom);    
}

} // namespace SCIRun
