/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  Painter.cc
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */
#include <sci_comp_warn_fixes.h>
#include <StandAlone/Apps/Seg3D/Painter.h>
#include <StandAlone/Apps/Seg3D/PointerToolSelectorTool.h>
#include <StandAlone/Apps/Seg3D/KeyToolSelectorTool.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <typeinfo>
#include <iostream>
#include <sci_gl.h>
#include <sci_algorithm.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Skinner/GeomSkinnerVarSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/CleanupManager.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Geom/TextRenderer.h>
#include <Core/Geom/FontManager.h>
#include <Core/Geom/GeomColorMappedNrrdTextureObj.h>
#include <Core/Skinner/Variables.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/SceneGraphEvent.h>
#include <Core/Util/FileUtils.h>
#include <Core/Algorithms/Visualization/NrrdTextureBuilderAlgo.h>


namespace SCIRun {



Painter::Painter(Skinner::Variables *variables, VarContext* ctx) :
  Parent(variables),
  cur_window_(0),
  tm_("Painter"),
  pointer_pos_(),
  windows_(),
  volumes_(),
  current_volume_(0),
  colormaps_(1, ColorMap::create_greyscale()),
  volume_lock_("Volume"),
  volume_texture_(0),
  filter_volume_(0),
  abort_filter_(false),
  status_(variables, "Painter::status", "")
{
  tm_.add_tool(new PointerToolSelectorTool(this), 50);
  tm_.add_tool(new KeyToolSelectorTool(this), 51);

  InitializeSignalCatcherTargets(0);
  Skinner::Signal *signal = new Skinner::Signal("LoadColorMap1D",
                                                this, get_vars());
  string path = findFileInPath("Rainbow.cmap", sci_getenv("SKINNER_PATH"));
  if (!path.empty()) {
    Skinner::Var<string> filename(get_vars(), "filename");
    filename = path + "Rainbow.cmap";
    LoadColorMap1D(signal);
  }
}

Painter::~Painter()
{
}




void
Painter::redraw_all()
{
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    (*i)->mark_redraw();
  }
}



ColorMapHandle
Painter::get_colormap(int id)
{
  ASSERT(colormaps_.size());
  id = Clamp(id, 0, colormaps_.size()-1);
  return colormaps_[id];
}
    


void
Painter::extract_all_window_slices() {
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    (*i)->extract_slices();
  }

  if (volume_texture_.get_rep()) {
    ShowVolumeRendering(0);
  }

}



void
Painter::get_data_from_layer_buttons()  
{
  for (unsigned int i = 0; i < layer_buttons_.size(); ++i) {
    layer_buttons_[i]->update_from_gui(0);
  }
}

void
Painter::rebuild_layer_buttons()  
{
  get_data_from_layer_buttons();
  unsigned int bpos = 0;  
  for (int i = volumes_.size()-1; i >= 0 ; --i) {
    build_layer_button(bpos, volumes_[i]);
  }
  for (; bpos < layer_buttons_.size(); ++bpos) {
    layer_buttons_[bpos]->visible_ = false;
    layer_buttons_[bpos]->volume_ = 0;
  }
  EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
}

void
Painter::build_layer_button(unsigned int &bpos, NrrdVolumeHandle &volume)  
{
  LayerButton *button = layer_buttons_[bpos];
  button->visible_ = true;
  button->layer_name_ = volume->name_;
  button->volume_ = volume;

  unsigned int level = 0;
  NrrdVolumeHandle parent = volume;
  while (parent->parent_.get_rep()) {
    level++;
    parent = parent->parent_;
  }
  button->indent_ = 20*level+5;
  if (volume->children_.empty()) {
    button->expand_width_ = 0;
  } else {
    button->expand_width_ = 20;
  }
    

  if (volume == current_volume_) {
    button->background_color_ = Skinner::Color(0.6, 0.6, 1.0, 0.75);
  } else {
    button->background_color_ = Skinner::Color(0.0, 0.0, 0.0, 0.0);
  }

  bpos++;  
  if (volume->expand_) {
    for (int i = volume->children_.size()-1; i >= 0 ; --i) {
      build_layer_button(bpos, volume->children_[i]);
    }
  }
}



void
Painter::build_volume_list(NrrdVolumes &volumes, NrrdVolumeHandle &volume)
{
  NrrdVolumes &children = volume.get_rep() ? volume->children_ : volumes_;
  for (unsigned int i = 0; i < children.size(); ++i) {
    volumes.push_back(children[i]);
    build_volume_list(volumes, children[i]);
  }
}

void
Painter::move_layer_up(NrrdVolumeHandle &layer)
{
  if (!layer.get_rep()) return;
  NrrdVolumes &volumes = 
    layer->parent_.get_rep() ? layer->parent_->children_ : volumes_;
  unsigned int i = 0;
  while (i < volumes.size() && volumes[i] != layer) ++i;
  ASSERT(volumes[i] == layer);
  if (i == volumes.size()-1) return;

  NrrdVolumeHandle temp = volumes[i+1];
  volumes[i+1] = volumes[i];
  volumes[i] = temp;
  
  extract_all_window_slices();
  rebuild_layer_buttons();
}

void
Painter::move_layer_down(NrrdVolumeHandle &layer)
{
  if (!layer.get_rep()) return;  
  NrrdVolumes &volumes = 
    layer->parent_.get_rep() ? layer->parent_->children_ : volumes_;
  unsigned int i = 0;
  while (i < volumes.size() && volumes[i] != layer) ++i;
  ASSERT(volumes[i] == layer);
  if (i == 0) return;

  NrrdVolumeHandle temp = volumes[i-1];
  volumes[i-1] = volumes[i];
  volumes[i] = temp;
  
  extract_all_window_slices();
  rebuild_layer_buttons();
}


void
Painter::opacity_down()
{
  if (current_volume_.get_rep()) {
    current_volume_->opacity_ = 
      Clamp(current_volume_->opacity_-0.05, 0.0, 1.0);
    redraw_all();
  }
}

void
Painter::opacity_up()
{
  if (current_volume_.get_rep()) {
    current_volume_->opacity_ = 
      Clamp(current_volume_->opacity_+0.05, 0.0, 1.0);
    redraw_all();
  }
}



void
Painter::cur_layer_down()
{
  if (volumes_.size() < 2 || current_volume_ == volumes_[0]) 
    return;
  for (unsigned int i = 1; i < volumes_.size(); ++i)
    if (current_volume_ == volumes_[i]) {
      current_volume_ = volumes_[i-1];
      rebuild_layer_buttons();
      return;
    }
}


void
Painter::cur_layer_up()
{
  if (volumes_.size() < 2 || current_volume_ == volumes_.back()) 
    return;
  for (unsigned int i = 0; i < volumes_.size()-1; ++i)
    if (current_volume_ == volumes_[i]) {
      current_volume_ = volumes_[i+1];
      rebuild_layer_buttons();
      return;
    }
}


void
Painter::reset_clut()
{
  if (current_volume_.get_rep()) {
    current_volume_->clut_min_ = current_volume_->data_min_;
    current_volume_->clut_max_ = current_volume_->data_max_;
    set_all_slices_tex_dirty();
    redraw_all();
  }
}



void
Painter::set_probe() {
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    (*i)->set_probe();
  }
  redraw_all();
}


void
Painter::set_all_slices_tex_dirty() {
  volume_lock_.lock();
  for (NrrdVolumes::iterator i = volumes_.begin(); i != volumes_.end(); ++i) {
    for (VolumeSlices_t::iterator j = (*i)->all_slices_.begin();
         j != (*i)->all_slices_.end(); ++j) {
      (*j)->set_tex_dirty();
    }
  }
  volume_lock_.unlock();
}


NrrdVolumeHandle
Painter::find_volume_by_name(const string &name) {
  for (unsigned int i = 0; i < volumes_.size(); ++i)
    if (volumes_[i]->name_ == name) 
      return volumes_[i];
  return 0;
}

string
Painter::unique_layer_name(string base) {
  string::size_type pos = base.find_last_not_of(" 0123456789");
  base = base.substr(0, pos+1);
  int i = 0;
  string name = base + " "+to_string(++i);
  while (find_volume_by_name(name).get_rep())
    name = base + " "+to_string(++i);
  return name;
}




void
Painter::create_undo_volume() {
  return;
}

void
Painter::undo_volume() {
#if 0
  if (!undo_volume_) return;
  NrrdVolume *vol = volume_map_[undo_volume_->name_];
  if (!vol) return;
  vol->nrrd_handle_ = undo_volume_->nrrd_handle_;
  vol->nrrd_handle_.detach();
  extract_all_window_slices();
  redraw_all();
#endif
  //  delete undo_volume_;
  //  undo_volume_ = 0;
}



Skinner::Drawable *
Painter::maker(Skinner::Variables *vars) 
{
  return new Painter(vars, 0);
}


NrrdVolumeHandle
Painter::make_layer(string name, NrrdDataHandle &nrrdh, unsigned int mask) {
  volume_lock_.lock();
  NrrdVolume *vol = new NrrdVolume(this, unique_layer_name(name), nrrdh, mask);
  volumes_.push_back(vol);
  rebuild_layer_buttons();
  current_volume_ = vol;
  volume_lock_.unlock();  
  return vol;
}


NrrdVolumeHandle
Painter::copy_current_layer(string suff) {
  if (!current_volume_.get_rep()) return 0;
  NrrdDataHandle nrrdh = current_volume_->nrrd_handle_;
  nrrdh.detach(); // Copies the layer memory to the new layer
  return make_layer(current_volume_->name_+suff,nrrdh,current_volume_->label_);
}


bool
Painter::merge_layer(NrrdVolumeHandle &vol1) {
  NrrdVolumeHandle vol2 = 0;
  NrrdVolumeHandle parent = vol1.get_rep() ? vol1->parent_ : 0;
  NrrdVolumes &volumes =  parent.get_rep() ? parent->children_ : volumes_;
  for (int i = 0; i < volumes.size()-1; ++i)
    if (volumes[i+1] == vol1)
      vol2 = volumes[i];
  if (!vol1.get_rep() || !vol2.get_rep()) return STOP_E;

#if 0

  
  if (volname == volume_order_.begin()) return STOP_E;
  NrrdVolume *vol1 = volume_map_[*volname];
  NrrdVolume *vol2 = volume_map_[*(--volname)];
    

  NrrdData *nout = new NrrdData();
  NrrdIter *ni1 = nrrdIterNew();
  NrrdIter *ni2 = nrrdIterNew();
    
  nrrdIterSetNrrd(ni1, vol1->nrrd_handle_->nrrd_);
  nrrdIterSetNrrd(ni2, vol2->nrrd_handle_->nrrd_);
  
  if (nrrdArithIterBinaryOp(nout->nrrd_, nrrdBinaryOpMultiply, ni1, ni2)) {
    char *err = biffGetDone(NRRD);
    string errstr = (err ? err : "");
    free(err);
    throw errstr;
  }

  nrrdIterNix(ni1);
  nrrdIterNix(ni2);

  nrrdKeyValueCopy(nout->nrrd_,  vol1->nrrd_handle_->nrrd_);
  nrrdKeyValueCopy(nout->nrrd_,  vol2->nrrd_handle_->nrrd_);
  
  vol1->nrrd_handle_->nrrd_ = nout->nrrd_;
  vol2->keep_ = 0;
  
  current_volume_ = vol1;
#endif
  return true;
}
  
 
} // end namespace SCIRun
