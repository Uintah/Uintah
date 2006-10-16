//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : VolumeSlice.cc
//    Author : McKay Davis
//    Date   : Fri Oct 13 15:35:55 2006

#include <StandAlone/Apps/Painter/Painter.h>
#include <sci_comp_warn_fixes.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <typeinfo>
#include <iostream>
#include <sci_gl.h>
#include <sci_algorithm.h>
#include <Core/Bundle/Bundle.h>
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

VolumeSlice::VolumeSlice(Painter *painter,
                              NrrdVolume *volume, 
                              SliceWindow *window,
                              Point &p, Vector &n) :
  painter_(painter),
  volume_(volume),
  window_(window),
  nrrd_dirty_(true),
  tex_dirty_(false),
  pos_(),
  xdir_(),
  ydir_(),
  plane_(p,n),
  texture_(0)
{
}


void
VolumeSlice::set_tex_dirty() {
  tex_dirty_ = true;
}


unsigned int 
VolumeSlice::axis() {
  ASSERT(volume_);
  return max_vector_magnitude_index(volume_->vector_to_index(plane_.normal()));
}


void
VolumeSlice::bind()
{
  if (tex_dirty_) { 
    texture_ = 0; 
  }


  if (!texture_.get_rep()) {
    vector<int> index = volume_->world_to_index(plane_.project(Point(0,0,0)));
    unsigned int ax = axis();
    unsigned int slice = index[ax];
    volume_->mutex_->lock();
    if (slice>=0 && slice < volume_->nrrd_handle_->nrrd_->axis[ax].size)
      texture_ = scinew ColorMappedNrrdTextureObj(volume_->nrrd_handle_, 
                                                  ax,
                                                  slice, 
                                                  slice);
    volume_->mutex_->unlock();

    tex_dirty_ = true;
  }


  if (texture_.get_rep() && tex_dirty_) {
    texture_->set_label(volume_->label_);
    texture_->set_clut_minmax(volume_->clut_min_, volume_->clut_max_);
    ColorMapHandle cmap = painter_->get_colormap(volume_->colormap_);
    texture_->set_colormap(cmap);
    GeomColorMappedNrrdTextureObj *slice = 
      new GeomColorMappedNrrdTextureObj(texture_);
    GeomSkinnerVarSwitch *gswitch = 
      new GeomSkinnerVarSwitch(slice, volume_->visible_);
    window_->get_geom_group()->addObj(gswitch, axis());
  }


  tex_dirty_ = false;

  return;
}



void
VolumeSlice::draw()
{
  if (nrrd_dirty_) {
    set_coords();
    nrrd_dirty_ = false;
    tex_dirty_ = true;
  }

  float a = volume_->opacity_;
  glColor4f(a,a,a,a);
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  CHECK_OPENGL_ERROR();  

  bind();
  if (texture_.get_rep())
    texture_->draw_quad(&pos_, &xdir_, &ydir_);
}



void
VolumeSlice::set_coords() {  
  vector<int> sindex = volume_->world_to_index(plane_.project(Point(0,0,0)));
  unsigned int ax = axis();

  Point origin = volume_->min(ax, 0);
  pos_ = volume_->min(ax, sindex[ax]);
  vector<int> index(volume_->nrrd_handle_->nrrd_->dim,0);

  int primary = (ax == 1) ? 2 : 1;
  index[primary] = volume_->nrrd_handle_->nrrd_->axis[primary].size;
  xdir_ = volume_->index_to_world(index) - origin;
  index[primary] = 0;

  int secondary = (ax == 3) ? 2 : 3;
  index[secondary] = volume_->nrrd_handle_->nrrd_->axis[secondary].size;
  ydir_ = volume_->index_to_world(index) - origin;
}

}
