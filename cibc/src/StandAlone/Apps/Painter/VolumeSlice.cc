//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : VolumeSlice.cc
//    Author : McKay Davis
//    Date   : Fri Oct 13 15:35:55 2006

#include <StandAlone/Apps/Painter/Painter.h>
#include <StandAlone/Apps/Painter/VolumeOps.h>
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

VolumeSlice::VolumeSlice(NrrdVolume *volume,
                         const Plane &plane,
                         NrrdDataHandle nrrd,
                         unsigned int label) :
  lock("Volume Slice"),
  ref_cnt(0),
  volume_(volume),
  nrrd_handle_(nrrd),
  outline_(0),
  texture_(0),
  geom_texture_(0),
  plane_(plane),
  label_(label),
  tex_dirty_(true),
  pos_(),
  xdir_(),
  ydir_()
{
  vector<int> sindex = volume_->world_to_index(plane_.project(Point(0,0,0)));
  unsigned int ax = axis();

  // Lower-Left origin corner for slice quad,
  // project into plane of window, to ensure volumes with different sample
  // spacings share same plane in view space.
  pos_ = plane_.project(volume_->min(ax, sindex[ax]));
  vector<int> index(volume_->nrrd_handle_->nrrd_->dim,0);

  Point origin = volume_->min(ax, 0);
  int primary = (ax == 1) ? 2 : 1;
  index[primary] = volume_->nrrd_handle_->nrrd_->axis[primary].size;
  xdir_ = volume_->index_to_world(index) - origin;
  index[primary] = 0;

  int secondary = (ax == 3) ? 2 : 3;
  index[secondary] = volume_->nrrd_handle_->nrrd_->axis[secondary].size;
  ydir_ = volume_->index_to_world(index) - origin;


  if (!nrrd_handle_.get_rep()) {
    extract_nrrd_slice_from_volume();
  } else {
    cerr << "";
  }

  if (!nrrd_handle_.get_rep()) {
    return;
  }

  ColorMapHandle cmap = volume_->get_colormap();
  texture_ = new ColorMappedNrrdTextureObj(nrrd_handle_, cmap);
  outline_ = new NrrdBitmaskOutline(nrrd_handle_);
  geom_texture_ = new GeomColorMappedNrrdTextureObj(texture_);
  tex_dirty_ = true;
  
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


#define NRRD_EXEC(__nrrd_command__) \
  if (__nrrd_command__) { \
    char *err = biffGetDone(NRRD); \
    cerr << string("Error on line #")+to_string(__LINE__) + \
	    string(" executing nrrd command: \n")+ \
            string(#__nrrd_command__)+string("\n")+ \
            string("Message: ")+string(err); \
    free(err); \
    return; \
  }




void
VolumeSlice::extract_nrrd_slice_from_volume() {  
  vector<int> sindex = volume_->world_to_index(plane_.project(Point(0,0,0)));
  unsigned int ax = axis();

  int min_slice = sindex[ax];
  int max_slice = sindex[ax];
  int slice = sindex[ax];  
  if (slice < 0 || slice >= (int)volume_->nrrd_handle_->nrrd_->axis[ax].size)
  {
    return;
  }

  volume_->mutex_->lock();
  nrrd_handle_ = new NrrdData;
  Nrrd *dst = nrrd_handle_->nrrd_;
  Nrrd *src = volume_->nrrd_handle_->nrrd_;
  ASSERT(src);

  if (min_slice != max_slice) {
    size_t *min = new size_t[src->dim];
    size_t *max = new size_t[src->dim];
    for (unsigned int i = 0; i < src->dim; i++) {
      min[i] = 0;
      max[i] = src->axis[i].size-1;
    }
    min[ax] = Min(min_slice, max_slice);
    max[ax] = Max(min_slice, max_slice);
    NrrdDataHandle tmp1_handle = new NrrdData;
    NRRD_EXEC(nrrdCrop(tmp1_handle->nrrd_, src, min, max));
    NRRD_EXEC(nrrdProject(dst, tmp1_handle->nrrd_, ax, 
                          nrrdMeasureMax, nrrdTypeDefault));
  } else {
    NRRD_EXEC(nrrdSlice(dst, src, ax, min_slice));
  }

  if (label_ && dst->type == nrrdTypeFloat) {
    nrrd_handle_ = VolumeOps::float_to_bit(nrrd_handle_, 0, label_);
  }

  volume_->mutex_->unlock();
}



#if 0
static GLubyte stripe[4*32] = { 
  0x33, 0x33, 0x33, 0x33,
  0x66, 0x66, 0x66, 0x66,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x99, 0x99, 0x99, 0x99,
  
  0x33, 0x33, 0x33, 0x33,
  0x66, 0x66, 0x66, 0x66,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x99, 0x99, 0x99, 0x99,
  
  0x33, 0x33, 0x33, 0x33,
  0x66, 0x66, 0x66, 0x66,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x99, 0x99, 0x99, 0x99,
  
  0x33, 0x33, 0x33, 0x33,
  0x66, 0x66, 0x66, 0x66,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x99, 0x99, 0x99, 0x99,
  
  0x33, 0x33, 0x33, 0x33,
  0x66, 0x66, 0x66, 0x66,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x99, 0x99, 0x99, 0x99,
  
  0x33, 0x33, 0x33, 0x33,
  0x66, 0x66, 0x66, 0x66,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x99, 0x99, 0x99, 0x99,
  
  0x33, 0x33, 0x33, 0x33,
  0x66, 0x66, 0x66, 0x66,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x99, 0x99, 0x99, 0x99,
  
  0x33, 0x33, 0x33, 0x33,
  0x66, 0x66, 0x66, 0x66,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x99, 0x99, 0x99, 0x99,
};


static GLubyte stripe2[4*36] = { 
  0x33, 0x33, 0x33, 0x33,
  0x99, 0x99, 0x99, 0x99,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x66, 0x66, 0x66, 0x66,

  0x33, 0x33, 0x33, 0x33,
  0x99, 0x99, 0x99, 0x99,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x66, 0x66, 0x66, 0x66,

  0x33, 0x33, 0x33, 0x33,
  0x99, 0x99, 0x99, 0x99,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x66, 0x66, 0x66, 0x66,

  0x33, 0x33, 0x33, 0x33,
  0x99, 0x99, 0x99, 0x99,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x66, 0x66, 0x66, 0x66,

  0x33, 0x33, 0x33, 0x33,
  0x99, 0x99, 0x99, 0x99,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x66, 0x66, 0x66, 0x66,

  0x33, 0x33, 0x33, 0x33,
  0x99, 0x99, 0x99, 0x99,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x66, 0x66, 0x66, 0x66,

  0x33, 0x33, 0x33, 0x33,
  0x99, 0x99, 0x99, 0x99,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x66, 0x66, 0x66, 0x66,

  0x33, 0x33, 0x33, 0x33,
  0x99, 0x99, 0x99, 0x99,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x66, 0x66, 0x66, 0x66,

  0x33, 0x33, 0x33, 0x33,
  0x99, 0x99, 0x99, 0x99,
  0xCC, 0xCC, 0xCC, 0xCC,
  0x66, 0x66, 0x66, 0x66,
};
#endif

static GLubyte stripe3[4*36] = { 
  0x11, 0x11, 0x11, 0x11,
  0x22, 0x22, 0x22, 0x22,
  0x44, 0x44, 0x44, 0x44,
  0x88, 0x88, 0x88, 0x88,

  0x11, 0x11, 0x11, 0x11,
  0x22, 0x22, 0x22, 0x22,
  0x44, 0x44, 0x44, 0x44,
  0x88, 0x88, 0x88, 0x88,

  0x11, 0x11, 0x11, 0x11,
  0x22, 0x22, 0x22, 0x22,
  0x44, 0x44, 0x44, 0x44,
  0x88, 0x88, 0x88, 0x88,

  0x11, 0x11, 0x11, 0x11,
  0x22, 0x22, 0x22, 0x22,
  0x44, 0x44, 0x44, 0x44,
  0x88, 0x88, 0x88, 0x88,

  0x11, 0x11, 0x11, 0x11,
  0x22, 0x22, 0x22, 0x22,
  0x44, 0x44, 0x44, 0x44,
  0x88, 0x88, 0x88, 0x88,

  0x11, 0x11, 0x11, 0x11,
  0x22, 0x22, 0x22, 0x22,
  0x44, 0x44, 0x44, 0x44,
  0x88, 0x88, 0x88, 0x88,

  0x11, 0x11, 0x11, 0x11,
  0x22, 0x22, 0x22, 0x22,
  0x44, 0x44, 0x44, 0x44,
  0x88, 0x88, 0x88, 0x88,

  0x11, 0x11, 0x11, 0x11,
  0x22, 0x22, 0x22, 0x22,
  0x44, 0x44, 0x44, 0x44,
  0x88, 0x88, 0x88, 0x88,

  0x11, 0x11, 0x11, 0x11,
  0x22, 0x22, 0x22, 0x22,
  0x44, 0x44, 0x44, 0x44,
  0x88, 0x88, 0x88, 0x88,

};
    

static GLubyte stripe4[4*36] = { 
  0x11, 0x11, 0x11, 0x11,
  0x88, 0x88, 0x88, 0x88,
  0x44, 0x44, 0x44, 0x44,
  0x22, 0x22, 0x22, 0x22,

  0x11, 0x11, 0x11, 0x11,
  0x88, 0x88, 0x88, 0x88,
  0x44, 0x44, 0x44, 0x44,
  0x22, 0x22, 0x22, 0x22,

  0x11, 0x11, 0x11, 0x11,
  0x88, 0x88, 0x88, 0x88,
  0x44, 0x44, 0x44, 0x44,
  0x22, 0x22, 0x22, 0x22,

  0x11, 0x11, 0x11, 0x11,
  0x88, 0x88, 0x88, 0x88,
  0x44, 0x44, 0x44, 0x44,
  0x22, 0x22, 0x22, 0x22,

  0x11, 0x11, 0x11, 0x11,
  0x88, 0x88, 0x88, 0x88,
  0x44, 0x44, 0x44, 0x44,
  0x22, 0x22, 0x22, 0x22,

  0x11, 0x11, 0x11, 0x11,
  0x88, 0x88, 0x88, 0x88,
  0x44, 0x44, 0x44, 0x44,
  0x22, 0x22, 0x22, 0x22,

  0x11, 0x11, 0x11, 0x11,
  0x88, 0x88, 0x88, 0x88,
  0x44, 0x44, 0x44, 0x44,
  0x22, 0x22, 0x22, 0x22,

  0x11, 0x11, 0x11, 0x11,
  0x88, 0x88, 0x88, 0x88,
  0x44, 0x44, 0x44, 0x44,
  0x22, 0x22, 0x22, 0x22,

  0x11, 0x11, 0x11, 0x11,
  0x88, 0x88, 0x88, 0x88,
  0x44, 0x44, 0x44, 0x44,
  0x22, 0x22, 0x22, 0x22,

};


void
VolumeSlice::draw()
{
  if (!volume_->visible_) return;
  if (!nrrd_handle_.get_rep()) return;
  if (!texture_.get_rep() || !outline_.get_rep()) return;

  //  float a = volume_->opacity_;
  //  glColor4f(a,a,a,a);
  glDisable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  //  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  CHECK_OPENGL_ERROR();  


  if (tex_dirty_) {
    tex_dirty_ = false;
    texture_->set_label(volume_->label_);
    texture_->set_clut_minmax(volume_->clut_min_, volume_->clut_max_);
    ColorMapHandle cmap = volume_->get_colormap();
    texture_->set_colormap(cmap);
    outline_->set_colormap(cmap);
  }
   
  int depth = 0;
  NrrdVolume *parent = volume_;
  while (parent->parent_) {
    parent = parent->parent_;
    depth++;
  }
  if (texture_.get_rep()) {    
    if (volume_->label_) {
      GLubyte *pattern = stripe4;
      switch (depth % 4) {
      case 0: pattern = stripe4; break;
      case 1: pattern = stripe4+8; break;
      case 2: pattern = stripe3; break;
      default:
      case 3: pattern = stripe3+8; break;
      }

      glPolygonStipple(pattern);
      glEnable(GL_POLYGON_STIPPLE);
    }
    texture_->set_opacity(volume_->opacity_);
    texture_->set_coords(pos_, xdir_, ydir_);
    texture_->draw_quad();
    if (volume_->label_) {
      glDisable(GL_POLYGON_STIPPLE);
    }    
  }
  
  if (volume_->label_ && outline_.get_rep()) {
    outline_->set_coords(pos_, xdir_, ydir_);
    outline_->draw_lines(3.0, volume_->label_);
  }
}


}
