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
//    File   : SliceWindow.cc
//    Author : McKay Davis
//    Date   : Fri Oct 13 15:08:39 2006


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

#ifdef _WIN32
#define snprintf _snprintf
#endif

namespace SCIRun {

SliceWindow::SliceWindow(Skinner::Variables *variables,
                                  Painter *painter) :  
  Skinner::Parent(variables), 
  painter_(painter),
  name_(variables->get_id()),
  viewport_(0),
  slices_(),
  slice_map_(),
  paint_layer_(0),
  center_(0,0,0),
  normal_(0,0,0),
  slice_num_(0,0), 
  axis_(2),
  zoom_(0,100.0),
  slab_min_(0,0),
  slab_max_(0,0),
  redraw_(true),
  autoview_(true),
  mode_(0,0),
  show_guidelines_(0,1),
  cursor_pixmap_(-1),
  show_grid_(variables, "SliceWindow::GridVisible",1),
  show_slices_(variables, "SliceWindow::SlicesVisible",1),
  geom_switch_(0),
  geom_group_(0)

{
  Skinner::Var<int> axis(variables, "axis", 2);
  set_axis(axis());
  //  axis_ = axis;
  REGISTER_CATCHER_TARGET(SliceWindow::redraw);
  REGISTER_CATCHER_TARGET(SliceWindow::Autoview);
  REGISTER_CATCHER_TARGET(SliceWindow::zoom_in);
  REGISTER_CATCHER_TARGET(SliceWindow::zoom_out);
}



void
SliceWindow::redraw() {
  throw_signal("SliceWindow::mark_redraw");
}

void
SliceWindow::push_gl_2d_view() {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);
  double vpw = get_region().width();
  double vph = get_region().height();
  glScaled(1.0/vpw, 1.0/vph, 1.0);
  CHECK_OPENGL_ERROR();
}


void
SliceWindow::pop_gl_2d_view() {
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  CHECK_OPENGL_ERROR();
}


void
SliceWindow::render_guide_lines(Point mouse) {
  if (!show_guidelines_()) return;

  //  GLdouble yellow[4] = { 1.0, 0.76, 0.1, 0.8 };
  GLdouble white[4] = { 1.0, 1.0, 1.0, 1.0 };

  push_gl_2d_view();
  double vpw = get_region().width();
  double vph = get_region().height();

  glColor4dv(white);
  glBegin(GL_LINES); 
  glVertex3d(0, mouse.y(), mouse.z());
  glVertex3d(vpw, mouse.y(), mouse.z());
  glVertex3d(mouse.x(), 0, mouse.z());
  glVertex3d(mouse.x(), vph, mouse.z());
  glEnd();
  CHECK_OPENGL_ERROR();

  pop_gl_2d_view();

}



// renders vertical and horizontal bars that represent
// selected slices in other dimensions
void
Painter::draw_slice_lines(SliceWindow &window)
{
  if (!current_volume_) return;
  double upp = 100.0 / window.zoom_;    // World space units per one pixel

  //  GLdouble blue[4] = { 0.1, 0.4, 1.0, 0.8 };
  //  GLdouble green[4] = { 0.5, 1.0, 0.1, 0.8 };
  GLdouble red[4] = { 0.8, 0.2, 0.4, 0.9 };

  // Vector scale = current_volume_->scale();
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  vector<int> zero_idx(current_volume_->nrrd_handle_->nrrd_->dim, 0);
  for (unsigned int win = 0; win < windows_.size(); ++win) {
      SliceWindow &window2 = *(windows_[win]);
      if (&window == &window2) continue;
      if (Dot(window.normal_, window2.normal_) > 0.999) continue;
      Vector span = Cross(window.normal_, window2.normal_);
      vector<double> span_index = current_volume_->vector_to_index(span);
      int span_axis = max_vector_magnitude_index(span_index);

      vector<int> pos_idx = current_volume_->world_to_index(window2.center_);
      vector<int> min_idx = pos_idx;
      min_idx[span_axis] = 0;
      vector<int> max_idx = pos_idx;
      max_idx[span_axis] = current_volume_->nrrd_handle_->nrrd_->axis[span_axis].size;
      Point min = current_volume_->index_to_world(min_idx);
      Point max = current_volume_->index_to_world(max_idx);
      vector<int> one_idx = zero_idx;
      one_idx[window2.axis_+1] = 1;
      double scale = (current_volume_->index_to_world(one_idx) - 
                      current_volume_->index_to_world(zero_idx)).length();
      Vector wid = window2.normal_;
      wid.normalize();
      wid *= Max(upp, scale);
      glColor4dv(red);
      glBegin(GL_QUADS);    
      glVertex3dv(&min(0));
      glVertex3dv(&max(0));
      min = min + wid;
      max = max + wid;
      glVertex3dv(&max(0));
      glVertex3dv(&min(0));
      glEnd();
  }
}




double div_d(double dividend, double divisor) {
  return Floor(dividend/divisor);
}

double mod_d(double dividend, double divisor) {
  return dividend - Floor(dividend/divisor)*divisor;
}


void
SliceWindow::render_frame(double x,
                                   double y,
                                   double border_wid,
                                   double border_hei,
                                   double *color1,
                                   double *color2)
{
  const double vw = get_region().width();
  const double vh = get_region().height();
  if (color1)
    glColor4dv(color1);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  CHECK_OPENGL_ERROR();

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);
  glScaled(1.0/vw, 1.0/vh, 1.0);
  CHECK_OPENGL_ERROR();

  glBegin(GL_QUADS);
  glVertex3d(x,y,0);
  glVertex3d(vw-x,y,0);
  glVertex3d(vw-x,y+border_hei,0);
  glVertex3d(x,y+border_hei,0);

  glVertex3d(vw-x-border_wid,y+border_hei,0);
  glVertex3d(vw-x,y+border_hei,0);
  glVertex3d(vw-x,vh-y,0);
  glVertex3d(vw-x-border_wid,vh-y,0);

  if (color2)
    glColor4dv(color2);

  glVertex3d(x,vh-y,0);
  glVertex3d(vw-x,vh-y,0);
  glVertex3d(vw-x-border_wid,vh-y-border_hei,0);
  glVertex3d(x,vh-y-border_hei,0);

  glVertex3d(x,y,0);
  glVertex3d(x+border_wid,y+border_hei,0);
  glVertex3d(x+border_wid,vh-y-border_hei,0);
  glVertex3d(x,vh-y-border_hei,0);

  glEnd();
  CHECK_OPENGL_ERROR();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  CHECK_OPENGL_ERROR();
}

  


// renders vertical and horizontal bars that represent
// selected slices in other dimensions
void
SliceWindow::render_grid()
{
  //  double one = 100.0 / zoom_;    // World space units per one pixel
  double units = zoom_ / 100.0;  // Pixels per world space unit
  const double pixels = 100.0;    // Number of target pixels for grid gap

  vector<double> gaps(1, 1.0);
  gaps.push_back(1/2.0);
  gaps.push_back(1/5.0);

  double realdiff = 10000;
  int selected = 0;
  for (unsigned int i = 0; i < gaps.size(); ++i) {
    bool done = false;
    double diff = fabs(gaps[i]*units-pixels);
    while (!done) {
      if (fabs((gaps[i]*10.0)*units - pixels) < diff) 
        gaps[i] *= 10.0;
      else if (fabs((gaps[i]/10.0)*units - pixels) < diff) 
        gaps[i] /= 10.0;
      else
        done = true;
      diff = fabs(gaps[i]*units-pixels);
    }

    if (diff < realdiff) {
      realdiff = diff;
      selected = i;
    }
  }
  double gap = gaps[selected];

  const Skinner::RectRegion &region = get_region();
  const int vw = Ceil(region.width());
  const int vh = Ceil(region.height());

  //  double grey1[4] = { 0.75, 0.75, 0.75, 1.0 };
  //  double grey2[4] = { 0.5, 0.5, 0.5, 1.0 };
  //  double grey3[4] = { 0.25, 0.25, 0.25, 1.0 };
  //  double white[4] = { 1,1,1,1 };
  //  render_frame(0,0, 15, 15, grey1);
  //  render_frame(15,15, 3, 3, white, grey2 );
  //  render_frame(17,17, 2, 2, grey3);
  double grid_color = 0.25;

  glDisable(GL_TEXTURE_2D);
  CHECK_OPENGL_ERROR();

  Point min = screen_to_world(Floor(region.x1()),Floor(region.y1()));
  Point max = screen_to_world(Ceil(region.x2())-1, Ceil(region.y2())-1);

  int xax = x_axis();
  int yax = y_axis();
  min(xax) = div_d(min(xax), gap)*gap;
  min(yax) = div_d(min(yax), gap)*gap;

  vector<string> lab;
  lab.push_back("X: ");
  lab.push_back("Y: ");
  lab.push_back("Z: ");

  int num = 0;
  Point linemin = min;
  Point linemax = min;
  linemax(yax) = max(yax);
  string str;
  TextRenderer *renderer = FontManager::get_renderer(20);
  renderer->set_color(1,1,1,1);
  renderer->set_shadow_color(0,0,0,1);
  renderer->set_shadow_offset(1, -1);
  while (linemin(xax) < max(xax)) {
    linemin(xax) = linemax(xax) = min(xax) + gap*num;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);

    glColor4d(grid_color, grid_color, grid_color, 0.25);
    glBegin(GL_LINES);
    glVertex3dv(&linemin(0));
    glVertex3dv(&linemax(0));
    glEnd();

    //    str = lab[xax]+to_string(linemin(xax));
    str = to_string(linemin(xax));
    Point pos = world_to_screen(linemin);
    renderer->render(str, pos.x()+1, 2, 
                     TextRenderer::SHADOW | 
                     TextRenderer:: SW | 
                     TextRenderer::REVERSE);
    
    //    pos = world_to_screen(linemax);
    renderer->render(str, pos.x()+1, vh-2, 
                     TextRenderer::SHADOW | 
                     TextRenderer:: NW | 
                     TextRenderer::REVERSE);
    num++;
  }
  //  int wid = text.width();

  num = 0;
  linemin = linemax = min;
  linemax(xax) = max(xax);
  while (linemin(yax) < max(yax)) {
    linemin(yax) = linemax(yax) = min(yax) + gap*num;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);

    glColor4d(grid_color, grid_color, grid_color, 0.25);
    glBegin(GL_LINES);
    glVertex3dv(&linemin(0));
    glVertex3dv(&linemax(0));
    glEnd();

    str = to_string(linemin(yax));
    Point pos = world_to_screen(linemin);
    renderer->render(str, 2, pos.y(), 
                     TextRenderer::SHADOW | 
                     TextRenderer::NW | 
                     TextRenderer::VERTICAL | 
                     TextRenderer::REVERSE);

    renderer->render(str, vw-2, pos.y(), 
                     TextRenderer::SHADOW | 
                     TextRenderer::NE | 
                     TextRenderer::VERTICAL | 
                     TextRenderer::REVERSE);
    num++;
  }
  CHECK_OPENGL_ERROR();
}



// Returns the index to the axis coordinate that is most parallel and 
// in the direction of X in the screen.  
// 0 for x, 1 for y, and 2 for z
int
SliceWindow::x_axis()
{
  Vector adir = Abs(x_dir());
  if ((adir[0] > adir[1]) && (adir[0] > adir[2])) return 0;
  if ((adir[1] > adir[0]) && (adir[1] > adir[2])) return 1;
  return 2;
}

// Returns the index to the axis coordinate that is most parallel and 
// in the direction of Y in the screen.  
// 0 for x, 1 for y, and 2 for z
int
SliceWindow::y_axis()
{
  Vector adir = Abs(y_dir());
  if ((adir[0] > adir[1]) && (adir[0] > adir[2])) return 0;
  if ((adir[1] > adir[0]) && (adir[1] > adir[2])) return 1;
  return 2;
}

void
SliceWindow::setup_gl_view()
{
  //  glViewport(region_.x1(), region_.y1(), 
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  //  CHECK_OPENGL_ERROR();
  if (axis_ == 0) { // screen +X -> +Y, screen +Y -> +Z
    glRotated(-90,0.,1.,0.);
    glRotated(-90,1.,0.,0.);
  } else if (axis_ == 1) { // screen +X -> +X, screen +Y -> +Z
    glRotated(-90,1.,0.,0.);
  }
  CHECK_OPENGL_ERROR();
  
  // Do this here because x_axis and y_axis functions use these matrices
  glGetIntegerv(GL_VIEWPORT, gl_viewport_);
  glGetDoublev(GL_MODELVIEW_MATRIX, gl_modelview_matrix_);
  glGetDoublev(GL_PROJECTION_MATRIX, gl_projection_matrix_);
  CHECK_OPENGL_ERROR();

  double hwid = get_region().width()*50.0/zoom_();
  double hhei = get_region().height()*50.0/zoom_();

  double cx = center_(x_axis());
  double cy = center_(y_axis());
  
  double diagonal = hwid*hwid+hhei*hhei;

  double maxz = center_(axis_) + diagonal*zoom_()/100.0;
  double minz = center_(axis_) - diagonal*zoom_()/100.0;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(cx - hwid, cx + hwid, 
	  cy - hhei, cy + hhei, 
	  minz, maxz);

  glGetIntegerv(GL_VIEWPORT, gl_viewport_);
  glGetDoublev(GL_MODELVIEW_MATRIX, gl_modelview_matrix_);
  glGetDoublev(GL_PROJECTION_MATRIX, gl_projection_matrix_);
  CHECK_OPENGL_ERROR();
}



// Right	= -X
// Left		= +X
// Posterior	= -Y
// Anterior	= +X
// Inferior	= -Z
// Superior	= +Z
void
SliceWindow::render_orientation_text()
{
#if 0
  TextRenderer *text = painter_->font3_;
  if (!text) return;

  int prim = x_axis();
  int sec = y_axis();
  
  string ltext, rtext, ttext, btext;

  if (painter_->anatomical_coordinates_()) {
    switch (prim % 3) {
    case 0: ltext = "R"; rtext = "L"; break;
    case 1: ltext = "P"; rtext = "A"; break;
    default:
    case 2: ltext = "I"; rtext = "S"; break;
    }
    
    switch (sec % 3) {
    case 0: btext = "R"; ttext = "L"; break;
    case 1: btext = "P"; ttext = "A"; break;
    default:
    case 2: btext = "I"; ttext = "S"; break;
    }
  } else {
    switch (prim % 3) {
    case 0: ltext = "-X"; rtext = "+X"; break;
    case 1: ltext = "-Y"; rtext = "+Y"; break;
    default:
    case 2: ltext = "-Z"; rtext = "+Z"; break;
    }
    switch (sec % 3) {
    case 0: btext = "-X"; ttext = "+X"; break;
    case 1: btext = "-Y"; ttext = "+Y"; break;
    default:
    case 2: btext = "-Z"; ttext = "+Z"; break;
    }
  }    


  if (prim >= 3) SWAP (ltext, rtext);
  if (sec >= 3) SWAP (ttext, btext);
  text->set_shadow_offset(2,-2);
  text->render(ltext, 2, get_region().height()/2,
            TextRenderer::W | TextRenderer::SHADOW | TextRenderer::REVERSE);

  text->render(rtext,get_region().width()-2, get_region().height()/2,
               TextRenderer::E | TextRenderer::SHADOW | TextRenderer::REVERSE);

  text->render(btext,get_region().width()/2, 2,
               TextRenderer::S | TextRenderer::SHADOW | TextRenderer::REVERSE);

  text->render(ttext,get_region().width()/2, get_region().height()-2, 
               TextRenderer::N | TextRenderer::SHADOW | TextRenderer::REVERSE);
#endif
}



void
SliceWindow::render_progress_bar() {
  GLdouble grey[4] = { 0.6, 0.6, 0.6, 0.6 }; 
  GLdouble white[4] = { 1.0, 1.0, 1.0, 1.0 }; 
  GLdouble black[4] = { 0.0, 0.0, 0.0, 1.0 }; 
  GLdouble yellow[4] = { 1.0, 0.76, 0.1, 1.0 };
  GLdouble lt_yellow[4] = { 0.8, 0.5, 0.1, 1.0 };  
  
  GLdouble *colors[5] = { lt_yellow, yellow, black, grey, white };
  GLdouble widths[5] = { 11, 9.0, 7.0, 5.0, 1.0 }; 

  push_gl_2d_view();

  double vpw = get_region().width();
  double vph = get_region().height();
  double x_off = 50;
  double h = 50;
  double gap = 5;
  //  double y_off = 20;

  Point ll(x_off, vph/2.0 - h/2, 0);
  Point lr(vpw-x_off, vph/2.0 - h/2, 0);
  Point ur(vpw-x_off, vph/2.0 + h/2, 0);
  Point ul(x_off, vph/2.0 + h/2, 0);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_LINE_SMOOTH);
  for (int pass = 2; pass < 5; ++pass) {
    glColor4dv(colors[pass]);
    glLineWidth(widths[pass]);    

    glBegin(GL_LINE_LOOP);
    {
      glVertex3dv(&ll(0));
      glVertex3dv(&lr(0));
      glVertex3dv(&ur(0));
      glVertex3dv(&ul(0));
    }
    glEnd();
  }
  glLineWidth(1.0);
  glDisable(GL_LINE_SMOOTH);
  CHECK_OPENGL_ERROR();

  Vector right = Vector(vpw - 2 *x_off - 2*gap, 0, 0);
  Vector up = Vector(0, h - gap * 2, 0);

  ll = ll + Vector(gap, gap, 0);
  lr = ll + right;
  ur = lr + up;
  ul = ll + up;

  glColor4dv(yellow);
  glBegin(GL_QUADS);
  glVertex3dv(&ll(0));
  glVertex3dv(&lr(0));
  glVertex3dv(&ur(0));
  glVertex3dv(&ul(0));
  glEnd();
  CHECK_OPENGL_ERROR();
  


  pop_gl_2d_view();
}





void
SliceWindow::render_text()
{
#if 0
  const int yoff = 19;
  const int xoff = 19;
  const double vw = get_region().width();
  const double vh = get_region().height();

  TextRenderer *renderer = FontManager::get_renderer(20);
  NrrdVolume *vol = painter_->current_volume_;


  font1.set_color(1.0, 1.0, 1.0, 1.0);
  const int y_pos = font1.height("X")+2;
  font1.render("Zoom: "+to_string(zoom_())+"%", xoff, yoff, 
               TextRenderer::SHADOW | TextRenderer::SW);


  if (vol) {
    const float ww = vol->clut_max_ - vol->clut_min_;
    const float wl = vol->clut_min_ + ww/2.0;
    font1.render("WL: " + to_string(wl) +  " -- WW: " + to_string(ww),
                 xoff, y_pos+yoff,
                 TextRenderer::SHADOW | TextRenderer::SW);

    font1.render("Min: " + to_string(vol->clut_min_) + 
                 " -- Max: " + to_string(vol->clut_max_),
                 xoff, y_pos*2+yoff, TextRenderer::SHADOW | TextRenderer::SW);
    if (this == painter_->event_.window_) {
      font1.render
                   region_.width()-2-xoff, yoff,
                   TextRenderer::SHADOW | TextRenderer::SE);
      vector<int> index = vol->world_to_index(painter_->event_.position_);
      if (vol->index_valid(index)) {
        font1.render(,
                     region_.width()-2-xoff, yoff+y_pos,
                     TextRenderer::SHADOW | TextRenderer::SE);
        double val = 1.0;
        vol->get_value(index, val);
        font1.render("Value: " + to_string(val),
                     region_.width()-2-xoff,y_pos*2+yoff, 
                     TextRenderer::SHADOW | TextRenderer::SE);
      }
    }
  }
    
  render_orientation_text();

  string str;
  if (!painter_->anatomical_coordinates_()) { 
    switch (axis_) {
    case 0: str = "Sagittal"; break;
    case 1: str = "Coronal"; break;
    default:
    case 2: str = "Axial"; break;
    }
  } else {
    switch (axis_) {
    case 0: str = "YZ Plane"; break;
    case 1: str = "XZ Plane"; break;
    default:
    case 2: str = "XY Plane"; break;
    }
  }

  if (mode_ == slab_e) str = "SLAB - "+str;
  else if (mode_ == mip_e) str = "MIP - "+str;

  font2.set_shadow_offset(2,-2);
  font2.render(str, region_.width() - 2, 2,
               TextRenderer::SHADOW | 
               TextRenderer::SE | 
               TextRenderer::REVERSE);
#endif
}
void
SliceWindow::extract_slices() {

  NrrdVolumes volumes;
  painter_->build_volume_list(volumes);

  for (unsigned int s = slices_.size(); s < slices_.size(); ++s) {
    delete slices_[s];
  }

  if (slices_.size() > volumes.size())
    slices_.resize(volumes.size());
  
  for (unsigned int s = slices_.size(); s < volumes.size(); ++s) {
    if (volumes[s] == painter_->current_volume_) {
      center_ = painter_->current_volume_->center();
    }
    VolumeSlice *slice = new VolumeSlice(painter_,volumes[s],this,center_,normal_);
    slices_.push_back(slice);
  }

  slice_map_.clear();
  for (unsigned int s = 0; s < volumes.size(); ++s) {
    if (volumes[s] == painter_->current_volume_ &&
        !painter_->current_volume_->inside_p(center_)) {
      int ax = axis_;
      center_(ax) = painter_->current_volume_->center()(ax);
    }

    slice_map_[volumes[s]] = slices_[s];
    slices_[s]->volume_ = volumes[s];
    slices_[s]->plane_ = Plane(center_, normal_);
    slices_[s]->nrrd_dirty_ = true;
  }

  

  if (paint_layer_) {
    delete paint_layer_;
    paint_layer_ = 0;
  }
}



void
SliceWindow::set_axis(unsigned int axis) {
  axis_ = axis % 3;
  normal_ = Vector(axis == 0 ? 1 : 0,
                   axis == 1 ? 1 : 0,
                   axis == 2 ? 1 : 0);
  extract_slices();
  redraw_ = true;
}


void
SliceWindow::prev_slice()
{
  NrrdVolume *volume = painter_->current_volume_;
  if (!volume) return;
  vector<double> delta = volume->vector_to_index(normal_);
  unsigned int index = max_vector_magnitude_index(delta);
  delta[index] /= fabs(delta[index]);
  Point new_center = center_ - volume->index_to_vector(delta);
  if (!painter_->current_volume_->inside_p(new_center)) return;
  center_ = new_center;
  extract_slices();
  painter_->redraw_all();
}


void
SliceWindow::next_slice()
{
  NrrdVolume *volume = painter_->current_volume_;
  if (!volume) return;
  vector<double> delta = volume->vector_to_index(-normal_);
  unsigned int index = max_vector_magnitude_index(delta);
  delta[index] /= fabs(delta[index]);
  Point new_center = center_ - volume->index_to_vector(delta);
  if (!painter_->current_volume_->inside_p(new_center)) return;
  center_ = new_center;
  extract_slices();
  painter_->redraw_all();
}

BaseTool::propagation_state_e
SliceWindow::zoom_in(event_handle_t)
{
  zoom_ *= 1.1;
  redraw();
  return CONTINUE_E;
}


BaseTool::propagation_state_e
SliceWindow::zoom_out(event_handle_t)
{
  zoom_ /= 1.1;
  redraw();
  return CONTINUE_E;
}
  
Point
SliceWindow::screen_to_world(unsigned int x, unsigned int y) {
  GLdouble xyz[3];
  gluUnProject(double(x)+0.5, double(y)+0.5, 0,
	       gl_modelview_matrix_, 
	       gl_projection_matrix_,
	       gl_viewport_,
	       xyz+0, xyz+1, xyz+2);
  xyz[axis_] = center_(axis_);
  return Point(xyz[0], xyz[1], xyz[2]);
}


Point
SliceWindow::world_to_screen(const Point &world)
{
  GLdouble xyz[3];
  gluProject(world(0), world(1), world(2),
             gl_modelview_matrix_, 
             gl_projection_matrix_,
	     gl_viewport_,
             xyz+0, xyz+1, xyz+2);
  xyz[0] -= gl_viewport_[0];
  xyz[1] -= gl_viewport_[1];

  return Point(xyz[0], xyz[1], xyz[2]);
}


Vector
SliceWindow::x_dir()
{
  return screen_to_world(1,0) - screen_to_world(0,0);
}

Vector
SliceWindow::y_dir()
{
  return screen_to_world(0,1) - screen_to_world(0,0);
}


void
SliceWindow::set_probe() {
  if (!painter_->current_volume_) return;
  if (painter_->cur_window_ != this &&
      painter_->current_volume_->inside_p(painter_->pointer_pos_)) {
    center_(axis_) = painter_->pointer_pos_(axis_);
    extract_slices();
  }
}

BaseTool::propagation_state_e
SliceWindow::Autoview(event_handle_t) {
  if (painter_->current_volume_) {
    autoview(painter_->current_volume_);
  }
  return CONTINUE_E;
}


void
SliceWindow::autoview(NrrdVolume *volume, double offset) {
  autoview_ = false;
  double wid = get_region().width() -  2*offset;
  double hei = get_region().height() - 2*offset;

  int xax = x_axis();
  int yax = y_axis();

  if (volume) {
    vector<int> zero(volume->nrrd_handle_->nrrd_->dim, 0);
    vector<int> index = zero;
    index[xax+1] = volume->nrrd_handle_->nrrd_->axis[xax+1].size;
    double w_wid = (volume->index_to_world(index) - 
                    volume->index_to_world(zero)).length();
    double w_ratio = wid/w_wid;
    
    index = zero;
    index[yax+1] = volume->nrrd_handle_->nrrd_->axis[yax+1].size;
    double w_hei = (volume->index_to_world(index) - 
                    volume->index_to_world(zero)).length();
    double h_ratio = hei/w_hei;
    
    zoom_ = Min(w_ratio*100.0, h_ratio*100.0);
    if (zoom_ < 1.0) zoom_ = 100.0; // ??
    center_(xax) = volume->center()(xax);
    center_(yax) = volume->center()(yax);
  } else {
    center_ = Point(0,0,0);
    zoom_ = 100;
  }
  redraw();
}
   

BaseTool::propagation_state_e
SliceWindow::process_event(event_handle_t event) {
  PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
  const Skinner::RectRegion &region = get_region();
  if (pointer) {
//     Point pointer_point(pointer->get_x(), pointer->get_y(),0);
//     Point min(region_.x1(), region_.y1(),0);
//     Point max(region_.x2(), region_.y2(),0);
    if (pointer->get_x() >= region.x1()  &&
        pointer->get_x() < region.x2()  &&
        pointer->get_y() >= region.y1()  &&
        pointer->get_y() < region.y2()) {
      //      if (region.inside(pointer->get_x(), pointer->get_y())) {
      painter_->cur_window_ = this;
      painter_->pointer_pos_ = screen_to_world(pointer->get_x(), 
                                               pointer->get_y());
    }
  }

  if (painter_->cur_window_ == this) {
    painter_->tm_.propagate_event(event);
  }

#if 0
  WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
  if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
    render_gl();

    
  }
#endif
  
  
  return Parent::process_event(event);

  //  return BaseTool::CONTINUE_E;
}



GeomIndexedGroup *
SliceWindow::get_geom_group() {
  if (!geom_group_) {
    geom_group_ = new GeomIndexedGroup();
    geom_switch_ = new GeomSkinnerVarSwitch(geom_group_, show_slices_);
    event_handle_t add_geom_switch_event = 
      new SceneGraphEvent(geom_switch_, get_id());
    EventManager::add_event(add_geom_switch_event);
  }
  return geom_group_;
}




string
double_to_string(double val)
{
  char s[50];
  snprintf(s, 49, "%1.2f", val);
  return string(s);
}

int
SliceWindow::get_signal_id(const string &signalname) const {
  if (signalname == "SliceWindow::mark_redraw") return 1;
  return 0;
}


BaseTool::propagation_state_e
SliceWindow::redraw(event_handle_t) {
  const Skinner::RectRegion &region = get_region();
  if (region.width() <= 0 || region.height() <= 0) return CONTINUE_E;
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport);    
  glViewport(Floor(region.x1()), Floor(region.y1()), 
             Ceil(region.width()), Ceil(region.height()));

  painter_->volume_lock_.lock();

  NrrdVolume *vol = painter_->current_volume_;

  Skinner::Var<string> clut_ww_wl(painter_->get_vars(), "clut_ww_wl", "");
  Skinner::Var<double> value(painter_->get_vars(), "cursor_value", 0.0);
  string clut_min_max = "";
  string xyz_pos = "";
  string sca_pos = "";
  if (vol) {
    const float ww = vol->clut_max_ - vol->clut_min_;
    const float wl = vol->clut_min_ + ww/2.0;
    clut_ww_wl = "WL: " + to_string(wl) +  " -- WW: " + to_string(ww);

    clut_min_max = ("Min: " + to_string(vol->clut_min_) + 
                    " -- Max: " + to_string(vol->clut_max_));
    vector<int> index = vol->world_to_index(painter_->pointer_pos_);
    if (vol->index_valid(index) && painter_->cur_window_ == this) {
      sca_pos = ("S: "+to_string(index[1])+
                 " C: "+to_string(index[2])+
                 " A: "+to_string(index[3]));
      xyz_pos = ("X: "+double_to_string(painter_->pointer_pos_.x())+
                 " Y: "+double_to_string(painter_->pointer_pos_.y())+
                 " Z: "+double_to_string(painter_->pointer_pos_.z()));
      double val = 0.0;
      vol->get_value(index, val);
      value = val;
    }
  }


  //  get_vars()->change_parent("value", value, "string", true);
  //get_vars()->change_parent("clut_min_max", clut_min_max, "string", true);
  //get_vars()->change_parent("clut_ww_wl", clut_ww_wl, "string", true);
  //get_vars()->change_parent("xyz_pos", xyz_pos, "string", true);
  //get_vars()->change_parent("sca_pos", sca_pos, "string", true);

    

  setup_gl_view();
  if (autoview_) {
    autoview(painter_->current_volume_);
    setup_gl_view();
  }
  CHECK_OPENGL_ERROR();


  for (unsigned int s = 0; s < slices_.size(); ++s) {
    if (paint_layer_ && slices_[s]->volume_ == paint_layer_->volume_)
      paint_layer_->draw();
    else 
      slices_[s]->draw();
  }

  if (show_grid_()) 
    render_grid();

  painter_->draw_slice_lines(*this);

  event_handle_t redraw_window = new RedrawSliceWindowEvent(*this);
  painter_->tm_.propagate_event(redraw_window);
  render_text();

  //if (painter_->cur_window_ == this) {
  //    Point windowpos(event_.x_, event_.y_, 0);
  //    window.render_guide_lines(windowpos);
  //}


  CHECK_OPENGL_ERROR();
  painter_->volume_lock_.unlock();
  glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  CHECK_OPENGL_ERROR();

  return CONTINUE_E;
}


#if 0
void
SliceWindow::render_vertical_text(FreeTypeTextTexture *text,
                                           double x, double y)
{
  string str = text->get_string();
  int hei = text->height();
  for (unsigned int i = 0; i < str.length(); ++i) 
  {
    text->set(str.substr(i, 1));
    text->draw(x,y, FreeTypeTextTexture::n);
    y -= 2 + hei;
  }
}
#endif


}
