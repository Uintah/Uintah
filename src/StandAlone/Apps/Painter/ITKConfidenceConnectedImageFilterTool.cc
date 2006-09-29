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
//    File   : ITKConfidenceConnectedImageFilterTool.cc
//    Author : McKay Davis
//    Date   : Tue Sep 26 18:44:34 2006

#include <StandAlone/Apps/Painter/Painter.h>
#include <sci_comp_warn_fixes.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <typeinfo>
#include <iostream>
#include <sci_gl.h>
#include <Core/Bundle/Bundle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/FreeTypeTextTexture.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Events/keysyms.h>

#ifdef HAVE_INSIGHT
#  include <itkConfidenceConnectedImageFilter.h>
#  include <itkThresholdSegmentationLevelSetImageFilter.h>
#endif


#ifdef _WIN32
#  define SCISHARE __declspec(dllimport)
#else
#  define SCISHARE
#endif

namespace SCIRun {

Painter::ITKConfidenceConnectedImageFilterTool::ITKConfidenceConnectedImageFilterTool(Painter *painter) :
  BaseTool("ITK Confidence Connected\nImage Filter"),
  PainterPointerTool(painter, "ITK Confidence Connected\nImage Filter"),
  seed_(),
  volume_(0),
  prefix_("ITKConfidenceConnectedImageFilterTool::"),
  numberOfIterations_(painter_->get_vars(), prefix_+"numberOfIterations"),
  multiplier_(painter_->get_vars(), prefix_+"multiplier"),
  replaceValue_(painter_->get_vars(), prefix_+"replaceValue"),
  initialNeighborhoodRadius_(painter_->get_vars(), 
                             prefix_+"initialNeighborhoodRadius")
{
}


BaseTool::propagation_state_e
Painter::ITKConfidenceConnectedImageFilterTool::pointer_down
(int b, int x, int y, unsigned int m, int t)
{
  BaseTool::propagation_state_e state = pointer_motion(b,x,y,m,t);

  return state;
}

BaseTool::propagation_state_e
Painter::ITKConfidenceConnectedImageFilterTool::pointer_up
(int b, int x, int y, unsigned int m, int t)
{
  return pointer_motion(b,x,y,m,t);
}


void
Painter::ITKConfidenceConnectedImageFilterTool::finish() {
  if (!volume_) 
    return;

  if (!volume_->index_valid(seed_))
    return;

#ifdef HAVE_INSIGHT 

#if 0
  painter_->get_vars()->insert("ToolDialog::text", 
                     " ITK Confidence Connected Filter Running...",
                     "string", true);

  painter_->redraw_all();
#endif
  
  typedef itk::ConfidenceConnectedImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D > FilterType;
  FilterType::Pointer filter = FilterType::New();
  FilterType::IndexType seed_point;
  for(unsigned int i = 0; i < seed_point.GetIndexDimension(); i++) {
    seed_point[i] = seed_[i+1];
  }
  
  filter->SetSeed(seed_point);
  filter->SetNumberOfIterations(numberOfIterations_());
  filter->SetMultiplier(multiplier_());
  filter->SetReplaceValue(replaceValue_());
  filter->SetInitialNeighborhoodRadius(initialNeighborhoodRadius_());

  string name = "Confidence Connected";
  NrrdVolume *temp = new NrrdVolume(volume_, name, 2);
  painter_->volume_map_[name] = temp;
  temp->colormap_ = 1;
  temp->clut_min_ = temp->data_min_ = 0.5;
  temp->clut_max_ = temp->data_max_ = 1.0;
  painter_->current_volume_ = temp;

  painter_->do_itk_filter<Painter::ITKImageFloat3D>(filter, 
                                                    temp->nrrd_handle_);
  painter_->show_volume(name);
  painter_->recompute_volume_list();

  painter_->set_all_slices_tex_dirty();
  painter_->redraw_all();
#endif
}

BaseTool::propagation_state_e
Painter::ITKConfidenceConnectedImageFilterTool::pointer_motion
(int b, int x, int y, unsigned int m, int t)
{
  if (b == 1 && !m) {
    if (!volume_) 
      volume_ = painter_->current_volume_;
    if (volume_) {
      vector<int> newseed = volume_->world_to_index(painter_->pointer_pos_);
      if (volume_->index_valid(newseed)) 
        seed_ = newseed;

      painter_->redraw_all();
      return STOP_E;
    }
  }
  return CONTINUE_E;

}



BaseTool::propagation_state_e 
Painter::ITKConfidenceConnectedImageFilterTool::process_event
(event_handle_t event)
{
  RedrawSliceWindowEvent *redraw = 
    dynamic_cast<RedrawSliceWindowEvent *>(event.get_rep());
  if (redraw) {
    draw_gl(redraw->get_window());
  }

  if (dynamic_cast<FinishEvent *>(event.get_rep())) {
    finish();
  }

  if (dynamic_cast<QuitEvent *>(event.get_rep())) {
    return QUIT_AND_STOP_E;
  }
 
  return CONTINUE_E;
}
  

void
Painter::ITKConfidenceConnectedImageFilterTool::draw_gl(Painter::SliceWindow &window)
{
  if (!volume_ || !volume_->index_valid(seed_)) return;

  vector<double> index(seed_.size());
  index[0] = seed_[0];
  for (unsigned int s = 1; s < index.size(); ++s)
    index[s] = seed_[s]+0.5;

  Vector left = window.x_dir();
  Vector up = window.y_dir();
  Point center = volume_->index_to_point(index);
  Point p;

  //  double one = 100.0 / window.zoom_; // world space units per one pixel
  double units = window.zoom_ / 100.0;  // Pixels per world space unit
  double s = units/2.0;
  double e = s+Clamp(s, 5.0, Max(units, 5.0));

  for (int pass = 0; pass < 3; ++pass) {
    glLineWidth(5 - pass*2.0);
    if (pass == 0)
      glColor4d(0.0, 0.0, 0.0, 1.0);
    else if (pass == 1)
      glColor4d(1.0, 0.0, 0.0, 1.0);
    else
      glColor4d(1.0, 0.7, 0.6, 1.0);

    glBegin(GL_LINES);    
    p = center + s * up;
    glVertex3dv(&p(0));
    p = center + e * up;
    glVertex3dv(&p(0));
    
    p = center - s * up;
    glVertex3dv(&p(0));
    p = center - e * up;
    glVertex3dv(&p(0));
    
    p = center + s * left;
    glVertex3dv(&p(0));
    p = center + e * left;
    glVertex3dv(&p(0));
    
    p = center - s * left;
    glVertex3dv(&p(0));
    p = center - e * left;
    glVertex3dv(&p(0));
    glEnd();
    CHECK_OPENGL_ERROR();
  }

  glLineWidth(1.0);
}

}
