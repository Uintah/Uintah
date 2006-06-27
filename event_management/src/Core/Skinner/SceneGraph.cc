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
//    File   : SceneGraph.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:03:01 2006

#include <Core/Skinner/SceneGraph.h>
#include <Core/Math/MiscMath.h>

namespace SCIRun {


Skinner::SceneGraph::SceneGraph(Variables *variables) :
  OpenGLViewer(0), 
  Skinner::Drawable(variables)
{
}

Skinner::SceneGraph::~SceneGraph() 
{
}

void
Skinner::SceneGraph::need_redraw() {
  EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
  OpenGLViewer::need_redraw();
}
  

int
Skinner::SceneGraph::width() const {
  return Ceil(region_.width());
}

int
Skinner::SceneGraph::height() const {
  return Ceil(region_.height());
}



BaseTool::propagation_state_e
Skinner::SceneGraph::process_event(event_handle_t event) {
  event.detach();
  PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
  if (pointer) {
    if (!region_.inside(pointer->get_x(), pointer->get_y())) {
      return CONTINUE_E;
    }
    pointer->set_x(pointer->get_x() - Floor(region_.x1()));
    pointer->set_y(Ceil(region_.y2()) - pointer->get_y());
  }

  tm_.propagate_event(event);

  WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
  if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
     GLint viewport[4];
     glGetIntegerv(GL_VIEWPORT, viewport);    
     glViewport(Floor(region_.x1()), Floor(region_.y1()), 
                Ceil(region_.width()), Ceil(region_.height()));

     glMatrixMode(GL_MODELVIEW);
     glPushMatrix();
     glLoadIdentity();
     glMatrixMode(GL_PROJECTION);
     glPushMatrix();
     glLoadIdentity();
     glDisable(GL_BLEND);

     redraw_frame();
     //save_frame();

     glDisable(GL_LIGHTING);
     glEnable(GL_BLEND);
     glDisable(GL_CULL_FACE);
     glDisable(GL_DEPTH_TEST);
     glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
     glMatrixMode(GL_MODELVIEW);
     glPopMatrix();
     glMatrixMode(GL_PROJECTION);
     glPopMatrix();    
  }
  
  return CONTINUE_E;
}



Skinner::Drawable *
Skinner::SceneGraph::maker(Variables *variables,
                           const Skinner::Drawables_t &child,
                           void *) 
{
  ASSERT(child.empty());
  return new SceneGraph(variables);
}


}
