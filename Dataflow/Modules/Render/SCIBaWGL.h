/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
 *  SCIBawGL.h: The Immersive WorkBench Interface
 *
 *  Written by:
 *   Milan Ikits
 *   Department of Computer Science
 *   University of Utah
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_project_module_SCIBaWGL_h
#define SCI_project_module_SCIBaWGL_h

#ifdef __sgi
#include <ulocks.h>
#endif

#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Dataflow/Modules/Render/ViewWindow.h>
#include <Dataflow/Modules/Render/BaWGL.h>

namespace SCIRun {


class SCIBaWGL;

class SCIBaWGLTimer : public Runnable {
public:
  friend class Viewer;
  friend class ViewWindow;
  ViewWindow* viewwindow;
  SCIBaWGL* bawgl;
  volatile int running, exit;
  
  int pinchID, stylusID;
  
  int pinch, pinch0, pinchChange;
  int stylus, stylus0, stylusChange;

  GLfloat realStylusMatrix[16], realStylus0Matrix[16];
  GLfloat virtualStylusMatrix[16], virtualStylusChangeMatrix[16];

  GLfloat realPinchMatrix[16], realPinch0Matrix[16];
  GLfloat surfacePinchMatrix[16];
  GLfloat virtualPinchMatrix[16];
  GLfloat virtualPinchChangeMatrix[16];
  
  GLfloat scaleFrom, scaleOriginal, navigateFrom;
  GLfloat velocity, azim, elev, roll, azim0, elev0, roll0, fly[16];

  GLfloat navSpeed;

  void run();
  void navigate();
  void pick();

public:
  SCIBaWGLTimer( ViewWindow*, SCIBaWGL* );
  ~SCIBaWGLTimer() { }

  void start( void );
  void stop( void );
  void quit( void );
};

class SCIBaWGL : public BaWGL {

   protected:
     SCIBaWGLTimer* timer;
     Thread* timerthread;
     
   public:
     SCIBaWGL( void );
     ~SCIBaWGL() { }

     bool shutting_down, redraw_enable;
     int pick;
     int scale, navigate;
     GLfloat scaleFrom, navigateFrom, velocity;

     int start( ViewWindow*, char* );
     void stop( void );
     void shutdown_ok( void );
 };

} // End namespace SCIRun

#endif
