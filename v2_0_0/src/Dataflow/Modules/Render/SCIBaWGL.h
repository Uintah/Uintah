/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

#include <GL/gl.h>

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
