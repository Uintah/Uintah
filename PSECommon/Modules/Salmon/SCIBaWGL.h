/*
 *  Bench.h: The Immersive WorkBench Interface
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

#include <GL/gl.h>
#include <ulocks.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Thread/Runnable.h>
#include <PSECommon/Modules/Salmon/Roe.h>
#include <PSECommon/Modules/Salmon/BaWGL.h>

namespace PSECommon {
namespace Modules {

using SCICore::Thread::Runnable;
using SCICore::Thread::Thread;

class SCIBaWGL;

class SCIBaWGLTimer : public Runnable {
public:
  friend class Salmon;
  friend class Roe;
  Roe* roe;
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
  SCIBaWGLTimer( Roe*, SCIBaWGL* );
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

     int start( Roe*, char* );
     void stop( void );
     void shutdown_ok( void );
 };

}}

#endif

#endif


