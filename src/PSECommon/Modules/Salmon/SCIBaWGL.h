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
protected:
  friend class Salmon;
  friend class Roe;
  Roe* roe;
  BaWGL* bawgl;
  int running, exit;
  
  int pinchID, stylusID;
  
  int pinch, pinchChange;
  int stylus, stylusChange;
  GLfloat pinchMatrix[16], pinchChangeMatrix[16];
  GLfloat stylusMatrix[16], stylusChangeMatrix[16], pos[3];
  GLfloat virtualStylusMatrix[16], virtualStylusChangeMatrix[16];

  GLfloat scaleFrom, scaleOriginal;
  GLfloat velocity, azim, elev, roll, azim0, elev0, fly[16];

  void run();

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
     bool shutting_down;

   public:
     SCIBaWGL( char* );
     ~SCIBaWGL() { }

     int start( Roe* );
     void stop( void );
     void shutdown_ok( void );
 };

}}

#endif


