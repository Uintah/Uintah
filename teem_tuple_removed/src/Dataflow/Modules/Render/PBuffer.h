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
 *  PBuffer.h: Render geometry to a pbuffer using opengl
 *
 *  Written by:
 *   Kurt Zimmerman and Milan Ikits
 *   Department of Computer Science
 *   University of Utah
 *   December 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef SCIRUN_PBUFFER_H
#define SCIRUN_PBUFFER_H

#include <GL/gl.h>
#include <GL/glx.h>


/* minimalistic pbuffer */


namespace SCIRun {

class PBuffer
{
public:
  PBuffer( int doubleBuffer = GL_FALSE );
  ~PBuffer() {}

  void create( Display* dpy, int screen,
	       int width, int height, 
	       int colorBits, int depthBits /* 8, 16 */);
  void destroy();
  void makeCurrent();
  inline bool is_valid(){ return valid_; }
  bool is_current();

  inline int width() { return width_; }
  inline int height() { return height_; }

private:
  int width_, height_;
  int colorBits_;
  int doubleBuffer_;
  int depthBits_;
  bool valid_;

  GLXContext cx_;
  GLXFBConfig* fbc_;
  GLXPbuffer pbuffer_;  //win_
  Display* dpy_;
  int screen_;
};

} // end namespace SCIRun

#endif // SCIRUN_PBUFFER_H
