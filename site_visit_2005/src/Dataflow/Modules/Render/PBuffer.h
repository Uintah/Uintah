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

#if (defined(__linux) && !defined(__ECC)) || defined(__APPLE__)
#define HAVE_PBUFFER
#endif

/* minimalistic pbuffer */


namespace SCIRun {

class PBuffer
{
public:
  PBuffer( int doubleBuffer = GL_FALSE );
  ~PBuffer() {}

  // Returns false if the creation fails.
#ifndef _WIN32
  bool create( Display* dpy, int screen, GLXContext sharedcontext,
	       int width, int height, 
	       int colorBits, int depthBits /* 8, 16 */);
#else
  bool create( Display* dpy, int screen, /*GLXContext sharedcontext,*/
	       int width, int height, 
	       int colorBits, int depthBits /* 8, 16 */);
#endif
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

#ifndef _WIN32
  GLXContext cx_;
#endif
#ifdef HAVE_PBUFFER
  GLXFBConfig* fbc_;
  GLXPbuffer pbuffer_;  //win_
#endif
  Display* dpy_;
  int screen_;
};

} // end namespace SCIRun

#endif // SCIRUN_PBUFFER_H
