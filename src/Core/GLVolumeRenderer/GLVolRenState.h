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


#ifndef GLVOLRENSTATE_H
#define GLVOLRENSTATE_H

#include <sci_gl.h>

#include <Core/Geometry/Ray.h>
#include <Core/Geometry/BBox.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <iostream>

namespace SCIRun {

using std::cerr;
class Polygon;
class Brick; 
using std::vector;
  
/**************************************
					 
CLASS
   GLVolRenState
   
   GLVolRenState Class.

GENERAL INFORMATION

   GLVolRenState.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   GLVolRenState

DESCRIPTION
   GLVolRenState class.  Use subclasses to implement the drawing State
   for the VolumeRenderer.
  
WARNING
  
****************************************/
class GLVolumeRenderer;
#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
class FragmentProgramARB;
#endif

class GLVolRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  GLVolRenState(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~GLVolRenState(){}
  // GROUP: Operations
  //////////
  // draw the Volume
  virtual void draw() = 0;
  // draw Wireframe
  virtual void drawWireFrame() = 0;
  
  void Reload(){ reload_ = true;}

  void NewBricks(){ newbricks_ = true; }
  void NewColorMap(){ newcmap_ = true; }

  void set_bounding_box(BBox &bb) { bounding_box_ = bb; }
protected:

  virtual void setAlpha(const Brick& brick) = 0;
  void computeView(Ray&);
  void loadColorMap(Brick&);
  void loadTexture( Brick& brick);
  void makeTextureMatrix(const Brick& brick);
  void enableTexCoords();
  void enableBlend();
  void drawPolys( vector<Polygon *> polys);
  void disableTexCoords();
  void disableBlend();
  void drawWireFrame(const Brick& brick);
  const GLVolumeRenderer*  volren;

  GLuint* texName;
  vector<GLuint> textureNames;
  bool reload_;
  bool newbricks_;
  bool newcmap_;

  BBox bounding_box_;

#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
  FragmentProgramARB *VolShader;
#endif
  GLuint cmap_texture_;

};


#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
class FragmentProgramARB
{
public:
  FragmentProgramARB (const char* str, bool isFileName = false);
  ~FragmentProgramARB ();
  void init( const char* str, bool isFileName );

  bool created();
  void create ();
  void update ();
  void destroy ();

  void bind ();
  void release ();
  void enable ();
  void disable ();
  void makeCurrent ();
  
protected:
  unsigned int mId;
  unsigned char* mBuffer;
  unsigned int mLength;
  char* mFile;
};
#endif

} // End namespace SCIRun

#endif


