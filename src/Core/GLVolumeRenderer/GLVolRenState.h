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

#ifndef GLVOLRENSTATE_H
#define GLVOLRENSTATE_H

#if defined(HAVE_GLEW)
#include <GL/glew.h>
#else
#include <GL/gl.h>
#endif


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
  static FragmentProgramARB *VolShader;
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


