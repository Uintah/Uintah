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

#ifndef VOLUMERENDERER_H
#define VOLUMERENDERER_H

#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomObj.h>

#include <Core/Containers/BinaryTree.h>

#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Core/Geom/TextureRenderer.h>


namespace Volume {

using SCIRun::GeomObj;
using SCIRun::DrawInfoOpenGL;
#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
class FragmentProgramARB;
#endif
class VolumeRenderer : public TextureRenderer
{
public:

  enum vol_ren_mode{ OVEROP, MIP, ATTENUATE };

  VolumeRenderer();
  VolumeRenderer(TextureHandle tex, ColorMapHandle map);
  VolumeRenderer(const VolumeRenderer&);
  ~VolumeRenderer(){};
  
  virtual void BuildTransferFunction();
  
  void SetNSlices(int s) { slices_ = s;}
  void SetSliceAlpha( double as){ slice_alpha_ = as;}
  void SetRenderMode( vol_ren_mode vrm) { mode_ = vrm; }

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
  virtual void draw();
  virtual void drawWireFrame();
  virtual void setup();
  virtual void cleanup();
  virtual void load_colormap();
#endif
  
  virtual GeomObj* clone();
  int slices() const { return slices_; }
  double slice_alpha() const { return slice_alpha_; }

protected:
  int               slices_;
  double            slice_alpha_;
  vol_ren_mode      mode_;
  unsigned char     transfer_function_[1024];
  
#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
  FragmentProgramARB *VolShader;
  FragmentProgramARB *FogVolShader;
#endif
  
};

} // End namespace SCIRun


#endif
