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
class VertexProgramARB;
class FragmentProgramARB;
class Pbuffer;
class CM2ShaderFactory;

class VolumeRenderer : public TextureRenderer
{
public:

  enum vol_ren_mode{ OVEROP, MIP };

  VolumeRenderer();
  VolumeRenderer(TextureHandle tex, ColorMapHandle map, Colormap2Handle cmap2);
  VolumeRenderer(const VolumeRenderer&);
  ~VolumeRenderer(){};
  
  virtual void BuildTransferFunction();
  virtual void BuildTransferFunction2();
  
  void SetNSlices(int s) { slices_ = s; }
  void SetSliceAlpha(double as){ slice_alpha_ = as;}
  void SetRenderMode(vol_ren_mode vrm) { mode_ = vrm; }

  inline void setShading(bool shading) { shading_ = shading; }
  inline void setMaterial(double ambient, double diffuse, double specular, double shine)
  { ambient_ = ambient; diffuse_ = diffuse; specular_ = specular; shine_ = shine; }
  inline void setLight(int light) { light_ = light; }
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
  virtual void draw();
  virtual void drawWireFrame();
  virtual void load_colormap();
#endif
  
  virtual GeomObj* clone();
  int slices() const { return slices_; }
  double slice_alpha() const { return slice_alpha_; }

protected:
  int slices_;
  double slice_alpha_;
  vol_ren_mode mode_;
  unsigned char transfer_function_[1024];

  bool shading_;
  double ambient_, diffuse_, specular_, shine_;
  int light_;
  
  FragmentProgramARB* VolShader1;
  FragmentProgramARB* VolShader4;
  FragmentProgramARB* FogVolShader1;
  FragmentProgramARB* FogVolShader4;
  FragmentProgramARB* LitVolShader;
  FragmentProgramARB* LitFogVolShader;
  FragmentProgramARB* VolShader1_2;
  FragmentProgramARB* VolShader4_2;
  FragmentProgramARB* FogVolShader1_2;
  FragmentProgramARB* FogVolShader4_2;
  FragmentProgramARB* LitVolShader_2;
  FragmentProgramARB* LitFogVolShader_2;
  VertexProgramARB* FogVertexShader;
  Pbuffer* pbuffer_;
  CM2ShaderFactory* shader_factory_;
  Pbuffer* texbuffer_;
  FragmentProgramARB* texshader_;
  bool use_pbuffer_;
};

} // End namespace SCIRun


#endif
