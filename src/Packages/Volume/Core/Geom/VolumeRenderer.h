//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : VolumeRenderer.h
//    Author : Milan Ikits
//    Date   : Sat Jul 10 11:26:26 2004

#ifndef VolumeRenderer_h
#define VolumeRenderer_h

#include <Core/Thread/Mutex.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomObj.h>

#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Core/Geom/TextureRenderer.h>

namespace Volume {

using SCIRun::GeomObj;
using SCIRun::DrawInfoOpenGL;

class VolShaderFactory;

class VolumeRenderer : public TextureRenderer
{
public:
  VolumeRenderer(TextureHandle tex, ColorMapHandle cmap1, Colormap2Handle cmap2,
                 int tex_mem);
  VolumeRenderer(const VolumeRenderer&);
  ~VolumeRenderer();

  void set_mode(RenderMode mode);
  void set_sampling_rate(double rate);
  void set_interactive_rate(double irate);
  void set_interactive_mode(bool mode);
  void set_adaptive(bool b);
  inline void set_shading(bool shading) { shading_ = shading; }
  inline void set_material(double ambient, double diffuse, double specular, double shine)
  { ambient_ = ambient; diffuse_ = diffuse; specular_ = specular; shine_ = shine; }
  inline void set_light(int light) { light_ = light; }
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
  virtual void draw();
  virtual void draw_wireframe();
#endif
  
  virtual GeomObj* clone();

protected:
  bool shading_;
  double ambient_, diffuse_, specular_, shine_;
  int light_;
  bool adaptive_;
};

} // End namespace SCIRun

#endif // VolumeRenderer_h
