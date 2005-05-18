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

#include <Core/Volume/Texture.h>
#include <Core/Volume/TextureRenderer.h>

namespace SCIRun {

class VolShaderFactory;

class VolumeRenderer : public TextureRenderer
{
public:
  VolumeRenderer(TextureHandle tex, ColorMapHandle cmap1, ColorMap2Handle cmap2,
                 int tex_mem);
  VolumeRenderer(const VolumeRenderer&);
  ~VolumeRenderer();

  void set_mode(RenderMode mode);
  void set_sampling_rate(double rate);
  void set_interactive_rate(double irate);
  void set_interactive_mode(bool mode);
  void set_adaptive(bool b);
  inline void set_gradient_range(double min, double max) { 
    grange_ = 1/(max-min); goffset_ = -min/(max-min); 
  }
  inline void set_shading(bool shading) { shading_ = shading; }
  inline void set_material(double ambient, double diffuse, double specular, double shine)
  { ambient_ = ambient; diffuse_ = diffuse; specular_ = specular; shine_ = shine; }
  inline void set_light(int light) { light_ = light; }
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);

  void draw_wireframe();
  void multi_level_draw();
  void draw_volume();
#endif

  double num_slices_to_rate(int slices);
  
  virtual GeomObj* clone();
  void set_draw_level( int i, bool b){ draw_level_[i] = b; }
  void set_level_alpha(int i, double v) { level_alpha_[i] = v; }
protected:
  double grange_, goffset_;
  bool shading_;
  double ambient_, diffuse_, specular_, shine_;
  int light_;
  bool adaptive_;
  vector< bool > draw_level_;
  vector< double > level_alpha_;

};

} // End namespace SCIRun

#endif // VolumeRenderer_h
