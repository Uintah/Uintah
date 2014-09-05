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
//    File   : ParticleFlowRenderer.h
//    Author : Kurt Zimmerman
//    Date   : Wed March 1, 2006



#ifndef ParticleFlowRenderer_h
#define ParticleFlowRenderer_h

#include <sci_gl.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>

namespace Uintah {
using namespace SCIRun;

#define VELOCITY_ARRAY 5
#define START_TIME_ARRAY 6

class ParticleFlowShader {
public:
  ParticleFlowShader(){}

  virtual ~ParticleFlowShader(){}
  
  typedef enum {
    EVertexShader,
    EFragmentShader,
  } EShaderType;

  GLint getUniLoc(GLuint program, const GLchar *name);
  static void printShaderInfoLog(GLuint shader);
  static void printProgramInfoLog(GLuint program);
  static int shaderSize(char *fileName, EShaderType shaderType);
  static int readShader(char *fileName, EShaderType shaderType,
                        char *shaderText, int size);
  
  static bool shader_functions_built_;
  static bool build_shader_functions();
  int readShaderSource(char *fileName, GLchar **vertexShader,
                       GLchar **fragmentShader);
  int installParticleShaders(const GLchar *particleVertex,
                             const GLchar *particleFragment);

  GLuint ProgramObject;
  GLuint VertexShaderObject;
  GLuint FragmentShaderObject;

private:

};

class ParticleFlowRenderer : public GeomObj
{
public:
  ParticleFlowRenderer();
  ParticleFlowRenderer(const ParticleFlowRenderer &);
  virtual ~ParticleFlowRenderer();

  void update_colormap( ColorMapHandle cmap );
  void update_vector_field( FieldHandle vfh, bool normalize = false );
  void update_time( double time );
  void update_transform(const Point& c, const Point& r, const Point& d);
  void set_animation( bool a ) 
  { if(animating_ != a) reset_ = true;  animating_ = a;}
  void set_time_increment(float i){ time_increment_ = i; }
  
  void createPoints(GLint w, GLint h);
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox& bb){ bb = fh_->mesh()->get_bounding_box();}
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const std::string& format, GeomSave*);
  void updateAnim();
  
  

private:

  static bool functions_initialized_;
  static bool shader_functions_initialized_;
  bool shaders_loaded_;
  bool initialized_;
  bool cmap_dirty_;
  bool flow_tex_dirty_;
  bool animating_;
  bool reset_;
  

  double time_;
  float time_increment_;
  float particle_time_;
  
  unsigned int nx_, ny_, nz_;
  
  GLuint flow_tex_;
  GLfloat *vfield_;
  
  
  ColorMapHandle cmap_h_;
  FieldHandle fh_;

  DrawInfoOpenGL* di_;
  ParticleFlowShader *shader_;

  GLint array_width_, array_height_; // 2D point array size...

  GLfloat *verts_;
  GLfloat *colors_;
  GLfloat *velocities_;
  GLfloat *start_times_;

  Transform shader_trans_;
  
  
  void drawPoints();
  void draw_flow_outline();
  void reload_flow_texture();
  
}; 



} // end namespace Uintah

#endif //ParticleFlowRenderer_h
