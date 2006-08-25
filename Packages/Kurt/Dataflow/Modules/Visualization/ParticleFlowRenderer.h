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

#include <Core/Datatypes/Field.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/Fbuffer.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>

#include <sci_gl.h>

#include <Packages/Kurt/Dataflow/Modules/Visualization/GLSLShader.h>


namespace Kurt {
using namespace SCIRun;

#define VELOCITY_ARRAY 5
#define START_POSITION_ARRAY 6


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
  void set_animation( bool a ) { animating_ = a;}
  void set_time_increment(float i){ time_increment_ = i; }
  void set_nsteps( unsigned int n) { nsteps_ = n; }
  void set_step_size( double d ) { step_size_ = d; }
  void recompute_points( bool r ){ recompute_ = r; }
  void set_particle_exponent( int e );
  void freeze( bool f ){ frozen_ = f; }
  void create_points(GLint w, GLint h);
  void reset(){ reset_ = true; }
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
  bool vshader_initialized_;
  bool cmap_dirty_;
  bool flow_tex_dirty_;
  bool part_tex_dirty_;
  bool animating_;
  bool reset_;
  bool recompute_;
  bool frozen_;

  double time_;
  float time_increment_;
  
  unsigned int nx_, ny_, nz_;
  unsigned int nsteps_;
  double step_size_;

  
  GLuint flow_tex_;
  GLuint part_tex_[2];
  GLfloat *vfield_;
  GLuint vb_;
  
  ColorMapHandle cmap_h_;
  FieldHandle fh_;

  DrawInfoOpenGL* di_;
  GLSLShader *shader_;
  GLSLShader *vshader_;
  

  int particle_power_;
  GLint array_width_, array_height_; // 2D point array size...

  GLfloat *start_verts_; //orginal vertices
  GLfloat *current_verts_; // current vertices
  GLfloat *colors_;
  GLfloat *velocities_;
  GLfloat *start_times_;

  Transform shader_trans_;
  Transform initial_particle_trans_;
  Fbuffer *fb_;
  GLuint rb_id_;
  GLint current_draw_buffer_;
  GLint start_pos_;
  GLint pos_;
  GLint flow_;
  
  bool have_shader_functions();
  bool Fbuffer_configure();
  bool shader_configure();
  bool vshader_configure();
  void draw_points();
  void draw_flow_outline();
  void reload_flow_texture();
  void load_part_texture(GLuint unit = 0, float *verts = 0);
  

  
}; 



} // end namespace Kurt

#endif //ParticleFlowRenderer_h
