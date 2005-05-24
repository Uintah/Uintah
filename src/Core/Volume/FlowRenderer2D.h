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
//    File   : FlowRenderer2D.h
//    Author : Kurt Zimmerman
//    Date   : Fri Apr 20 13:00:00 2005

#ifndef FlowRenderer2D_h
#define FlowRenderer2D_h

#include <Core/Containers/Array2.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Ray.h>
#include <Core/Thread/Mutex.h>

namespace SCIRun {

class Pbuffer;
class FragmentProgramARB;
template <class Data> class ImageField;

class FlowRenderer2D : public GeomObj
{

public:
  enum FlowMode { MODE_NONE, MODE_LIC, MODE_IBFV, MODE_LEA };

  FlowRenderer2D( FieldHandle vfield, ColorMapHandle cmap, int tex_mem);
  FlowRenderer2D(const FlowRenderer2D&);
  ~FlowRenderer2D();

  void init();
  void set_field(FieldHandle f);
  void set_colormap(ColorMapHandle cmap);
  void set_mode(FlowMode mode);
  void set_sw_raster(bool b);
  bool use_pbuffer() { return use_pbuffer_; }
  void set_blend_num_bits(int b);
  bool use_blend_buffer();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);

  void draw_wireframe();
  void draw();
#endif
  virtual GeomObj* clone();
  virtual void get_bounds(BBox& bb);
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  int w_, h_;
  FieldHandle field_;
  ImageField<Vector> *ifv_;
  ColorMapHandle cmap_;
  Mutex mutex_;
  FlowMode mode_;
  bool shading_;
  bool sw_raster_;
  DrawInfoOpenGL* di_;

#ifdef SCI_OPENGL
  bool pbuffers_created_;


  Array2<float> cmap_array_;
  unsigned int cmap_tex_;
  bool cmap_dirty_;

  Array3<float> adv_array_;
  unsigned int adv_tex_;
  bool adv_dirty_;
  bool adv_is_initialized_;
  Array3<float> conv_array_;
  unsigned int conv_tex_;
  bool conv_dirty_;
  bool conv_is_initialized_;

  Array3<float> flow_array_;
  unsigned int flow_tex_;
  bool flow_dirty_;
  bool re_accum_;

  Array3<float> noise_array_;
  unsigned int noise_tex_;
  bool build_noise_;
  
  bool use_pbuffer_;
  int buffer_width_;
  int buffer_height_;

  Pbuffer* adv_buffer_;
  Pbuffer* noise_buffer_;
  Pbuffer* blend_buffer_;

  int blend_num_bits_;
  bool use_blend_buffer_;
  
  int free_tex_mem_;

  vector<pair<float,float> > shift_list_;
  int current_shift_;
 
  FragmentProgramARB *conv_init_;
  FragmentProgramARB *adv_init_;
  FragmentProgramARB *conv_accum_;
  FragmentProgramARB *adv_accum_;
  FragmentProgramARB *conv_rewire_;
  FragmentProgramARB *adv_rewire_;
  FragmentProgramARB *conv_init_rect_;
  FragmentProgramARB *adv_init_rect_;
  FragmentProgramARB *conv_accum_rect_;
  FragmentProgramARB *adv_accum_rect_;
  FragmentProgramARB *conv_rewire_rect_;
  FragmentProgramARB *adv_rewire_rect_;

  bool is_initialized_;

  float tex_coords_[8];
  float pos_coords_[12];
  

  Ray compute_view();
  void create_pbuffers(int w, int h);
  void build_colormap();
  void bind_colormap();
  void release_colormap();
  void build_flow_tex();
  void bind_flow_tex();
  void release_flow_tex();
  void build_noise();
  void rebuild_noise(){ build_noise_ = true; }
  void bind_noise();
  void release_noise();
  void build_adv(float scale,pair<float, float>& shift);
  void load_adv();
  void bind_adv();
  void release_adv();
  void build_conv(float scale);
  void load_conv();
  void bind_conv();
  void release_conv();
  void next_shift(int *shft);

  void adv_init( Pbuffer*& pb, float scale,
                 pair<float, float>& shift);

  void adv_accum( float pixel_x, float pixel_y,
                  float scale, pair<float,float>&  shift);
  void adv_rewire();
  void conv_accum( float pixel_x, float pixel_y, float scale);
  void conv_rewire();
  float get_interpolated_value( Array3<float>& array, float x, float y );
#endif
  
public:
  bool auto_tex_;
  float tex_x_, tex_y_;
  void set_auto_tex( bool b ){ auto_tex_ = b; }
  bool get_auto_tex() { return auto_tex_; }
  void set_tex_x( float f ) { tex_x_ = f; }
  void set_tex_y( float f ) { tex_y_ = f; }
  float get_tex_x() { return tex_x_; }
  float get_tex_y() { return tex_y_; }
};

} // End namespace SCIRun

#endif // FlowRenderer2D_h
