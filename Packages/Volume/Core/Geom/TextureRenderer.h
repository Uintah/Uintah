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
//    File   : TextureRenderer.h
//    Author : Milan Ikits
//    Date   : Wed Jul  7 23:34:33 2004

#ifndef TextureRenderer_h
#define TextureRenderer_h

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
#include <Packages/Volume/Core/Datatypes/Colormap2.h>

namespace Volume {

using SCIRun::GeomObj;
using SCIRun::DrawInfoOpenGL;
using SCIRun::ColorMap;
using SCIRun::Mutex;

class TextureRenderer : public GeomObj
{
public:

  TextureRenderer();
  TextureRenderer(TextureHandle tex, ColorMapHandle map, Colormap2Handle cmap2);
  TextureRenderer(const TextureRenderer&);
  virtual ~TextureRenderer();

  virtual void BuildTransferFunction() = 0;
  virtual void BuildTransferFunction2() = 0;

  void SetTexture( TextureHandle tex ) {
    mutex_.lock(); tex_ = tex; new_texture_ = true; mutex_.unlock();
  }

  void SetColorMap( ColorMapHandle cmap){
    mutex_.lock(); cmap_ = cmap; //BuildTransferFunction(); 
    cmap_has_changed_ = true; mutex_.unlock();
  }

  void SetColormap2(Colormap2Handle cmap2) {
    mutex_.lock();
    cmap2_ = cmap2;
    cmap2_dirty_ = true;
    mutex_.unlock();
  }
  
  void SetInterp( bool i) { interp_ = i;}

  void set_bounding_box(BBox &bb) { bounding_box_ = bb; }


#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time) = 0;
  virtual void draw() = 0;
  virtual void drawWireFrame() = 0;
  virtual void load_colormap() = 0;

protected:
  void compute_view(Ray& ray);
  void load_texture( Brick& b);
  void make_texture_matrix( const Brick& b);
  void drawPolys( vector<Polygon *> polys );
  void enable_tex_coords();
  void disable_tex_coords();

  unsigned int cmap_texture_;

  Array3<unsigned char> cmap2_array_;
  unsigned int cmap2_texture_;
#endif

public:
  virtual GeomObj* clone() = 0;
  virtual void get_bounds(BBox& bb){ tex_->get_bounds( bb ); }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);

  TextureHandle texH() const { return tex_; }
  DrawInfoOpenGL* di() const { return di_; }

  bool interp() const { return interp_; }

protected:
  TextureHandle         tex_;
  Mutex                 mutex_;
  ColorMapHandle        cmap_;
  Colormap2Handle       cmap2_;

  bool                  new_texture_;
  bool                  cmap_has_changed_;
  bool                  cmap2_dirty_;
  DrawInfoOpenGL       *di_;
 
  bool                  interp_;
  int                   lighting_;
  
  bool                  reload_;

  BBox                  bounding_box_;

  static int            r_count_;
};

} // end namespace Volume

#endif // TextureRenderer_h
