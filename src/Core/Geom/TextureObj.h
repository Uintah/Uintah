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


/*
 *  TextureObj.h
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   January, 2006
 *
 *  Copyright (C) 2006 SCI Group
 */

#ifndef SCIRun_Dataflow_Modules_Render_TextureObj_h
#define SCIRun_Dataflow_Modules_Render_TextureObj_h

#include <Core/Datatypes/NrrdData.h>
#include <string>

namespace SCIRun {

using std::string;

#include <Core/Geom/share.h>
class SCISHARE TextureObj {
public:
  TextureObj(NrrdDataHandle &nrrd_handle);
  TextureObj(int, int, int);

  ~TextureObj();
  //  void                  draw(int n, float *vertices, float *tex_coords);
  void                  draw(int n, Point *vertices, float *tex_coords);
  void                  set_nrrd(NrrdDataHandle &nrrd_handle);
  void			set_color(float, float, float, float);
  void			set_color(float rgba[4]);
  int			width() { return width_; };
  int		        height() { return height_; };
  void                  set_dirty() { dirty_ = true; }
  NrrdDataHandle	nrrd_handle_;
  bool			bind();
  unsigned int          tex_id() { return texture_id_; }
private:

  void			pad_to_power_of_2();
  
  int			width_;
  int			height_;
  float 		color_[4];
  bool			dirty_;
  unsigned int		texture_id_;
};

}

  
#endif
