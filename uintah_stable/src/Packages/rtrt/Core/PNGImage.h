/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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



#ifndef PNGIMAGE_H
#define PNGIMAGE_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Core/Persistent/PersistentSTL.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace rtrt {
  class PNGImage;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::PNGImage&);
  template <>
  void Pio(Piostream& stream, std::vector<rtrt::Color>& data);
}

namespace rtrt {

  using std::vector;
  using std::string;

class PNGImage
{

 protected:
  unsigned            width_,height_;
  unsigned            max_;
  bool                valid_;
  vector<Color>       image_;
  vector<float>       alpha_;
  bool                flipped_;

 public:
  PNGImage() {} // for Pio.
  PNGImage(const string& s, bool flip=false);
  PNGImage(int nu, int nv, bool flip=false);
  
  virtual ~PNGImage() {}
  
  friend void SCIRun::Pio(SCIRun::Piostream&, PNGImage&);
  
  unsigned get_width() { return width_; }
  unsigned get_height() { return height_; }
  unsigned get_size() { return max_; }
  bool valid() { return valid_; }

  void get_dimensions_and_data(Array2<Color>& c, Array2<float>& d,
			       int& nu, int& nv);

#if 0
  Color& operator()(unsigned u, unsigned v)
  {
    if (v>=height_) v=height_-1;
    if (u>=width_) u=width_-1;
    return image_[v*width_+u];
  }

  const Color& operator()(unsigned u, unsigned v) const
  {
    if (v>=height_) v=height_-1;
    if (u>=width_) u=width_-1;
    return image_[v*width_+u];
  }
#endif

  bool write_ppm(const char* filename, int bin=1);

  bool read_image(const char* filename);
};

} // end namespace

#endif





