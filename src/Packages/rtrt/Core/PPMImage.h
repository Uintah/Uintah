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



#ifndef RTRT_PPMIMAGE_H
#define RTRT_PPMIMAGE_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array2.h>

#include <Core/Persistent/PersistentSTL.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

#include <cstdio>
#include <cstdlib>

namespace rtrt {
class PPMImage;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::PPMImage&);
template <>
void Pio(Piostream& stream, std::vector<rtrt::Color>& data);
}

namespace rtrt {

using std::vector;
using std::string;

class PPMImage
{

 protected:
  unsigned            u_,v_;
  unsigned            max_;
  bool                valid_;
  vector<rtrt::Color> image_;
  bool                flipped_;

  void eat_comments_and_whitespace(std::ifstream &str);

 public:
  PPMImage() {} // for Pio.
  PPMImage(const string& s, bool flip=false); 
  PPMImage(int nu, int nv, bool flip=false);
  virtual ~PPMImage() {}

  friend void SCIRun::Pio(SCIRun::Piostream&, PPMImage&);

  unsigned get_width() { return u_; }
  unsigned get_height() { return v_; }
  unsigned get_size() { return max_; }
  bool valid() { return valid_; }

  void get_dimensions_and_data(Array2<rtrt::Color> &c, int &nu, int &nv);
  
  rtrt::Color &operator()(unsigned u, unsigned v)
  {
    if (v>=v_) v=v_-1;
    if (u>=u_) u=u_-1;
    return image_[v*u_+u];
  }

  const rtrt::Color &operator()(unsigned u, unsigned v) const
  {
    if (v>=v_) v=v_-1;
    if (u>=u_) u=u_-1;
    return image_[v*u_+u];
  }

  bool write_image(const char* filename, int bin=1);
  bool read_image(const char* filename);
};

} // end namespace

#endif





