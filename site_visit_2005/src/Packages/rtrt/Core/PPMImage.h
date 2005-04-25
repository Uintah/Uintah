
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

#include <stdio.h>
#include <stdlib.h>

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





