
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





