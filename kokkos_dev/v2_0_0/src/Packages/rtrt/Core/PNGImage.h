
#ifndef PNGIMAGE_H
#define PNGIMAGE_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array2.h>
#include <vector>
#include <Core/Persistent/PersistentSTL.h>

#include <string>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include "png.h"


namespace rtrt {
class PNGImage;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::PNGImage&);
template <>
void Pio(Piostream& stream, std::vector<rtrt::Color>& data);
}

namespace rtrt {

using std::ifstream;
using std::ofstream;
using std::cout;
using std::cerr;
using std::vector;
using std::string;

class PNGImage
{

 protected:
  unsigned            u_,v_;
  unsigned            max_;
  bool                valid_;
  vector<rtrt::Color> image_;
  vector<float>       alpha_;
  bool                flipped_;

  void eat_comments_and_whitespace(ifstream &str);

 public:
  PNGImage() {} // for Pio.
  
  PNGImage(const string& s, bool flip=false) 
    : flipped_(flip) 
  { 
    
    valid_ = read_image(s.c_str()); //read also the alpha mask

  }
  
  PNGImage(int nu, int nv, bool flip=false) 
    : u_(nu), v_(nv), valid_(false), flipped_(flip) 
  {
    image_.resize(u_*v_);
    alpha_.resize(u_*v_);
  }
  
  virtual ~PNGImage() {}
  
  friend void SCIRun::Pio(SCIRun::Piostream&, PNGImage&);
  
  unsigned get_width() { return u_; }
  unsigned get_height() { return v_; }
  unsigned get_size() { return max_; }

  void get_dimensions_and_data(Array2<rtrt::Color> &c, Array2<float> &d, int &nu, int &nv) {
    if (valid_) {
      c.resize(u_+2,v_+2);  // image size + slop for interpolation
      d.resize(u_+2, v_+2);
      nu=u_;
      nv=v_;
      for (unsigned v=0; v<v_; ++v)
        for (unsigned u=0; u<u_; ++u)
	  {
	    c(u,v)=image_[v*u_+u];
	    d(u,v)=alpha_[v*u_+u];
	  }
      
    } else {
      c.resize(0,0);
      d.resize(0,0);
      nu=0;
      nv=0;
    }
  }

  bool valid() { return valid_; }

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

  
  /*float operator()(unsigned u, unsigned v)
  {
    if (v>=v_) v=v_-1;
    if (u>=u_) u=u_-1;
    return alpha_[v*u_+u];
    }*/

    


  bool write_image(const char* filename, int bin=1)
  {
    ofstream outdata(filename);
    if (!outdata.is_open()) {
      cerr << "PPMImage: ERROR: I/O fault: couldn't write image file: "
	   << filename << "\n";
      return false;
    }
    if (bin)
      outdata << "P6\n# PPM binary image created with rtrt\n";
    else
      outdata << "P3\n# PPM ASCII image created with rtrt\n";

    outdata << u_ << " " << v_ << "\n";
    outdata << "255\n";

    unsigned char c[3];
    if (bin) {
      for(unsigned v=0;v<v_;++v){
	for(unsigned u=0;u<u_;++u){
	  c[0]=(unsigned char)(image_[v*u_+u].red()*255);
	  c[1]=(unsigned char)(image_[v*u_+u].green()*255);
	  c[2]=(unsigned char)(image_[v*u_+u].blue()*255);
	  outdata.write((char *)c, 3);
	}
      }
    } else {
      int count=0;
      for(unsigned v=0;v<v_;++v){
	for(unsigned u=0;u<u_;++u, ++count){
	  if (count == 5) { outdata << "\n"; count=0; }
	  outdata << (int)(image_[v*u_+u].red()*255) << " ";
	  outdata << (int)(image_[v*u_+u].green()*255) << " ";
	  outdata << (int)(image_[v*u_+u].blue()*255) << " ";
	}
      }
    }
    return true;
    }

  bool read_image(const char* filename);
};

} // end namespace

#endif





