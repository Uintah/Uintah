
#ifndef PPMIMAGE_H
#define PPMIMAGE_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array2.h>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>

namespace rtrt {

static void eat_comments_and_whitespace(ifstream &str)
{
  char c;
  str.get(c);
  for(;;) {
    if (c==' '||c=='\t'||c=='\n') {
      str.get(c);
      continue;
    } else if (c=='#') {
      str.get(c);
      while(c!='\n')
        str.get(c);
    } else {
      str.unget();
      break;
    }
  }
}

class PPMImage
{

 protected:
  unsigned            u_,v_;
  unsigned            max_;
  bool                valid_;
  vector<rtrt::Color> image_;
  bool                flipped_;

 public:
  PPMImage(const string& s) : valid_(false), flipped_(false) 
  { 
    read_image(s.c_str()); 
  }
  PPMImage(int nu, int nv) : u_(nu), v_(nv), valid_(false), flipped_(false) 
  {
    image_.resize(u_*v_);
  }

  virtual ~PPMImage() {}

  unsigned get_width() { return u_; }
  unsigned get_height() { return v_; }
  unsigned get_size() { return max_; }

  void get_dimensions_and_data(Array2<rtrt::Color> &c, int &nu, int &nv, 
                               bool flipped=false) {
    c.resize(u_+2,v_+2);  // image size + slop for interpolation
    nu=u_;
    nv=v_;
    flipped_ = flipped;
    if (!flipped_) {
      for (unsigned u=0; u<u_; ++u)
        for (unsigned v=0; v<v_; ++v)
          c(u,v)=image_[v*u_+u];
    } else {
      for (unsigned u=0; u<u_; ++u)
        for (unsigned v=0; v<v_; ++v)
          c(u,v_-v-1)=image_[v*u_+u];
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

  bool write_image(const char* filename, int bin=1)
  {
    ofstream outdata(filename);
    if (!outdata.is_open()) {
      cerr << "PPMImage: ERROR: I/O fault: couldn't write image file: "
	   << filename << endl;
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

  bool read_image(const char* filename)
  {
    ifstream indata(filename);
    unsigned char color[3];
    string token;
    
    if (!indata.is_open()) {
      cerr << "PPMImage: ERROR: I/O fault: no such file: " 
           << filename << endl;
      valid_ = false;
      return false;
    }
    
    indata >> token; // P6
    if (token != "P6" && token != "P3") {
      cerr << "PPMImage: WARNING: format error: file not a PPM: "
           << filename << endl;
    }

    eat_comments_and_whitespace(indata);
    indata >> u_ >> v_;
    eat_comments_and_whitespace(indata);
    indata >> max_;
    eat_comments_and_whitespace(indata);
    image_.resize(u_*v_);
    if (token == "P6") {
      for(unsigned v=0;v<v_;++v){
	for(unsigned u=0;u<u_;++u){
	  indata.read((char*)color, 3);
	  image_[v*u_+u]=rtrt::Color(color[0]/(double)max_,
                                     color[1]/(double)max_,
                                     color[2]/(double)max_);
	}
      }    
    } else { // P3
      int r, g, b;
      for(unsigned v=0;v<v_;++v){
	for(unsigned u=0;u<u_;++u){
	  indata >> r >> g >> b;
	  image_[v*u_+u]=rtrt::Color(r/(double)max_,
                                     g/(double)max_,
                                     b/(double)max_);
	}
      }    
    }
    valid_ = true;
    return true;
  }
};

} // end namespace

#endif
