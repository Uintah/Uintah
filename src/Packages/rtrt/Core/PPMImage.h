
#ifndef PPMIMAGE_H
#define PPMIMAGE_H 1

#include <Packages/rtrt/Core/Color.h>
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

  unsigned      u_,v_;
  unsigned      size_;
  bool          valid_;
  vector<Color> image_;

 public:

  PPMImage(const string& s) : valid_(false) { read_image(s.c_str()); }
  virtual ~PPMImage() {}

  unsigned get_width() { return u_; }
  unsigned get_height() { return v_; }
  unsigned get_size() { return size_; }

  Color &operator()(unsigned u, unsigned v)
  {
    return image_[v*u_+u];
  }

  const Color &operator()(unsigned u, unsigned v) const
  {
    return image_[v*u_+u];
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
    if (token != "P6") {
      cerr << "PPMImage: WARNING: format error: file not a binary PPM: "
           << filename << endl;
    }

    eat_comments_and_whitespace(indata);
    indata >> u_ >> v_;
    eat_comments_and_whitespace(indata);
    indata >> size_;
    eat_comments_and_whitespace(indata);
    image_.resize(u_*v_);
    for(unsigned v=0;v<v_;++v){
      for(unsigned u=0;u<u_;++u){
        indata.read((char*)color, 3);
        image_[v*u_+u]=Color(color[0]/(double)size_,
                             color[1]/(double)size_,
                             color[2]/(double)size_);
      }
    }    
    valid_ = true;
    return true;
  }
  
};

} // end namespace

#endif





