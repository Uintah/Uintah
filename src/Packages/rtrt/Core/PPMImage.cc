
#include <Packages/rtrt/Core/PPMImage.h>

#include <Core/Persistent/PersistentSTL.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace std;

PPMImage::PPMImage(const string& s, bool flip) 
  : flipped_(flip) 
{ 
  valid_ = read_image(s.c_str());
}

PPMImage::PPMImage(int nu, int nv, bool flip) 
  : u_(nu), v_(nv), valid_(false), flipped_(flip) 
{
  image_.resize(u_*v_);
}

void
PPMImage::eat_comments_and_whitespace(ifstream &str)
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

void PPMImage::get_dimensions_and_data(Array2<rtrt::Color> &c,
                                       int &nu, int &nv) {
  if (valid_) {
    c.resize(u_+2,v_+2);  // image size + slop for interpolation
    nu=u_;
    nv=v_;
    for (unsigned v=0; v<v_; ++v)
      for (unsigned u=0; u<u_; ++u)
        c(u,v)=image_[v*u_+u];
  } else {
    c.resize(0,0);
    nu=0;
    nv=0;
  }
}

bool PPMImage::write_image(const char* filename, int bin)
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

bool
PPMImage::read_image(const char* filename)
{
  ifstream indata(filename);
  unsigned char color[3];
  string token;
    
  if (!indata.is_open()) {
    cerr << "PPMImage: ERROR: I/O fault: no such file: " 
	 << filename << "\n";
    valid_ = false;
    return false;
  }
    
  indata >> token; // P6
  if (token != "P6" && token != "P3" && token != "P5" && token != "P2") {
    cerr << "PPMImage: WARNING: format error: file not a PPM: "
	 << filename << "\n";
    valid_ = false;
    return false;
  }

  cerr << "PPMImage: reading image: " << filename;
  if (flipped_)
    cerr << " (flipped!)";
  cerr << endl;

  eat_comments_and_whitespace(indata);
  indata >> u_ >> v_;
  eat_comments_and_whitespace(indata);
  indata >> max_;
  // After we get this we only want to eat a new line rather than
  // trying to get all the whitespace, as this has the potential to
  // eat into our data if the data just happens to match '\t', ' ', or
  // '\n'.
  {
    char c;
    for(;;) {
      indata.get(c);
      if (c=='\n') {
	break;
      }
    }
  }
  image_.resize(u_*v_);
  if (token == "P6") {
    for(unsigned v=0;v<v_;++v){
      for(unsigned u=0;u<u_;++u){
	indata.read((char*)color, 3);
	if (flipped_) {
	  image_[(v_-v-1)*u_+u]=rtrt::Color(color[0]/(double)max_,
					    color[1]/(double)max_,
					    color[2]/(double)max_);
	} else {
	  image_[v*u_+u]=rtrt::Color(color[0]/(double)max_,
				     color[1]/(double)max_,
				     color[2]/(double)max_);
	}
      }
    }    
  } else if (token == "P3") { // P3
    int r, g, b;
    for(unsigned v=0;v<v_;++v){
      for(unsigned u=0;u<u_;++u){
	indata >> r >> g >> b;
	if (flipped_) {
	  image_[(v_-v-1)*u_+u]=rtrt::Color(r/(double)max_,
					    g/(double)max_,
					    b/(double)max_);
	} else {
	  image_[v*u_+u]=rtrt::Color(r/(double)max_,
				     g/(double)max_,
				     b/(double)max_);
	}
      }
    }    
  } else if (token == "P5") { // P3
    for(unsigned v=0;v<v_;++v){
      for(unsigned u=0;u<u_;++u){
	indata.read((char*)color, 1);
	if (flipped_) {
	  image_[(v_-v-1)*u_+u]=rtrt::Color(color[0]/(double)max_,
					    color[0]/(double)max_,
					    color[0]/(double)max_);
	} else {
	  image_[v*u_+u]=rtrt::Color(color[0]/(double)max_,
				     color[0]/(double)max_,
				     color[0]/(double)max_);
	}
      }
    }    
  } else if (token == "P2") { // P3
    int val;
    for(unsigned v=0;v<v_;++v){
      for(unsigned u=0;u<u_;++u){
	indata >> val;
	if (flipped_) {
	  image_[(v_-v-1)*u_+u]=rtrt::Color(val/(double)max_,
					    val/(double)max_,
					    val/(double)max_);
	} else {
	  image_[v*u_+u]=rtrt::Color(val/(double)max_,
				     val/(double)max_,
				     val/(double)max_);
	}
      }
    }    
  } else {
    cerr << "Don't know how to read image with magic number "<<token<<"\n";
    valid_ = false;
    return false;
  }
  valid_ = true;
  return true;
}


const int PPMIMAGE_VERSION = 1;


namespace SCIRun {

template <> 
void Pio(Piostream& stream, std::vector<rtrt::Color>& data)
{ 
  stream.begin_class("STLVector", STLVECTOR_VERSION);
  
  int size=(int)data.size();
  stream.io(size);
  
  if(stream.reading()){
    data.resize(size);
  }

  if (stream.supports_block_io()) {
    rtrt::Color &c = *data.begin();
    stream.block_io(&c, sizeof(rtrt::Color), size);
  } else {
    for (int i = 0; i < size; i++)
      {
        Pio(stream, data[i]);
      }
  }
  stream.end_class();  
}

void Pio(SCIRun::Piostream &str, rtrt::PPMImage& obj)
{
  str.begin_class("PPMImage", PPMIMAGE_VERSION);
  SCIRun::Pio(str, obj.u_);
  SCIRun::Pio(str, obj.v_);
  SCIRun::Pio(str, obj.valid_);
  SCIRun::Pio(str, obj.image_);
  SCIRun::Pio(str, obj.flipped_);
  str.end_class();
}
} // end namespace SCIRun
