
#include <Packages/rtrt/Core/PPMImage.h>
#include <Core/Persistent/PersistentSTL.h>

#include <map>

using namespace rtrt;
using namespace std;

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

map<string,PPMImage*> readInImages;

bool
PPMImage::read_image(const char* filename)
{
  ifstream indata(filename);
  unsigned char color[3];
  string token;
    
#if 0
  map<string,PPMImage*>::iterator iter = 
                               readInImages.find( filename );
  if( iter != readInImages.end() )
    {
      cout << "Returning already read image!\n";
      return (*iter).second;
    }
#endif

  if (!indata.is_open()) {
    cerr << "PPMImage: ERROR: I/O fault: no such file: " 
	 << filename << "\n";
    valid_ = false;
    return false;
  }
    
  indata >> token; // P6
  if (token != "P6" && token != "P3") {
    cerr << "PPMImage: WARNING: format error: file not a PPM: "
	 << filename << "\n";
  }

  cerr << "PPMImage: reading image: " << filename;
  if (flipped_)
    cerr << " (flipped!)";
  cerr << endl;

#if 0
  readInImages[filename] = this;
#endif

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
  } else { // P3
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
