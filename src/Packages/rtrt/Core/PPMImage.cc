
#include <Packages/rtrt/Core/PPMImage.h>
#include <Core/Persistent/PersistentSTL.h>

using namespace rtrt;

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

const int PPMIMAGE_VERSION = 1;

namespace SCIRun {
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
