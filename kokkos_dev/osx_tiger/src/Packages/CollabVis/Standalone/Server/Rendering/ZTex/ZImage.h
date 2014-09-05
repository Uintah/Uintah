#ifndef _Z_image_h
#define _Z_image_h

#include <Malloc/Allocator.h>
#include <Rendering/ZTex/Image.h>
#include <Rendering/ZTex/Utils.h>
#include <Rendering/ZTex/MiscMath.h>

#include <iostream>
#include <fstream>
using namespace std;

namespace SemotusVisum {
namespace Rendering {

class ZImage : public Image
{
public:
  inline unsigned int *IPix() const {
    return (unsigned int *)Pix;
  }
	
  inline unsigned int &operator()(int x, int y) {
    unsigned int *F = IPix();
    return F[y*wid+x];
  }
	
  inline unsigned int &operator[](int x) {
    unsigned int *F = IPix();
    return F[x];
  }

  inline virtual ~ZImage() {
  }

  inline ZImage() : Image() {
  }
  
  inline ZImage(const int w, const int h, const bool fill = false) :
    Image(w, h, 4, fill) {
  }	
		
  // Hooks the given image into this image object.
  virtual inline void SetImage(unsigned int *p,
    const int w, const int h, const int ch = 4) {
    Image::SetImage((unsigned char *)p, w, h, ch);
  }  
};

} // namespace Tools
} // namespace Remote

#endif
