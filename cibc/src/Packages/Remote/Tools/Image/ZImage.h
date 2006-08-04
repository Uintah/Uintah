#ifndef _Z_image_h
#define _Z_image_h

#include <Packages/Remote/Tools/Image/Image.h>
#include <Packages/Remote/Tools/Util/Utils.h>
#include <Packages/Remote/Tools/Math/MiscMath.h>

#include <iostream>
#include <fstream>

namespace Remote {
using namespace std;
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

  inline ~ZImage() {
  }

  inline ZImage() : Image() {
  }
  
  inline ZImage(const int w, const int h, const bool fill = false) :
    Image(w, h, 4, fill) {
  }
	
  // Read a PZM file.
  inline ZImage(const char *FName) {
    LoadPZM(FName);
  }

  void LoadPZM(const char *FName) {
      ifstream InFile(FName, ios::in | ios::binary);
    if(!InFile.is_open())
      {
	cerr << "Failed to open PFM file `" << FName << "'.\n";
      }
		
    char Magic1, Magic2;
    InFile >> Magic1 >> Magic2;
		
    if(Magic1 != 'P' || Magic2 != 'Z') {
      cerr << FName << " is not a known PZM file.\n";
      InFile.close();
    }
		
    InFile.get();
    char c = InFile.peek();
    while(c == '#') {
      char line[1000];
      InFile.getline(line, 999);
      cerr << line << endl;
      c = InFile.peek();
    }
		
    int dyn_Z;
    InFile >> wid >> hgt >> dyn_Z;
    InFile.get();
		
    if(dyn_Z != 255) {
      cerr << "Must be 255. Was " << dyn_Z << endl;
    }
		
    chan = 4;
    size = wid * hgt;
    dsize = size * chan;
    unsigned int *F = new unsigned int[size];
    Pix = (unsigned char *)F;
		
    InFile.read((char *)Pix, dsize);
		
#ifdef SCI_LITTLE_ENDIAN
    // Intel is little-endian.
    // Always assume they are stored as big-endian.
    ConvertLong((unsigned int *)Pix, size);
#endif
		
    InFile.close();
  }
		
  // Hooks the given image into this image object.
  virtual inline void SetImage(unsigned int *p,
    const int w, const int h, const int ch = 4) {
    Image::SetImage((unsigned char *)p, w, h, ch);
  }
  
  // Save the unsigned int image as a PZM file.
  bool SavePZM(const char *FName) const {
    if(Pix == NULL || chan < 1 || wid < 1 || hgt < 1) {
      cerr << "Image is not defined. Not saving.\n";
      return true;
    }
    
    if(chan != 4) {
      cerr << "Can't save a " << chan << " channel image as a PZM.\n";
      return true;
    }
    
#if defined ( SCI_MACHINE_sgi ) || defined ( SCI_MACHINE_hp )
    ofstream OF(FName);
    if(!OF.rdbuf()->is_open())
#else
      ofstream OF(FName, ios::out | ios::binary);
    if(!OF.is_open())
#endif
      {
	cerr << "Failed to open file `" << FName << "'.\n";
	return true;
      }
		
    OF << "PZ\n" << wid << " " << hgt << endl << 255 << endl;
		
#ifdef SCI_LITTLE_ENDIAN
    // Need to convert unsigned ints to big-endian before saving.
    char *Tmp = new char[dsize];
    memcpy(Tmp, Pix, dsize);
    ConvertLong((unsigned int *)Tmp, size);
    OF.write((char *)Tmp, dsize);
    delete [] Tmp;
#else
    OF.write((char *)Pix, dsize);
#endif 
    OF.close();
		
    cerr << "Wrote PZM file " << FName << endl;
		
    return false;
  }
	
  // Map the Z image to colors and save it as a GIF.
#ifdef SCI_USE_TIFF
  inline void SaveConv(char *fname) {
    Image Bob(wid, hgt, 3);
    unsigned char *B = Bob.Pix;
    
    for(int i=0, j=0; i<Bob.dsize; ) {
      B[i++] = Pix[j++];
      B[i++] = Pix[j++];
      B[i++] = Pix[j++];
      
      j++;
    }
    
    Bob.SaveTIFF(fname);
  }
#endif
};

} // End namespace Remote


#endif
