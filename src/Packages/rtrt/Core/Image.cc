
#include <Packages/rtrt/Core/Image.h>

#include <GL/gl.h>

#include <fstream>
#include <stdio.h>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

////////////////////////////////////////////
// OOGL stuff
BasicTexture * rtrtTopTex;
ShadedPrim   * rtrtTopTexQuad;
BasicTexture * rtrtBotTex;
ShadedPrim   * rtrtBotTexQuad;
BasicTexture * rtrtMidBotTex;
ShadedPrim   * rtrtMidTopTexQuad;
BasicTexture * rtrtMidTopTex;
ShadedPrim   * rtrtMidBotTexQuad;
////////////////////////////////////////////

Persistent* image_maker() {
  return new Image();
}

// initialize the static member type_id
PersistentTypeID Image::type_id("Image", "Persistent", image_maker);

Image::Image(int xres, int yres, bool stereo)
    : xres(xres), yres(yres), stereo(stereo)
{
    image=0;
    resize_image();
}

Image::~Image()
{
    if(image){
	delete[] buf;
	delete[] image;
    }
}

void Image::resize_image()
{
    if(image){
	delete[] buf;
	delete[] image;
    }
    image=new Pixel*[yres*(stereo?2:1)];
    int linesize=128;
    buf=new char[xres*yres*sizeof(Pixel)*(stereo?2:1)+linesize];
    unsigned long b=(unsigned long)buf;
    int off=(int)(b%linesize);
    Pixel* p;
    if(off){
	p=(Pixel*)(buf+linesize-off);
    } else {
	p=(Pixel*)buf;
    }

    int yr=stereo?2*yres:yres;
    for(int y=0;y<yr;y++){
	image[y]=p;
	p+=xres;
    }
}

void Image::resize_image(const int new_xres, const int new_yres) {
  xres = new_xres;
  yres = new_yres;
  resize_image();
}

void Image::draw( int window_size, bool fullscreen )
{
#if defined(HAVE_OOGL)
  if( fullscreen ) {
    if( window_size == 0 ) {
      // Because textures must be powers of 2 in size, we have broken
      // the rtrt render window into two textures, one of 512 pixels
      // ans one of 128 pixels.
      rtrtBotTex->reset( GL_UNSIGNED_BYTE, &image[0][0] );
      rtrtBotTexQuad->draw();
      rtrtTopTex->reset( GL_UNSIGNED_BYTE, &image[512][0] );
      rtrtTopTexQuad->draw();
    }
  else
    {
      rtrtMidBotTex->reset( GL_UNSIGNED_BYTE, &image[0][0] );
      rtrtMidBotTexQuad->draw();
      rtrtMidTopTex->reset( GL_UNSIGNED_BYTE, &image[256][0] );
      rtrtMidTopTexQuad->draw();
    }
  } else 
#endif
    {
    if(stereo){
      glDrawBuffer(GL_BACK_LEFT);
      glRasterPos2i(0,0);
      glDrawPixels(xres, yres, GL_RGBA, GL_UNSIGNED_BYTE, &image[0][0]);
      glDrawBuffer(GL_BACK_RIGHT);
      glRasterPos2i(0,0);
      glDrawPixels(xres, yres, GL_RGBA, GL_UNSIGNED_BYTE, &image[yres][0]);
    } else {
      glDrawBuffer(GL_BACK);
      glRasterPos2i(0,0);
      glDrawPixels(xres, yres, GL_RGBA, GL_UNSIGNED_BYTE, &image[0][0]);
    }
  }
}

void Image::set(const Pixel& value)
{
    for(int y=0;y<yres;y++){
	for(int x=0;x<xres;x++){
	    image[y][x]=value;
	}
    }
}

void Image::save(char* filename)
{
  ofstream out(filename);
  if(!out){
    cerr << "error opening file: " << filename << '\n';
    return;
  }
  printf("Image res(%d,%d)\n",xres, yres);
  out.write((char*)&image[0][0], (int)(sizeof(Pixel)*xres*yres));
  out.close();
  printf("Finished writing image\n");
}

void Image::save_ppm(char *filename)
{
  // We need to find a filename that isn't already taken
  FILE *input_test;
  char new_filename[1000];
  sprintf(new_filename, "%s.ppm", filename);
  int count = 0;
  // I'm placing a max on how high count can get to prevent an
  // infinate loop when the path is just bad and no amount of testing
  // will create a viable filename.
  while ((input_test = fopen(new_filename, "r")) != 0 && count < 500) {
    fclose(input_test);
    input_test = 0;
    sprintf(new_filename, "%s%02d.ppm", filename, count++);
  }
  ofstream outdata(new_filename);
  if (!outdata.is_open()) {
    cerr << "Image::save_ppm: ERROR: I/O fault: couldn't write image file: "
	 << new_filename << "\n";
    return;
  }
  outdata << "P6\n# PPM binary image created with rtrt\n";
  
  outdata << xres << " " << yres << "\n";
  outdata << "255\n";
  
  unsigned char c[3];
  for(int v=yres-1;v>=0;--v){
    for(int u=0;u<xres;++u){
      c[0]=image[v][u].r;
      c[1]=image[v][u].g;
      c[2]=image[v][u].b;
      outdata.write((char *)c, 3);
    }
  }
  outdata.close();
  printf("Finished writing image to %s\n", new_filename);
}

// this code is added for tiled images...

const int IMAGE_VERSION = 1;

void 
Image::io(SCIRun::Piostream &str)
{
  str.begin_class("Image", IMAGE_VERSION);
  SCIRun::Pio(str, xres);
  SCIRun::Pio(str, yres);
  SCIRun::Pio(str, stereo);

  if (str.reading()) {
    image = 0;
    resize_image();
  }
  
  for(int y=0;y<yres;y++){
    for(int x=0;x<xres;x++){
      Pio(str, image[y][x].r);
      Pio(str, image[y][x].g);
      Pio(str, image[y][x].b);
      Pio(str, image[y][x].a);
    }
  }
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Image*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Image::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Image*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
