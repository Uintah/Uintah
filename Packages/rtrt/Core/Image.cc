
#include <Packages/rtrt/Core/Image.h>
#include <sci_values.h>

#include <GL/gl.h>

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <sgi_stl_warnings_on.h>

#include <stdio.h>
#include <math.h>

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

ofstream Image::ppm_streamR;
ofstream Image::ppm_streamL;

Image::Image(int xres, int yres, bool stereo)
  : xres(xres), yres(yres), stereo(stereo)
{
    image=0;
    depth=0;
    resize_image();
}

Image::~Image()
{
    if(image){
	delete[] image_buf;
	delete[] image;
    }
}

void Image::resize_image()
{
    if(image){
	delete[] image_buf;
	delete[] image;
    }
    image=new Pixel*[yres*(stereo?2:1)];
    int linesize=128;
    image_buf=new char[xres*yres*sizeof(Pixel)*(stereo?2:1)+linesize];
    unsigned long b=(unsigned long)image_buf;
    int off=(int)(b%linesize);
    Pixel* p;
    if(off){
	p=(Pixel*)(image_buf+linesize-off);
    } else {
	p=(Pixel*)image_buf;
    }

    int yr=stereo?2*yres:yres;
    for(int y=0;y<yr;y++){
	image[y]=p;
	p+=xres;
    }
  if(depth){
    delete[] depth_buf;
    delete[] depth;
  }
  depth=new float*[yres*(stereo?2:1)];
  depth_buf=new float[xres*yres*(stereo?2:1)+linesize];
  b=(unsigned long)depth_buf;
  off=(int)(b%linesize);
  float* f;
  if(off){
    f=(depth_buf+linesize-off);
  } else {
    f=depth_buf;
  }
  
  for(int y=0;y<yr;y++){
    depth[y]=f;
    f+=xres;
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
    if(!stereo){
      glDrawBuffer(GL_BACK);
      glRasterPos2i(0,0);
      glDrawPixels(xres, yres, GL_RGBA, GL_UNSIGNED_BYTE, &image[0][0]);
    } else {
      glDrawBuffer(GL_BACK_LEFT);
      glRasterPos2i(0,0);
      glDrawPixels(xres, yres, GL_RGBA, GL_UNSIGNED_BYTE, &image[0][0]);
      glDrawBuffer(GL_BACK_RIGHT);
      glRasterPos2i(0,0);
      glDrawPixels(xres, yres, GL_RGBA, GL_UNSIGNED_BYTE, &image[yres][0]);
    }
  }
}

void Image::draw_depth( float max_depth ) {
  int num_pixels = xres*yres;
  float *pixel = &depth[0][0];
  Pixel *pic = image[0];
  
#if 0
  float max = *pixel;
  for(int i = 1; i < num_pixels; i++) {
    float val = *pixel;
    if (val > max)
      max = val;
    pixel++;
  }
  cerr << "max = "<<max<<"\n";
  float inv_md = 1.0f/max;
#else
  float inv_md = 1.0f/max_depth;
  if (max_depth == 0) inv_md = 0;
#endif
  pixel = depth[0];
  for(int i = 0; i < num_pixels; i++, pixel++, pic++) {
    float val;
    val = *pixel;
    if (val > 0) {
      if (val < max_depth)
        val = sqrtf(val*inv_md) * 255;
      else
        val = 255;
    } else {
      val = 0;
    }
    unsigned char val_uc = (unsigned char)val;
    pic[0] = Pixel(val_uc, val_uc, val_uc, 255);
  }
#if 0
  glDrawBuffer(GL_BACK);
  glRasterPos2i(0,0);
  glDrawPixels(xres, yres, GL_LUMINANCE, GL_FLOAT, &depth[0][0]);
#else
  draw(0,0);
#endif
}

void Image::draw_sils_on_image( float max_depth ) {
  float *pixel = &depth[0][0];
  float inv_md;
  if (max_depth > 0)
    inv_md = 1.0f/max_depth;
  else
    inv_md = 0;
  pixel = &depth[0][0];

#if 0
  float max_val = -MAXFLOAT;
  float min_val = MAXFLOAT;
#endif
  for(int j = 0; j < yres; j++) {
    int ylow = j>0?j-1:0;
    int yhigh = j<yres-1?j+1:yres-1;
    for(int i = 0; i < xres; i++) {
#if 0
      float val;
      val = depth[j][i];
      if (val > 0) {
        if (val < max_depth)
          val = sqrtf(val*inv_md) * 255;
        else
          val = 255;
      } else {
        val = 0;
      }
      image[j][i] = Pixel(val, val, val, 255);
#else
      int xlow = i>0?i-1:0;
      int xhigh = i<xres-1?i+1:xres-1;

#if 0
      // Compute the first derivative
      float dx = 0
        - depth[j][xlow] - depth[j][xlow]
        + depth[j][xhigh] + depth[j][xhigh]
        - depth[ylow][xlow]
        + depth[ylow][xhigh]
        - depth[yhigh][xlow]
        + depth[yhigh][xhigh];
      float dy = 0
        - depth[yhigh][xlow]
        - depth[yhigh][i] - depth[yhigh][i]
        - depth[yhigh][xhigh]
        + depth[ylow][xlow]
        + depth[ylow][i] + depth[ylow][i]
        + depth[ylow][xhigh];

      // Threshold the second derivative
      float first_der = dx*dx + dy*dy;

      if (first_der > max_val) max_val = first_der;
      if (first_der < min_val) min_val = first_der;

      first_der = first_der/500*255;
      if (first_der > 255)
        val = 255;
      else
        val = 0;
#endif
      //      if (val > 0) {
      if (true) {
        // Compute the laplacian
        float inside_val = depth[j][i];
        inside_val *= 8;

        float outside_val =
          depth[j][xlow] +
          depth[j][xhigh] +
          depth[ylow][xlow] +
          depth[ylow][i] +
          depth[ylow][xhigh] +
          depth[yhigh][xlow] +
          depth[yhigh][i] +
          depth[yhigh][xhigh];

#if 0
        if (inside_val < 0 && inside_val > outside_val)
          image[j][i] = Pixel(val, 0, 0, 255);
        else if (inside_val < 0 && inside_val < outside_val)
          image[j][i] = Pixel(val, val, 0, 255);
        else if (inside_val > 0 && inside_val > outside_val)
          image[j][i] = Pixel(0, val, val, 255);
        else if (inside_val > 0 && inside_val < outside_val)
          image[j][i] = Pixel(0, 50, 0, 255);
        else
          image[j][i] = Pixel(0, 0, val, 255);
#elif 0
        if (inside_val > 0 && inside_val > outside_val)
          image[j][i] = Pixel(255, 255, 255, 255);
        else
          image[j][i] = Pixel(0, 0, 0, 255);
#else
        if (inside_val - outside_val > inv_md)
          image[j][i] = Pixel(0, 0, 0, 255);
#endif
        //        if (val > max_val) max_val = val;
        //        if (val < min_val) min_val = val;
        //      val = (val*.5)+128;
        //      val = fabsf(val)*10;
      } else {
        //        image[j][i] = Pixel(0, 0, 0, 255);
      }      
#endif
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

void Image::set_depth(int x, int y, double d) {
  if (d <= MAXFLOAT/10)
    depth[y][x] = d;
  else
    depth[y][x] = MAXFLOAT/10;
  //depth[y][x] = 0;
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

void Image::save_ppm(char *filename, bool draw_sils, float max_depth)
{
  // We need to find a filename that isn't already taken
  FILE *input_test;
  char new_filename[1000];
  if (!stereo)
    sprintf(new_filename, "%s00.ppm", filename);
  else
    sprintf(new_filename, "%s00-L.ppm", filename);
  int count = 0;
  // I'm placing a max on how high count can get to prevent an
  // infinate loop when the path is just bad and no amount of testing
  // will create a viable filename.
  while ((input_test = fopen(new_filename, "r")) != 0 && count < 500) {
    fclose(input_test);
    input_test = 0;
    if (!stereo)
      sprintf(new_filename, "%s%02d.ppm", filename, ++count);
    else
      sprintf(new_filename, "%s%02d-L.ppm", filename, ++count);
  }

  if (draw_sils) draw_sils_on_image(max_depth);
  
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
  printf("Finished writing image of size (%d, %d) to %s\n", xres, yres,
	 new_filename);
  if (stereo) {
    sprintf(new_filename, "%s%02d-R.ppm", filename, count);
    outdata.open(new_filename);
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
	c[0]=image[v+yres][u].r;
	c[1]=image[v+yres][u].g;
	c[2]=image[v+yres][u].b;
	outdata.write((char *)c, 3);
      }
    }
    outdata.close();
    printf("Finished writing Right image of size (%d, %d) to %s\n", xres, yres,
	   new_filename);
  }
}

void Image::save_depth(char *filename)
{
  // We need to find a filename that isn't already taken
  FILE *input_test;
  char new_filename[1000];
  if (!stereo)
    sprintf(new_filename, "%s00.nrrd", filename);
  else
    sprintf(new_filename, "%s00-L.nrrd", filename);
  int count = 0;
  // I'm placing a max on how high count can get to prevent an
  // infinate loop when the path is just bad and no amount of testing
  // will create a viable filename.
  while ((input_test = fopen(new_filename, "r")) != 0 && count < 500) {
    fclose(input_test);
    input_test = 0;
    if (!stereo)
      sprintf(new_filename, "%s%02d.nrrd", filename, ++count);
    else
      sprintf(new_filename, "%s%02d-L.nrrd", filename, ++count);
  }
  ofstream outdata(new_filename);
  if (!outdata.is_open()) {
    cerr << "Image::save_depth: ERROR: I/O fault: couldn't open image file to write: " << new_filename << "\n";
    return;
  }
  outdata << "NRRD0001\n";

  outdata << "type: float\n";
  outdata << "dimension: 2\n";
  outdata << "sizes: "<< xres << " " << yres << "\n";
#ifdef __sgi
  outdata << "endian: big\n";
#else
  outdata << "endian: little\n";
#endif
  outdata << "encoding: raw\n\n";
  
  for(int v=yres-1;v>=0;--v)
    outdata.write((char*)depth[v], xres*sizeof(float));

  outdata.close();
  printf("Finished writing image of size (%d, %d) to %s\n", xres, yres,
	 new_filename);
  if (stereo) {
    sprintf(new_filename, "%s%02d-R.nrrd", filename, count);
    outdata.open(new_filename);
    if (!outdata.is_open()) {
      cerr << "Image::save_ppm: ERROR: I/O fault: couldn't image file to write: "
	   << new_filename << "\n";
      return;
    }

    outdata << "NRRD0001\n";

    outdata << "type: float\n";
    outdata << "dimension: 2\n";
    outdata << "sizes: "<< xres << " " << yres << "\n";
#ifdef __sgi
    outdata << "endian: big\n";
#else
    outdata << "endian: little\n";
#endif
    outdata << "encoding: raw\n\n";
  
    for(int v=yres-1;v>=0;--v)
      outdata.write((char*)depth[v+yres], xres*sizeof(float));

    outdata.close();

    printf("Finished writing Right image of size (%d, %d) to %s\n", xres, yres,
	   new_filename);
  }
}

bool Image::start_ppm_stream(char *filename) {
  // Make sure we close up any previous streams
  end_ppm_stream();
  
  // We need to find a filename that isn't already taken
  FILE *input_test;
  char new_filename[1000];
  if (!stereo)
    sprintf(new_filename, "%s00.ppm_stream", filename);
  else
    sprintf(new_filename, "%s00-L.ppm_stream", filename);
  int count = 0;
  // I'm placing a max on how high count can get to prevent an
  // infinate loop when the path is just bad and no amount of testing
  // will create a viable filename.
  while ((input_test = fopen(new_filename, "r")) != 0 && count < 500) {
    fclose(input_test);
    input_test = 0;
    if (!stereo)
      sprintf(new_filename, "%s%02d.ppm_stream", filename, ++count);
    else
      sprintf(new_filename, "%s%02d-L.ppm_stream", filename, ++count);
  }

  // Open the files
  ppm_streamL.open(new_filename);
  if (!ppm_streamL.is_open()) {
    cerr << "Image::save_ppm: ERROR: I/O fault: couldn't write image file: "
	 << new_filename << "\n";
    return false;
  }

  if (stereo) {
    sprintf(new_filename, "%s%02d-R.ppm_stream", filename, count);
    ppm_streamR.open(new_filename);
    if (!ppm_streamR.is_open()) {
      cerr << "Image::save_ppm: ERROR: I/O fault: couldn't write image file: "
	   << new_filename << "\n";
      return false;
    }
  }
  return true;
}

void Image::end_ppm_stream() {
  if (ppm_streamL.is_open())
    ppm_streamL.close();
  if (ppm_streamR.is_open())
    ppm_streamR.close();
}

void Image::save_to_ppm_stream() {
  if (!ppm_streamL.is_open()) {
    cerr << "Image::save_to_ppm_stream: Can't save to a closed stream\n";
    return;
  }
  ppm_streamL << "P6\n# PPM binary image created with rtrt\n";
  
  ppm_streamL << xres << " " << yres << "\n";
  ppm_streamL << "255\n";
  
  unsigned char c[3];
  for(int v=yres-1;v>=0;--v){
    for(int u=0;u<xres;++u){
      c[0]=image[v][u].r;
      c[1]=image[v][u].g;
      c[2]=image[v][u].b;
      ppm_streamL.write((char *)c, 3);
    }
  }
  
  printf("Finished writing image of size (%d, %d) to ppm_stream\n",xres, yres);
  
  if (stereo) {
    if (!ppm_streamR.is_open()) {
      cerr << "Image::save_to_ppm_stream: Can't save right image to a closed stream\n";
      return;
    }
    ppm_streamR << "P6\n# PPM binary image created with rtrt\n";
  
    ppm_streamR << xres << " " << yres << "\n";
    ppm_streamR << "255\n";
  
    unsigned char c[3];
    for(int v=yres-1;v>=0;--v){
      for(int u=0;u<xres;++u){
	c[0]=image[v+yres][u].r;
	c[1]=image[v+yres][u].g;
	c[2]=image[v+yres][u].b;
	ppm_streamR.write((char *)c, 3);
      }
    }
    printf("Finished writing Right image of size (%d, %d) to ppm_stream\n",
           xres, yres);
  }
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
