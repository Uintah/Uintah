
#include "Image.h"
#include <GL/gl.h>
#include <fstream>
#include <stdio.h>

using namespace rtrt;

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

void Image::draw()
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
    }
    out.write((char*)&image[0][0], (int)(sizeof(Pixel)*xres*yres));
}

// this code is added for tiled images...

