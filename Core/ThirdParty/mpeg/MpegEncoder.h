/*
 * SCIRunMpeg.h
 *
 * Defines interface for using the Stanford encoder with SCIRun.
 *
 * Cameron Christensen
 *
 * Copyright SCI Group, University of Utah, March 1999
 */

/* Beware all ye who enter here:
 * This code is very nastily hacked.  It uses various parts of the Stanford
 * mpeg encoder and algorithms from the netpbmplus package (to convert RGB
 * to YUV).  
 */

/* This code (and OpenGL.cc) need to be linked against the mpeg encoder. */

/* The roots of this code (the Stanford encoder) are written in nasty C with
 * tons of nasty globals all over the place.  Hopefully this won't conflict
 * with anything in SCIRun.
 * These additional globals are also needed, although with some manipulation
 * they could probably be worked into the program.
 */

#ifndef _MPEGENCODER_H_
#define _MPEGENCODER_H_



/* This is the old Stanford main() function, renamed. */
#ifdef __cplusplus
extern "C" {
#endif
int StartMpegEncoder(int, char**);

/* Call this function after each frame is generated (each time imageY, imageU,
 * and imageV are updated.
 */
extern void SCIRunEncode(int lasttime, unsigned char* imageY,
			 unsigned char* imageU, unsigned char* imageV);
#ifdef __cplusplus
}
#endif

class MpegEncoder {

private:
  unsigned char *imageY;
  unsigned char *imageU;
  unsigned char *imageV;
  int sx, sy;
  
public:
  MpegEncoder() { sx = sy = 0; imageY = imageU = imageV = 0x0; }
  ~MpegEncoder() {}

  /* Call this to begin encoding an mpeg. */
  void BeginMpegEncode(char *name, int sizex, int sizey); 

  /* Call this with each frame you want added to the mpeg. */
  void EncodeFrame(unsigned char* red, unsigned char* green,
		   unsigned char* blue);
  
  /* Call this when you're done dumping frames to the mpeg */
  void DoneEncoding();

};

#endif
