/*
 * MpegEncoder.h
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
 * Note: Steve says it *will* affect portability, but that alone probably
 * isn't a strong enough reason not to include it. Just don't compile it
 * on non-unix platforms.
 *
 * Note also: The -PF option has been replaced by a user constraint to
 * only provide image sizes in multiples of 16. The code in OpenGL.cc has
 * been modified to pad images out, but if this code is ever used anywhere \
 * else the user will need to be aware of it.
 */

#ifndef _MPEGENCODER_H_
#define _MPEGENCODER_H_

namespace PSECommon {
namespace Modules {

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
  /*
   * NOTE: sx and sy must be evenly divisible by 16.  Ensure this in calling
   * program.
   */
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

} // End namespace Modules
} // End namespace PSECommon


#endif
