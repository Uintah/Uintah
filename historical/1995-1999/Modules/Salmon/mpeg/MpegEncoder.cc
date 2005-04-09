/*
 * MpegEncoder.cc
 *
 * Defines interface for using the Stanford encoder with SCIRun.
 *
 * Cameron Christensen
 *
 * Copyright SCI Group, University of Utah, March 1999
 */

/* This is pretty ugly code. */

#include "MpegEncoder.h"
#include <stdio.h>
#include <string.h>
#include <fstream.h>

#define CCIR601

void MpegEncoder::BeginMpegEncode(char *name, int sizex, int sizey) {
  
  if(!name || sizex < 1 || sizey < 1)
    printf("Error: invalid arguments to BeginMpegEncode.\n");

  sx = sizex; sy = sizey;
  if(sx % 2 || sy % 2) printf("warning, evil things will happen now...\n");
  
  int sizeY = sizex*sizey;
  int sizeUV = sizeY/4;
  
  imageY = new unsigned char[sizeY];
  imageU = new unsigned char[sizeUV];
  imageV = new unsigned char[sizeUV];

  /* create s_argc, s_argv for start function */
  int s_argc = 0;
  char** s_argv = new char*[20];
  s_argv[s_argc] = new char[3];
  strcpy(s_argv[s_argc++], "-a");
  s_argv[s_argc] = new char[2];
  strcpy(s_argv[s_argc++], "0");
  s_argv[s_argc] = new char[3];
  strcpy(s_argv[s_argc++], "-b");
  s_argv[s_argc] = new char[2];
  strcpy(s_argv[s_argc++], "10000"); /* 10,000 will cause lots/no problems.*/
  s_argv[s_argc] = new char[3];
  strcpy(s_argv[s_argc++], "-h");
  s_argv[s_argc] = new char[5];
  sprintf(s_argv[s_argc++], "%d", sizex);
  s_argv[s_argc] = new char[3];
  strcpy(s_argv[s_argc++], "-v");
  s_argv[s_argc] = new char[5];
  sprintf(s_argv[s_argc++], "%d", sizey);
  //s_argv[s_argc] = new char[4];
  //strcpy(s_argv[s_argc++], "-PF");
  s_argv[s_argc] = new char[3];
  strcpy(s_argv[s_argc++], "-s");
  s_argv[s_argc] = new char[8];
  strcpy(s_argv[s_argc++], name);
    
  /* call StartMpegEncoder to begin */
  //printf("calling startMpegEncoder()...\n");
  StartMpegEncoder(s_argc, s_argv);
}

void MpegEncoder::EncodeFrame(unsigned char* red, unsigned char* green,
			      unsigned char* blue) {
  /* convert r,g,b to y,u,v */

  // temporary pointers to row arrays
  unsigned char *y1ptr, *y2ptr, *uptr, *vptr;
  
  long u,v,y0,y1,y2,y3,u0,u1,u2,u3,v0,v1,v2,v3;
  unsigned char r0,g0,b0,r1,g1,b1,r2,g2,b2,r3,g3,b3;

  // take rows two by two
  int row, col;
  for (row = 0; row < sy; row += 2) {
    //pP1 = pixelrow1; pP2 = pixelrow2;
    //y1ptr = y1buf; y2ptr = y2buf; vptr = vbuf; uptr = ubuf;
    y1ptr = imageY + (sx * row);
    y2ptr = imageY + (sx * (row+1));
    uptr = imageU + (sx/2) * (row/2);
    vptr = imageV + (sx/2) * (row/2);
    
    for (col = 0 ; col < sx; col += 2) {
      
      /* read in a block of four pixels (a square) */
      /* first pixel */
      r0 = red[ sx*row + col ];
      g0 = green[ sx*row + col ];
      b0 = blue[ sx*row + col ];
      /* 2nd pixel */
      r1 = red[ sx*row + col+1 ];
      g1 = green[ sx*row + col+1 ];
      b1 = blue[ sx*row + col+1 ];
      /* 3rd pixel */
      r2 = red[ sx*(row+1) + col ];
      g2 = green[ sx*(row+1) + col ];
      b2 = blue[ sx*(row+1) + col ];
      /* 4th pixel */
      r3 = red[ sx*(row+1) + col+1 ];
      g3 = green[ sx*(row+1) + col+1 ];
      b3 = blue[ sx*(row+1) + col+1 ];


      /* The JFIF RGB to YUV Matrix for $00010000 = 1.0

	 [Y]   [19595   38469    7471][R]
	 [U] = [-11056  -21712  32768][G]
	 [V]   [32768   -27440  -5328][B]

	 */

      y0 =  19595 * r0 + 38469 * g0 +  7471 * b0;
      u0 = -11056 * r0 - 21712 * g0 + 32768 * b0;
      v0 =  32768 * r0 - 27440 * g0 -  5328 * b0;

      y1 =  19595 * r1 + 38469 * g1 +  7471 * b1;
      u1 = -11056 * r1 - 21712 * g1 + 32768 * b1;
      v1 =  32768 * r1 - 27440 * g1 -  5328 * b1;

      y2 =  19595 * r2 + 38469 * g2 +  7471 * b2;
      u2 = -11056 * r2 - 21712 * g2 + 32768 * b2;
      v2 =  32768 * r2 - 27440 * g2 -  5328 * b2;

      y3 =  19595 * r3 + 38469 * g3 +  7471 * b3;
      u3 = -11056 * r3 - 21712 * g3 + 32768 * b3;
      v3 =  32768 * r3 - 27440 * g3 -  5328 * b3;

      /* mean the chroma for subsampling */

      u  = (u0+u1+u2+u3)>>2;
      v  = (v0+v1+v2+v3)>>2;

#ifdef CCIR601  // the standard for MPEG supposedly?

      y0 = (y0 * 219)/255 + 1048576;
      y1 = (y1 * 219)/255 + 1048576;
      y2 = (y2 * 219)/255 + 1048576;
      y3 = (y3 * 219)/255 + 1048576;

      u  = (u * 224)/255 ;
      v  = (v * 224)/255 ;
#endif


      *y1ptr++  = (y0 >> 16) ;
      *y1ptr++  = (y1 >> 16) ;
      *y2ptr++  = (y2 >> 16) ;
      *y2ptr++  = (y3 >> 16) ;


      *uptr++   = (u >> 16)+128 ;
      *vptr++   = (v >> 16)+128 ;

    }
    //fwrite(y1buf, (cols & ~1), 1, yf);
    //fwrite(y2buf, (cols & ~1), 1, yf);
    //fwrite(ubuf, cols/2, 1, uf);
    //fwrite(vbuf, cols/2, 1, vf);
  }
  
  ofstream outfileY("/tmp/enc.Y", ios::out);
  ofstream outfileU("/tmp/enc.U", ios::out);
  ofstream outfileV("/tmp/enc.V", ios::out);

  outfileY.write(imageY, sx*sy);
  outfileU.write(imageU, sx*sy/4);
  outfileV.write(imageV, sx*sy/4);

  outfileY.close();
  outfileU.close();
  outfileV.close();
  
  // call SCIRunEncode...
  SCIRunEncode(0, imageY, imageU, imageV);
}

void MpegEncoder::DoneEncoding() {
  printf("Closing MPEG.\n");
  SCIRunEncode(1, imageY, imageU, imageV);
  
  delete[] imageY;
  delete[] imageU;
  delete[] imageV;
}
