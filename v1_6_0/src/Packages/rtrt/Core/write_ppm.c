/*=========================================================================
//
//
//
//
//
//
//
//
//
//
//=========================================================================*/

#ifndef __PPM_H__
#include <Packages/rtrt/Core/ppm.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <math.h>

/* 
** write the image with a given width & height to a file called filename,
** the data is given as a stream of chars (unsigned bytes) in the pointer 
** ptr.
*/
int __WritePPM( char *f, char *ptr, int width, int height, int mode )
{
  time_t thetime;
  int x, y, i;
  unsigned char r, g, b, bw=0;
  FILE *out;

  if (mode<0) 
    {
      Error("LIBGFX: Bad image format type!");
      return GFXIO_UNSUPPORTED;
    }
  if (mode==PBM_RAW || mode==PBM_ASCII)
    Warning("LIBGFX: Distortions occur converting to PBM format!");

  out = fopen(f, "wb");
  if (!out) {
    char buf[256];
    sprintf( buf, "LIBGFX: Unable to open file '%s', output lost!", f );
    Error( buf );
    return GFXIO_OPENERROR;
  }
  
  fprintf(out, "P%d\n", mode);  
  thetime = time(0);
  fprintf(out, "# File created by Chris Wyman's PPM Library on %s",
	  ctime(&thetime));
  fprintf(out, "%d %d\n", width, height);
  /* PBM's are just 1's and 0's, so there's no max component entry */
  if (mode!=PBM_RAW && mode!=PBM_ASCII)
    fprintf(out, "%d\n", 255); 
  
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      r = *(ptr+4*(y*width+x));  
      g = *(ptr+4*(y*width+x)+1);
      b = *(ptr+4*(y*width+x)+2);

      if (mode == PPM_RAW)
	fprintf(out, "%c%c%c", r, g, b);
      else if (mode == PPM_ASCII)
	{
	  fprintf(out, "%d %d %d ", r, g, b);
	  if (((++i) % 5) == 0) fprintf( out, "\n" ); 
	}
      else if (mode == PGM_RAW)
	fprintf(out, "%c", ((r+g+b)/3));
      else if (mode == PGM_ASCII)
	{
	  fprintf(out, "%d ", (r+g+b)/3);
	  if (((++i) % 15) == 0) fprintf( out, "\n" ); 
	}
      else if (mode == PBM_ASCII)
	{
	  fprintf(out, "%d ", ( (((r+g+b)/3)<128) ? 1 : 0));
	  if (((++i) % 15) == 0) fprintf( out, "\n" ); 
	}
      else if (mode == PBM_RAW)
	{
	  char add=0;
	  if (((r+g+b)/3)<128) add=1;
	  bw = (bw << 1)+add;
	  if (((++i) % 8) == 0) 
	    {
	      fprintf( out, "%c", bw );
	      bw = 0;
	    }
	  else if ((x+1)==width) 
	    /* there's padding at the end of rows, evidently... grr! */
	    {
	      int bits=8-(i%8);
	      bw = bw<<bits;
	      fprintf( out, "%c", bw );
	      bw=i=0;
	    }
	}
    }
  }
  fprintf(out, "\n");
  fclose(out);

  return GFXIO_OK;
}


/* 
** write the image with a given width & height to a file called filename,
** the data is given as a stream of floats (in range 0-1) in the pointer 
** ptr.
*/
int WriteImageFloat( int mode, char *f, float *ptr, int width, int height )
{
  int i;
  char *im;

  if (mode < PPM_MAX)   /* writing a PPM type image */
    {
      im = (char *)malloc( width * height * 4 * sizeof( char ) );
      
      for (i = 0; i < width * height; i++)
	{
	  *(im+(i*4)) = (char)(((float)*(ptr+(i*3)))*255.0);
	  *(im+(i*4)+1) = (char)(((float)*(ptr+(i*3)+1))*255.0);
	  *(im+(i*4)+2) = (char)(((float)*(ptr+(i*3)+2))*255.0);
	  *(im+(i*4)+3) = (char)255;
	}
      
      i = __WritePPM( f, im, width, height, mode );
      free( im );

      return i;
    }
  else if (mode == RGBE_IMAGE)
    {
      return WriteRGBEImage( f, ptr, width, height );
    }

  Error("LIBGFX: Unknown file type in WriteImageFloat()!");

  return GFXIO_UNSUPPORTED;
}

/* 
** write the image with a given width & height to a file called filename,
** the data is given as a stream of floats (in range 0-1) in the pointer 
** ptr.
*/
int WriteGammaImageFloat( int mode, char *f, float *ptr, int width, int height, double gamma )
{
  int i;
  char *im;
  double r, g, b;

  if (gamma<=0) gamma=2;

  if (mode < PPM_MAX)
    {
      im = (char *)malloc( width * height * 4 * sizeof( char ) );
      if (!im)
	FatalError("LIBGFX: Memory allocation error!");
      
      for (i = 0; i < width * height; i++)
	{
	  r = pow( *(ptr+(i*3)), 1.0/gamma );
	  g = pow( *(ptr+(i*3)+1), 1.0/gamma );
	  b = pow( *(ptr+(i*3)+2), 1.0/gamma );
	  
	  *(im+(i*4)) = (char)(r*255.0);
	  *(im+(i*4)+1) = (char)(g*255.0);
	  *(im+(i*4)+2) = (char)(b*255.0);
	  *(im+(i*4)+3) = (char)255;
	}
      
      i = __WritePPM( f, im, width, height, mode );
      
      free( im );
      return i;
    }

  Error("LIBGFX: WriteGammaImageFloat() currently only handles PPM-type files!");

  return GFXIO_UNSUPPORTED;
}

/*
** given a texture structure, write out the data in it to the PPM
** file 'file'
*/
int WriteTextureToImage( int mode, char *f, texture *T )
{
  if (mode < PPM_MAX)
    {
      ConvertTextureToUByte( T );
      ConvertTextureToRGBA( T );
      return __WritePPM( f, T->texImage, T->img_width, T->img_height, mode );
    }

  Error("LIBGFX: Unsupported image format in WriteTextureToImage()!");
  
  return GFXIO_UNSUPPORTED;
}

/* 
** given a stream of characters, write to a PPM/PGM/PBM file
**
*/
int WriteImage( int mode, char *f, char *ptr, int width, int height )
{
  if (mode >= PPM_MAX)
    {
      Error("LIBGFX: WriteImage() only writes PPM-type files!");
      return GFXIO_UNSUPPORTED;
    }
  return __WritePPM( f, ptr, width, height, mode );
}





/* 
** write the image with a given width & height to a file called filename,
** the data is given as a stream of chars (unsigned bytes) in the pointer 
** ptr.
*/
int WriteGammaPGMFloat( char *f, float *ptr, int width, int height, double gamma )
{
  time_t thetime;
  int x, y;
  float c=0;
  FILE *out;

  if (gamma<=0) gamma=2;

  out = fopen(f, "wb");
  if (!out) {
    char buf[256];
    sprintf( buf, "LIBGFX: Unable to open file '%s', output lost!", f );
    Error( buf );
    return GFXIO_OPENERROR;
  }
  
  fprintf(out, "P%d\n", PGM_RAW);  
  thetime = time(0);
  fprintf(out, "# File created by Chris Wyman's PPM Library on %s",
	  ctime(&thetime));
  fprintf(out, "%d %d\n", width, height);
  fprintf(out, "%d\n", 255); 
  
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      c = *(ptr+(y*width+x));  
      c = pow( c, 1.0/gamma );
      fprintf(out, "%c", (char)(c*255));
    }
  }
  fprintf(out, "\n");
  fclose(out);

  return GFXIO_OK;
}

