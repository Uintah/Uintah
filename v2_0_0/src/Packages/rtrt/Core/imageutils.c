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

#include <stdlib.h>

float *ConvertUBTYEtoFLOAT( char *s, int len )
{
  float *ptr = (float *)malloc( len * sizeof( float ) );
  int i;

  if (!ptr)
    FatalError("LIBGFX: Unable to allocate memory!");

  for (i=0;i<len;i++)
    ptr[i] = s[i]/255.0;

  return ptr;
}

char *ConvertFLOATtoUBYTE( float *s, int len )
{
  char *ptr = (char *)malloc( len * sizeof( char ) );
  int i;

  if (!ptr)
    FatalError("LIBGFX: Unable to allocate memory!");

  for (i=0;i<len;i++)
    if (s[i] < 0) ptr[i] = 0;
    else if (s[i] > 1) ptr[i] = 1;
    else ptr[i] = 255*s[i];

  return ptr;
}

float *ConvertFloatRGBtoRGBA( float *s, int pix )
{
  int i;
  float *ptr = (float *)malloc( 4 * pix * sizeof ( float ) );
  
  if (!ptr)
    FatalError("LIBGFX: Unable to allocate memory!");

  for (i=0;i<pix;i++)
    {
      ptr[4*i] = s[3*i];
      ptr[4*i+1] = s[3*i+1];
      ptr[4*i+2] = s[3*i+2];
      ptr[4*i+3] = 1;
    }
  
  return ptr;
}

float *ConvertFloatRGBAtoRGB( float *s, int pix )
{
  int i;
  float *ptr = (float *)malloc( 3 * pix * sizeof ( float ) );
  
  if (!ptr)
    FatalError("LIBGFX: Unable to allocate memory!");

  for (i=0;i<pix;i++)
    {
      ptr[3*i] = s[4*i];
      ptr[3*i+1] = s[4*i+1];
      ptr[3*i+2] = s[4*i+2];
    }
  
  return ptr;
}

char *ConvertUByteRGBtoRGBA( char *s, int pix )
{
  char *ptr=(char *)malloc(4 * pix * sizeof( char ) );
  int i;

  if (!ptr)
    FatalError("LIBGFX: Unable to allocate memory!");

  for (i=0;i<pix;i++)
    {
      ptr[4*i] = s[3*i];
      ptr[4*i+1] = s[3*i+1];
      ptr[4*i+2] = s[3*i+2];
      ptr[4*i+3] = 255;
    }
  
  return ptr;
}

char *ConvertUByteRGBAtoRGB( char *s, int pix )
{
  char *ptr=(char *)malloc(3 * pix * sizeof( char ) );
  int i;

  if (!ptr)
    FatalError("LIBGFX: Unable to allocate memory!");

  for (i=0;i<pix;i++)
    {
      ptr[3*i] = s[4*i];
      ptr[3*i+1] = s[4*i+1];
      ptr[3*i+2] = s[4*i+2];
    }
  
  return ptr;
}

texture *ConvertTextureToRGBA( texture *t )
{
  if (t->format == GFXIO_RGBA) return t;
  
  if (t->type == GFXIO_UBYTE)
    {
      char *tmp = ConvertUByteRGBtoRGBA( t->texImage,
					 t->img_width*t->img_height );
      free ( t->texImage );
      t->texImage = tmp;
      t->format = GFXIO_RGBA;
      return t;
    }
  if (t->type == GFXIO_FLOAT)
    {
      float *tmp = ConvertFloatRGBtoRGBA( t->texImagef,
					  t->img_width*t->img_height );
      free ( t->texImagef );
      t->texImagef = tmp;
      t->format = GFXIO_RGBA;
      return t;
    }

  FatalError("LIBGFX: Could not convert texture to RGBA!");
  return NULL;
}

texture *ConvertTextureToRGB( texture *t )
{
  if (t->format == GFXIO_RGB) return t;
  
  if (t->type == GFXIO_UBYTE)
    {
      char *tmp = ConvertUByteRGBAtoRGB( t->texImage,
					 t->img_width*t->img_height );
      free ( t->texImage );
      t->texImage = tmp;
      t->format = GFXIO_RGB;
      return t;
    }
  if (t->type == GFXIO_FLOAT)
    {
      float *tmp = ConvertFloatRGBAtoRGB( t->texImagef,
					  t->img_width*t->img_height );
      free ( t->texImagef );
      t->texImagef = tmp;
      t->format = GFXIO_RGB;
      return t;
    }

  FatalError("LIBGFX: Could not convert texture to RGB!");
  return NULL;
}

texture *ConvertTextureToFloat( texture *t )
{
  if (t->type == GFXIO_FLOAT ) return t;

  if (t->format == GFXIO_RGBA)
    {
      t->texImagef = ConvertUBTYEtoFLOAT( t->texImage, 
					  4*t->img_width*t->img_height );
      free( t->texImage );
      t->type = GFXIO_FLOAT ;
    }
  else if (t->format == GFXIO_RGB)
    {
      t->texImagef = ConvertUBTYEtoFLOAT( t->texImage, 
					  3*t->img_width*t->img_height );
      free( t->texImage );
      t->type = GFXIO_FLOAT ;
    }

  FatalError("LIBGFX: Could not convert texture to float-type!");
  return NULL;
}

texture *ConvertTextureToUByte( texture *t )
{
  if (t->type == GFXIO_UBYTE ) return t;

  if (t->format == GFXIO_RGBA)
    {
      t->texImage = ConvertFLOATtoUBYTE( t->texImagef, 
					  4*t->img_width*t->img_height );
      free( t->texImagef );
      t->type = GFXIO_UBYTE ;
    }
  else if (t->format == GFXIO_RGB)
    {
      t->texImage = ConvertFLOATtoUBYTE( t->texImagef, 
					  3*t->img_width*t->img_height );
      free( t->texImagef );
      t->type = GFXIO_UBYTE ;
    }

  FatalError("LIBGFX: Could not convert texture to ubyte-type!");
  return NULL;
}
