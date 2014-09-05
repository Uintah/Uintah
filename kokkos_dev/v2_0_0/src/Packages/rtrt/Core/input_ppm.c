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

#include <stdio.h>
#include <stdlib.h>

#ifndef __PPM_H__
#include <Packages/rtrt/Core/ppm.h>
#endif

/* 
** check if integer specified is a valid 
** image mode 
*/
int IsValidMode( int mode )
{
  if (mode==PPM_ASCII) return 1;
  if (mode==PGM_ASCII) return 1;
  if (mode==PBM_ASCII) return 1;
  if (mode==PPM_RAW) return 1;
  if (mode==PGM_RAW) return 1;
  if (mode==PBM_RAW) return 1;
  return 0;
}

/* is this mode a raw mode? */
int RawMode( int mode )
{
  if (mode==PPM_RAW) return 1;
  if (mode==PGM_RAW) return 1;
  if (mode==PBM_RAW) return 1;
  return 0;
}

/* is this mode a raw mode? */
int ASCIIMode( int mode )
{
  if (mode==PPM_ASCII) return 1;
  if (mode==PGM_ASCII) return 1;
  if (mode==PBM_ASCII) return 1;
  return 0;
}


/* a fatal, terminating error */
void FatalError( char *msg )
{
  char buf[512];  
  sprintf( buf, "Fatal Error: %s\n", msg );
  fprintf( stderr, "%s\n", buf );
  exit(-1);
}

/* a non-fatal, non-terminating error */
void Error( char *msg )
{
  char buf[512];  
  sprintf( buf, "Error: %s\n", msg );
  fprintf( stderr, buf );
}

/* a warning to the user...  */
void Warning( char *msg )
{
  char buf[512];  
  sprintf( buf, "Warning: %s\n", msg );
  fprintf( stderr, buf );
}



/* read a texture from a file */
texture *ReadPPMTexture( char *infilename )
{
  FILE *infile;
  char string[256], buf[1000];
  int i, j, count=0;
  int img_max;
  long img_size;
  int r, g, b, bw;
  int c, mode, raw=0, ascii=0;
  texture *t;
  
  t = (texture *)malloc( sizeof ( texture ) );
  
  if (!t) FatalError( "Unable to allocate memory for texture structure!" );
    
  /* open file containing texture */
  if ((infile = fopen(infilename, "rb")) == NULL) {
    sprintf(buf, "LIBGFX: Can't open file '%s'!", infilename);
    FatalError( buf );
  }
  
  /* read and discard first line (the P3 or P6, etc...) */
  fgets(string, 256, infile);
  mode = string[1]-'0';
  if ((!IsValidMode(mode)) || (string[0] != 'P' && string[0] != 'p'))
    {
      sprintf(buf, "LIBGFX: Invalid PPM format specification in '%s'!", 
	      infilename);
      FatalError(buf);
    }
  raw = RawMode( mode );
  ascii = ASCIIMode( mode );

  /* discard all the comments at the top of the file */
  fgets(string, 256, infile);
  while (string[0] == '#')
    fgets(string, 256, infile);

  /* read image size and max component value */
  sscanf(string, "%d %d", &t->img_width, &t->img_height);
  
  /* PBMs are just 1's and 0's, so there's no max_component */
  if (mode != PBM_RAW && mode != PBM_ASCII)
    {
      fscanf(infile, "%d ", &img_max);
      if ((raw && img_max > 255) || (img_max <= 0))
	FatalError( "LIBGFX: Invalid value for maximum image color!" );
    }

  /* allocate texture array */
  img_size = 4*t->img_height*t->img_width;
  if ((t->texImage = (char *)calloc(img_size, sizeof(char))) == NULL)
    FatalError("LIBGFX: Cannot allocate memory for image!");
  
  /* read image data */
  c = 0;
  for (i=0; i<t->img_height; i++) {
    for (j=0; j<t->img_width; j++) {
      if (raw)
	{
	  if (mode==PBM_RAW)
	    {
	      int bit_pos;
	      if ((count%8)==0) bw=fgetc(infile);
	      bit_pos = 7-(count%8);
	      r=g=b=((bw & (1 << bit_pos))?0:255);
	      count++;
	      if ((j+1)==t->img_width) count=0;
	    }
	  else if (mode==PGM_RAW)
	    r=g=b=fgetc(infile);
	  else if (mode==PPM_RAW)
	    {
	      r = fgetc(infile);
	      g = fgetc(infile);
	      b = fgetc(infile);
	    }
	  t->texImage[c++] = r;
	  t->texImage[c++] = g;
	  t->texImage[c++] = b;
	  t->texImage[c++] = 255;
	}
      else /* then ASCII mode */
	{
	  if (mode==PBM_ASCII)
	    {
	      fscanf(infile, "%d", &r);
	      if (r!=1 && r!=0) r=0;
	      r=g=b=r*255;
	    }
	  else if (mode==PGM_ASCII)
	    {
	      fscanf(infile, "%d", &r);
	      r=g=b=r;
	    }
	  else if (mode==PPM_ASCII)
	    fscanf(infile, "%d %d %d", &r, &g, &b);
	  t->texImage[c++] = r;
	  t->texImage[c++] = g;
	  t->texImage[c++] = b;
	  t->texImage[c++] = 255;
	}
    }
  }
  
  t->type = GFXIO_UBYTE;
  t->format = GFXIO_RGBA;

  fclose( infile );
  
  return t;
}
