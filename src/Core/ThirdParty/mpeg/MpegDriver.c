/*
 * Driver program for mpeg encoder.
 *
 * Cameron Christensen
 * SCI Group
 * University of Utah
 * February 1999
 */

#include <stdio.h>
#include <math.h>
/* link against mpeg.c */

extern int StartMpegEncoder(int, char**);
extern void SCIRunEncode(int lasttime, unsigned char* imageY,
			 unsigned char* imageU, unsigned char* imageV);

int main() {
  int sizeX = 320;
  int sizeY = 200;

  /* raster */
  unsigned char R[sizeX][sizeY];
  unsigned char G[sizeX][sizeY];
  unsigned char B[sizeX][sizeY];
  unsigned char *imageY;
  unsigned char *imageU;
  unsigned char *imageV;
  
  int checkersize = 2;
  int i=0, j=0, k;
  double c;
  double period;
  int godown = 0;
    
  /* create s_argc, s_argv for start function */
  int s_argc = 0;
  char** s_argv = (char**)malloc(s_argc*sizeof(char*));
  s_argv[s_argc] = (char*)malloc(3);
  strcpy(s_argv[s_argc++], "-a");
  s_argv[s_argc] = (char*)malloc(2);
  strcpy(s_argv[s_argc++], "0");
  s_argv[s_argc] = (char*)malloc(3);
  strcpy(s_argv[s_argc++], "-b");
  s_argv[s_argc] = (char*)malloc(2);
  strcpy(s_argv[s_argc++], "10000"); /* 10,000 will cause lots/no problems.*/
  s_argv[s_argc] = (char*)malloc(3);
  strcpy(s_argv[s_argc++], "-h");
  s_argv[s_argc] = (char*)malloc(4);
  strcpy(s_argv[s_argc++], "320");
  s_argv[s_argc] = (char*)malloc(3);
  strcpy(s_argv[s_argc++], "-v");
  s_argv[s_argc] = (char*)malloc(4);
  strcpy(s_argv[s_argc++], "200");
  s_argv[s_argc] = (char*)malloc(4);
  strcpy(s_argv[s_argc++], "-PF");
  s_argv[s_argc] = (char*)malloc(3);
  strcpy(s_argv[s_argc++], "-s");
  s_argv[s_argc] = (char*)malloc(8);
  strcpy(s_argv[s_argc++], "out.mpg");
    
  /* call StartMpegEncoder to begin */
  StartMpegEncoder(s_argc, s_argv);
  
  /* create images */
  imageY = (unsigned char*)malloc(sizeX*sizeY);
  imageU = (unsigned char*)malloc(sizeX*sizeY/4);
  imageV = (unsigned char*)malloc(sizeX*sizeY/4);
    
  while(checkersize > 0) /*creating_images)*/ {
    printf("encoding image (checkersize=%i)...\n",checkersize);
    if(!godown) {
      if(checkersize > 100)
	godown = 1;
      period = 2.0*M_PI/checkersize;
      checkersize+=5;
    }
    else {
     period = 2.0*M_PI/checkersize;
     checkersize -= 5;

    }

    printf("Encoding RGB...\n");
    /* create RGB frame */
    for(i=0; i<sizeY; i++)
      for(j=0; j<sizeX; j++) {
	
	c = sin(i*period) * sin(j*period);
	R[j][i] = G[j][i] = B [j][i] = (c > 0) ? 0 : 255;
	
	/*R[j][i] = G[j][i] = B [j][i] = j % checkersize * i % checkersize;*/
      }
    printf("Encoding YUV...\n");
    /* convert to YUV frame */
    for(i=0; i<sizeY; i++) {
      for(j=0; j<sizeX; j++) {
	
	imageY[i*sizeX+j] =
	  0.299*R[j][i] + 0.587*G[j][i] + 0.114*B[j][i];
	  
	/*
	imageY[i*sizeX+j] =
	  0.257*R[j][i] + 0.504*G[j][i] + 0.098*B[j][i] + 16;
	  */
      }
    }
    for(i=0; i<sizeY; i+=2) 
      for(k=0; k<sizeX; k+=2) {
	
	imageU[(i/2)*(sizeX/2)+k/2] =
	  -1*0.1687*R[k][i] - 0.3313*G[k][i] + 0.5*B[k][i];
	imageV[(i/2)*(sizeX/2)+k/2] =
	  0.5*R[k][i] - 0.4187*G[k][i] - 0.0813*B[k][i];
	  
	/*
	imageU[i*sizeX/2+k] =
	  0.439*R[k][i] - 0.368*G[k][i] - 0.071*B[k][i] + 128;
	imageV[i*sizeX/2+k] =
	  -1*0.148*R[k+1][i]-0.291*G[k+1][i] + 0.439*B[k+1][i] + 128;
	  */
      }
     	
    
    /* call SCIRunEncode(0) after each image is generated. */
    printf("Calling SCIRunEncode()...\n");
    SCIRunEncode(0, imageY, imageU, imageV);
  }

  /* call SCIRunEncode(1) to encode remaining images and dump file. */
  SCIRunEncode(1, imageY, imageU, imageV);

  return 0;

}
