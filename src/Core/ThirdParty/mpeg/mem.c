/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/
/*
************************************************************
mem.c

This file contains the basic memory manipulation structures.

************************************************************
*/

/*LABEL mem.c */

#include <stdio.h>
#include "mem.h"

/*PUBLIC*/

extern void CopyMem();
extern ClearMem();
extern SetMem();
extern MEM *MakeMem();
extern void FreeMem();
extern MEM *LoadMem();
extern MEM *LoadPartialMem();
extern MEM *SaveMem();
extern MEM *SavePartialMem();

/*PRIVATE*/

#define DATALENGTH 100000 /*Maximum internal buffer*/

/*START*/

/*BFUNC

CopyMem() copies the entire contents of m2 to m1. 

EFUNC*/

void CopyMem(m1,m2)
     MEM *m1;
     MEM *m2;
{
  BEGIN("CopyMem");

  memcpy(m2->data,m1->data,m1->width*m1->height);
}

/*BFUNC

ClearMem() clears a memory structure by setting it to all zeroes.

EFUNC*/

ClearMem(m1)
     MEM *m1;
{
  BEGIN("ClearMem");

  memset(m1->data,0,m1->width*m1->height);
}

/*BFUNC

SetMem() clears a memory structure by setting it to all a value.

EFUNC*/

SetMem(value,m1)
     int value;
     MEM *m1;
{
  BEGIN("ClearMem");

  memset(m1->data,value,m1->width*m1->height);
}

/*BFUNC

MakeMem() creates a memory structure out of a given width and height.

EFUNC*/

MEM *MakeMem(width,height)
     int width;
     int height;
{
  BEGIN("MakeMem");
  MEM *temp;

  if (!(temp=MakeStructure(MEM)))
    {
      WHEREAMI();
      printf("Cannot create Memory structure.\n");
      exit(ERROR_MEMORY);
    }
  temp->len = width*height;
  temp->width = width;
  temp->height = height;
  if (!(temp->data=(unsigned char *)calloc(width*height,
					   sizeof(unsigned char))))
    {
      WHEREAMI();
      printf("Cannot allocate data storage for Memory structure.\n");
      exit(ERROR_MEMORY);
    }
  return(temp);
}

/*BFUNC

FreeMem() frees a memory structure.

EFUNC*/

void FreeMem(mem)
     MEM *mem;
{
  BEGIN("FreeMem");

  free(mem->data);
  free(mem);
}


/*BFUNC

LoadMem(width,height,)
     ) loads an Mem with a designated width; loads an Mem with a designated width, height, and
filename into a designated memory structure. If the memory structure
is NULL, one is created for it.

EFUNC*/

MEM *LoadMem(filename,width,height,omem)
     char *filename;
     int width;
     int height;
     MEM *omem;
{
  BEGIN("LoadMem");
  int length;
  MEM *temp;
  FILE *inp;

  if ((inp = fopen(filename,"r")) == NULL)
    {
      WHEREAMI();
      printf("Cannot open filename %s.\n",filename);
      exit(ERROR_BOUNDS);
    }
  fseek(inp,0,2);
  length = ftell(inp);
  rewind(inp);
  if ((width*height) != length)
    {
      WHEREAMI();
      printf("Bad Height and Width\n");
      exit(ERROR_BOUNDS);
    }
  if (omem) {temp=omem;}
  else {temp = MakeMem(width,height);}
  fread(temp->data,sizeof(unsigned char),temp->width*temp->height,inp);
  fclose(inp);
  return(temp);
}

/*BFUNC

LoadPartialMem(width,height,)
     ) loads an Mem with a designated width; loads an Mem with a designated width, height, and
filename into a designated memory structure. The file is of pwidth and
pheight, and if different than the width and height of the memory
structure, the structure is padded with 128's. If the memory structure
is NULL, one is created for it.

EFUNC*/

MEM *LoadPartialMem(filename,pwidth,pheight,width,height,omem)
     char *filename;
     int pwidth;
     int pheight;
     int width;
     int height;
     MEM *omem;
{
  BEGIN("LoadPartialMem");
  int i,length;
  unsigned char *bptr;
  MEM *temp;
  FILE *inp;

  if ((inp = fopen(filename,"r")) == NULL)
    {
      WHEREAMI();
      printf("Cannot open filename %s.\n",filename);
      exit(ERROR_BOUNDS);
    }
  fseek(inp,0,2);
  length = ftell(inp);
  rewind(inp);
  if ((pwidth*pheight) != length)
    {
      WHEREAMI();
      printf("Bad Height and Width\n");
      exit(ERROR_BOUNDS);
    }
  if (omem) {temp=omem;}
  else {temp = MakeMem(width,height);}
  for(bptr=temp->data,i=0;i<pheight;i++)
    {
      fread(bptr,sizeof(unsigned char),pwidth,inp);
      memset(bptr+pwidth,128,temp->width-pwidth);
      bptr += temp->width;
    }
  if (pheight<temp->height)
    {
      memset(temp->data+pheight*temp->width,128,
	     (temp->height-pheight)*temp->width);
    }

  fclose(inp);
  return(temp);
}

/*BFUNC

SaveMem() saves the designated memory structure to the appropriate
filename.

EFUNC*/

MEM *SaveMem(filename,mem)
     char *filename;
     MEM *mem;
{
  BEGIN("SaveMem");
  FILE *out;

  if ((out = fopen(filename,"w")) == NULL)
    {
      WHEREAMI();
      printf("Cannot open filename %s.\n",filename);
      exit(ERROR_BOUNDS);
    }
  fwrite(mem->data,sizeof(unsigned char),mem->width*mem->height,out);
  fclose(out);
  return(mem);
}

/*BFUNC

SavePartialMem() saves the designated memory structure to the appropriate
filename.

EFUNC*/

MEM *SavePartialMem(filename,pwidth,pheight,mem)
     char *filename;
     int pwidth;
     int pheight;
     MEM *mem;
{
  BEGIN("SavePartialMem");
  int i;
  unsigned char *bptr;
  FILE *out;

  if ((out = fopen(filename,"w")) == NULL)
    {
      WHEREAMI();
      printf("Cannot open filename %s.\n",filename);
      exit(ERROR_BOUNDS);
    }
  for(bptr=mem->data,i=0;i<pheight;i++)
    {
      fwrite(bptr,sizeof(unsigned char),pwidth,out);
      bptr += mem->width;
    }
  fclose(out);
  return(mem);
}

/*END*/
