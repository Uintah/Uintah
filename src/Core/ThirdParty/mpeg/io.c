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
io.c

This is the IO/ motion frame stuff.  It is derived from the JPEG io.c
package, so it is somewhat unsuited for fixed-frame ratio fixed
component number MPEG work.

************************************************************
*/

/*LABEL io.c */

#include <stdio.h>
#include "globals.h"

/* added by Cameron */

extern void SCIReadFS(unsigned char* imageY,
		      unsigned char* imageU,
		      unsigned char* imageV);
extern int HorizontalSize;     /* from mpeg.c */
extern int VerticalSize;

/*PUBLIC*/

extern void MakeFS();
extern void SuperSubCompensate();
extern void SubCompensate();
extern void AddCompensate();
extern void Sub2Compensate();
extern void Add2Compensate();
extern void MakeMask();
extern void ClearFS();
extern void InitFS();
extern void ReadFS();
extern void InstallIob();
extern void InstallFSIob();
extern void WriteFS();
extern void MoveTo();
extern int Bpos();
extern void ReadBlock();
extern void WriteBlock();
extern void PrintIob();

static void Get4Ptr();
static void Get2Ptr();

/*PRIVATE*/

extern IMAGE *CImage;
extern FRAME *CFrame;
extern FSTORE *CFStore;

extern int FrameInterval;
extern int FrameDistance;

extern int MVDH;
extern int MVDV;
extern int MX;
extern int MY;
extern int NX;
extern int NY;

extern int Loud;
extern int ImageType;

int BlockWidth = BLOCKWIDTH;
int BlockHeight = BLOCKHEIGHT;

static int Mask[64];
static int Nask[64];

IOBUF *Iob;


/*START*/

/*BFUNC

MakeFS() constructs an IO structure and assorted book-keeping
instructions for all components of the frame.

EFUNC*/

void MakeFS(flag)
     int flag;
{
  BEGIN("MakeFS");
  int i;

  CFStore = MakeStructure(FSTORE);
  CFStore->NumberComponents=CFrame->NumberComponents;
  for(i=0;i<CFStore->NumberComponents;i++)
    {
      if (!(CFStore->Iob[i]=MakeStructure(IOBUF)))
	{
	  WHEREAMI();
	  printf("Cannot make IO structure\n");
	  exit(ERROR_MEMORY);
	}
      CFStore->Iob[i]->flag = flag;
      CFStore->Iob[i]->hpos = 0;
      CFStore->Iob[i]->vpos = 0;
      CFStore->Iob[i]->hor = CFrame->hf[i];
      CFStore->Iob[i]->ver = CFrame->vf[i];
      CFStore->Iob[i]->width = CFrame->Width[i];
      CFStore->Iob[i]->height = CFrame->Height[i];
      CFStore->Iob[i]->mem = MakeMem(CFrame->Width[i],
				    CFrame->Height[i]);
    }      
}

/*BFUNC

SuperSubCompensate(arrays,)
     ) subtracts off the compensation from three arrays; subtracts off the compensation from three arrays,
forward compensation from the first, backward from the second,
interpolated from the third. This is done with a corresponding portion
of the memory in the forward and backward IO buffers.

EFUNC*/

void SuperSubCompensate(fmcmatrix,bmcmatrix,imcmatrix,XIob,YIob)
     int *fmcmatrix;
     int *bmcmatrix;
     	int *imcmatrix;
     IOBUF *XIob;
     IOBUF *YIob;
{
  BEGIN("SuperSubCompensate");
  int i,/*a,b,*/val;
  int *mask,*nask;

  MakeMask(MX, MY, Mask, XIob);
  MakeMask(NX, NY, Nask, YIob);

  /* Old stuff pre-SantaClara */
/*
  a = (16*(FrameInterval - FrameDistance) + FrameInterval/2)/FrameInterval;
  b = 16 - a;
*/

  for(mask=Mask,nask=Nask,i=0;i<64;i++)
    {
      *fmcmatrix++ -= *mask;
      *bmcmatrix++ -= *nask;

      /* Old stuff pre-SantaClara */
      /* 
      val = a*(*mask++) + b *(*nask++);
      if (val > 0) {val = (val+8)/16;}
      else {val = (val-8)/16;}
      */
      /* Should always be positive */
      val = ((*mask++)+(*nask++)+1)>>1;
      *imcmatrix++ -= val;
    }
}

/*BFUNC

Sub2Compensate() does a subtraction of the prediction from the current
matrix with a corresponding portion of the memory in the forward and
backward IO buffers.

EFUNC*/

void Sub2Compensate(matrix,XIob,YIob)
     int *matrix;
     IOBUF *XIob;
     IOBUF *YIob;
{
  BEGIN("Sub2Compensate");
  int i,/*a,b,*/val;
  int *mask,*nask;

  MakeMask(MX, MY, Mask, XIob);
  MakeMask(NX, NY, Nask, YIob);

  /* Old stuff pre-SantaClara */
  /*
  a = (16*(FrameInterval - FrameDistance) + FrameInterval/2)/FrameInterval;
  b = 16 - a;
  */

  for(mask=Mask,nask=Nask,i=0;i<64;i++)
    {
      /* Old stuff pre-SantaClara */
      /* 
      val = a*(*mask++) + b *(*nask++);
      if (val > 0) {val = (val+8)/16;}
      else {val = (val-8)/16;}
      */
      val = ((*mask++)+(*nask++)+1)>>1;
      *matrix++ -= val;
    }
}


/*BFUNC

Add2Compensate() does an addition of the prediction from the current
matrix with a corresponding portion of the memory in the forward and
backward IO buffers.

EFUNC*/


void Add2Compensate(matrix,XIob,YIob)
     int *matrix;
     IOBUF *XIob;
     IOBUF *YIob;
{
  BEGIN("Add2Compensate");
  int i,/*a,b,*/val;
  int *mask,*nask;
  
  MakeMask(MX, MY, Mask, XIob);
  MakeMask(NX, NY, Nask, YIob);

  /* Old stuff pre-SantaClara */
  /*
  a = (16*(FrameInterval - FrameDistance) + FrameInterval/2)/FrameInterval;
  b = 16 - a;
  */
  
  for(mask=Mask,nask=Nask,i=0;i<64;i++)
    {
      /* Old stuff pre-SantaClara */
      /*
      val = a*(*mask++) + b *(*nask++);
      if (val > 0) {val = (val+8)/16;}
      else {val = (val-8)/16;}
      */
      val = ((*mask++)+(*nask++)+1)>>1;
      *matrix += val;
      if (*matrix>255) {*matrix=255;}
      else if (*matrix<0){*matrix=0;}
      matrix++;
    }
}

/*BFUNC

SubCompensate() does a subtraction of the prediction from the current
matrix with a corresponding portion of the memory in the target IO
buffer.

EFUNC*/

void SubCompensate(matrix,XIob)
     int *matrix;
     IOBUF *XIob;
{
  BEGIN("SubCompensate");
  int i;
  int *mask;

  MakeMask(MX, MY, Mask, XIob);

  for(mask=Mask,i=0;i<64;i++)
    *matrix++ -= *mask++;
}


/*BFUNC

AddCompensate() does an addition of the prediction from the current
matrix with a corresponding portion of the memory in the target IO
buffer.

EFUNC*/

void AddCompensate(matrix,XIob)
     int *matrix;
     IOBUF *XIob;
{
  BEGIN("AddCompensate");
  int i;
  int *mask;

  MakeMask(MX, MY, Mask, XIob);
  for(mask=Mask,i=0;i<64;i++)
    {
      *matrix += *mask++;
      if (*matrix>255) {*matrix=255;}
      else if (*matrix<0){*matrix=0;}
      matrix++;
    }
}

void MakeMask(x,y,mask,XIob)
     int x;
     int y;
     int *mask;
     IOBUF *XIob;
{
  BEGIN("MakeMask");
  int i,j,rx,ry,dx,dy;
  unsigned char *aptr,*bptr,*cptr,*dptr;

  rx = x>>1;  ry = y>>1;
  dx = x&1;  dy = y&1;
  aptr = (((XIob->vpos *  BlockHeight) + ry)*XIob->width)
    + (XIob->hpos * BlockWidth) + rx
      + XIob->mem->data;
  if (dx)
    {
      bptr = aptr + dx;
      if (dy)
	{
	  cptr = aptr + XIob->width;
	  dptr = cptr + dx;
	  Get4Ptr(XIob->width,mask,aptr,bptr,cptr,dptr);
	}
      else
	{
	  Get2Ptr(XIob->width,mask,aptr,bptr);
	}
    }
  else if (dy)
    {
      cptr = aptr + XIob->width;
      Get2Ptr(XIob->width,mask,aptr,cptr);
    }
  else
    {
      for(i=0;i<BlockHeight;i++)
	{
	  for(j=0;j<BlockWidth;j++)
	    *(mask++) = *aptr++;
	  aptr = aptr-BlockWidth+XIob->width;
	}
    }
}

static void Get4Ptr(width,matrix,aptr,bptr,cptr,dptr)
     int width;
     int *matrix;
     unsigned char *aptr;
     unsigned char *bptr;
     unsigned char *cptr;
     unsigned char *dptr;
{
  int i,j;

  for(i=0;i<BlockHeight;i++) /* should be unrolled */
    {
      for(j=0;j<BlockWidth;j++)
	{
	  *(matrix++) = ((((int)*aptr++) + ((int)*bptr++) +
			  ((int)*cptr++) + ((int)*dptr++) + 2) >> 2);
	}
      aptr = aptr-BlockWidth+width;
      bptr = bptr-BlockWidth+width;
      cptr = cptr-BlockWidth+width;
      dptr = dptr-BlockWidth+width;
    }
}


static void Get2Ptr(width,matrix,aptr,bptr)
     int width;
     int *matrix;
     unsigned char *aptr;
     unsigned char *bptr;
{
  int i,j;

  for(i=0;i<BlockHeight;i++)  /* should be unrolled */
    {
      for(j=0;j<BlockWidth;j++)
	{
	  *(matrix++) = ((((int) *aptr++) + ((int)*bptr++) + 1) >> 1);
	}
      aptr = aptr-BlockWidth+width;
      bptr = bptr-BlockWidth+width;
    }
}

/*BFUNC

CopyCFS2FS() copies all of the CFrame Iob's to a given frame store.

EFUNC*/

void CopyCFS2FS(fs)
     FSTORE *fs;
{
  BEGIN("CopyIob2FS");
  int i;

  for(i=0;i<CFStore->NumberComponents;i++)
    CopyMem(CFStore->Iob[i]->mem,fs->Iob[i]->mem);
}

/*BFUNC

ClearFS() clears the entire frame store passed into it.

EFUNC*/

void ClearFS()
{
  BEGIN("ClearFS");
  int i;

  for(i=0;i<CFStore->NumberComponents;i++)
    ClearMem(CFStore->Iob[i]->mem);
}

/*BFUNC

InitFS() initializes a frame store that is passed into it. It creates
the IO structures and the memory structures.

EFUNC*/

void InitFS()
{
  BEGIN("InitFS");
  int i;

  for(i=0;i<CFStore->NumberComponents;i++)
    {
      if (!(CFStore->Iob[i]=MakeStructure(IOBUF)))
	{
	  WHEREAMI();
	  printf("Cannot create IO structure.\n");
	  exit(ERROR_MEMORY);
	}
      CFStore->Iob[i]->flag = 0;
      CFStore->Iob[i]->hpos = 0;
      CFStore->Iob[i]->vpos = 0;
      CFStore->Iob[i]->hor = CFrame->hf[i];
      CFStore->Iob[i]->ver = CFrame->vf[i];
      CFStore->Iob[i]->width = CFrame->Width[i];
      CFStore->Iob[i]->height = CFrame->Height[i];
      CFStore->Iob[i]->mem = MakeMem(CFrame->Width[i],
				CFrame->Height[i]);
    }
}

/*BFUNC

ReadFS() loads the memory images from the filenames designated in the
CFrame structure.

EFUNC*/

void ReadFS()
{
  BEGIN("ReadFS");
  int i;

  printf("this fuction shouldn't be called for SCI-anything.\n");
  /*
  for(i=0;i<CFrame->NumberComponents;i++)
    {
      if (CImage->PartialFrame)
	CFStore->Iob[i]->mem = LoadPartialMem(CFrame->ComponentFileName[i],
					      CFrame->PWidth[i],
					      CFrame->PHeight[i],
					      CFrame->Width[i],
					      CFrame->Height[i],
					      CFStore->Iob[i]->mem);
      else
	CFStore->Iob[i]->mem = LoadMem(CFrame->ComponentFileName[i],
				       CFrame->Width[i],
				       CFrame->Height[i],
				       CFStore->Iob[i]->mem);
    }
    */
  /*  //bcopy(imageY, CFStore->Iob[0]->mem->data, sizeX*sizeY); 
  //bcopy(imageU, CFStore->Iob[1]->mem->data, sizeX*sizeY/2);
  //bcopy(imageV, CFStore->Iob[2]->mem->data, sizeX*sizeY/2);
  */
}

void SCIReadFS(unsigned char* imageY, unsigned char* imageU,
	       unsigned char* imageV)
{
  BEGIN("ReadFS");
  int i;

  /*
  for(i=0;i<CFrame->NumberComponents;i++)
    {
      if (CImage->PartialFrame)
	CFStore->Iob[i]->mem = LoadPartialMem(CFrame->ComponentFileName[i],
					      CFrame->PWidth[i],
					      CFrame->PHeight[i],
					      CFrame->Width[i],
					      CFrame->Height[i],
					      CFStore->Iob[i]->mem);
      else
	CFStore->Iob[i]->mem = LoadMem(CFrame->ComponentFileName[i],
				       CFrame->Width[i],
				       CFrame->Height[i],
				       CFStore->Iob[i]->mem);
    }
    */
  int sizeX = HorizontalSize;
  int sizeY = VerticalSize;
  bcopy(imageY, CFStore->Iob[0]->mem->data, sizeX*sizeY); 
  bcopy(imageU, CFStore->Iob[1]->mem->data, sizeX*sizeY/4);
  bcopy(imageV, CFStore->Iob[2]->mem->data, sizeX*sizeY/4); 
}
/*BFUNC

InstallIob() installs a particular CFrame Iob as the target Iob.

EFUNC*/

void InstallIob(index)
     int index;
{
  BEGIN("InstallIob");

  Iob = CFStore->Iob[index];
}

void InstallFSIob(fs,index)
     FSTORE *fs;
     int index;
{
  Iob = fs->Iob[index];
}


/*BFUNC

WriteFS() writes the frame store out.

EFUNC*/

void WriteFS()
{
  BEGIN("WriteIob");
  int i;

  for(i=0;i<CFrame->NumberComponents;i++)
    {
      if (CImage->PartialFrame)
	SavePartialMem(CFrame->ComponentFileName[i],
		       CFrame->PWidth[i],
		       CFrame->PHeight[i],
		       CFStore->Iob[i]->mem);
      else
	SaveMem(CFrame->ComponentFileName[i],CFStore->Iob[i]->mem);
    }  
}

/*BFUNC

MoveTo() moves the installed Iob to a given location designated by the
horizontal and vertical offsets.

EFUNC*/

void MoveTo(hp,vp,h,v)
     int hp;
     int vp;
     int h;
     int v;
{
  BEGIN("MoveTo");

  Iob->hpos = hp*Iob->hor + h;
  Iob->vpos = vp*Iob->ver + v;
}

/*BFUNC

Bpos() returns the designated MDU number inside of the frame of the
installed Iob given by the input gob, mdu, horizontal and vertical
offset. It returns 0 on error.

EFUNC*/

int Bpos(hp,vp,h,v)
     int hp;
     int vp;
     int h;
     int v;
{
  BEGIN("Bpos");

  return((vp*Iob->ver + v)*(Iob->width/BlockWidth) + 
	 (hp * Iob->hor + h));
}


/*BFUNC

ReadBlock() reads a block from the currently installed Iob into a
designated matrix.

EFUNC*/

void ReadBlock(store)
     int *store;
{
  BEGIN("ReadBlock");
  int i,j;
  unsigned char *loc;

  loc = Iob->vpos*Iob->width*BlockHeight
    + Iob->hpos*BlockWidth+Iob->mem->data;
  for(i=0;i<BlockHeight;i++)
    {
      for(j=0;j<BlockWidth;j++) {*(store++) = *(loc++);}
      loc += Iob->width - BlockWidth;
    }
  if ((++Iob->hpos % Iob->hor)==0)
    {
      if ((++Iob->vpos % Iob->ver) == 0)
	{
	  if (Iob->hpos < 
	      ((Iob->width - 1)/(BlockWidth*Iob->hor))*Iob->hor + 1)
	    {
	      Iob->vpos -= Iob->ver;
	    }
	  else {Iob->hpos = 0;}
	}
      else {Iob->hpos -= Iob->hor;}
    }
}

/*BFUNC

WriteBlock() writes a input matrix as a block into the currently
designated IOB structure.

EFUNC*/

void WriteBlock(store)
     int *store;
{
  int i,j;
  unsigned char *loc;

  loc = Iob->vpos*Iob->width*BlockHeight +
    Iob->hpos*BlockWidth+Iob->mem->data;
  for(i=0;i<BlockHeight;i++)
    {
      for(j=0;j<BlockWidth;j++)	{*(loc++) =  *(store++);}
      loc += Iob->width - BlockWidth;
    }
  if ((++Iob->hpos % Iob->hor)==0)
    {
      if ((++Iob->vpos % Iob->ver) == 0)
	{
	  if (Iob->hpos < 
	      ((Iob->width - 1)/(BlockWidth*Iob->hor))*Iob->hor + 1)
	    {
	      Iob->vpos -= Iob->ver;
	    }
	  else {Iob->hpos = 0;}
	}
      else {Iob->hpos -= Iob->hor;}
    }
}

/*BFUNC

PrintIob() prints out the current Iob structure to the standard output
device.

EFUNC*/

void PrintIob()
{
  printf("IOB: %x\n",Iob);
  if (Iob)
    {
      printf("hor: %d  ver: %d  width: %d  height: %d\n",
	     Iob->hor,Iob->ver,Iob->width,Iob->height);
      printf("flag: %d  Memory Structure: %x\n",Iob->flag,Iob->mem);
    }
}

/*END*/

