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
me.c

This file does much of the motion estimation and compensation.

************************************************************
*/

/*LABEL me.c */

#include "globals.h"

/*PUBLIC*/

extern void initme();
extern void HPFastBME();
extern void BruteMotionEstimation();
extern void InterpolativeBME();

static int Do4Check();
static int Do2Check();

/*PRIVATE*/

static int MinX,MaxX,BMinX,BMaxX;
static int MinY,MaxY,BMinY,BMaxY;
static int CircleLimit;

static int VAR;
static int VAROR;
static int MWOR;

int MeVAR[1024];
int MeVAROR[1024];
int MeMWOR[1024];
int MX;
int MY;
int NX;  /* Useless here, but used elsewhere */
int NY; 
int MV;
int MeX[1024];
int MeY[1024];
int MeVal[1024];
int MeN=0;

int **FMX;
int **BMX;
int **FMY;
int **BMY;

extern MEM **FFS;
int SearchLimit = 15;
int MVPrediction = 0;  /* Sets some complicated prediction */
                       /* experimental, unknown effects */
int MVTelescope=1;     /* Sets telescopic motion estimation (default)*/
                       /* esp good for small search windows */
BLOCK nb,rb;

extern int FrameInterval;

#define COMPARISON >=  /* This is to compare for short-circuit exit */

/*START*/

/*BFUNC

initme() initializes the motion estimation to the proper number of
estimated frames, by FrameInterval.

EFUNC*/

void initme()
{
  BEGIN("initme");
  int i;

  FMX = (int **) calloc(FrameInterval+1,sizeof(int *));
  BMX = (int **) calloc(FrameInterval+1,sizeof(int *));
  FMY = (int **) calloc(FrameInterval+1,sizeof(int *));
  BMY = (int **) calloc(FrameInterval+1,sizeof(int *));

  for(i=0;i<FrameInterval+1;i++)
    {
      FMX[i] = (int *) calloc(8192,sizeof(int));
      BMX[i] = (int *) calloc(8192,sizeof(int));
      FMY[i] = (int *) calloc(8192,sizeof(int));
      BMY[i] = (int *) calloc(8192,sizeof(int));
     }
}


static int Do4Check(aptr,bptr,cptr,dptr,eptr,width,lim)
     unsigned char *aptr;
     unsigned char *bptr;
     unsigned char *cptr;
     unsigned char *dptr;
     unsigned char *eptr;
     int width;
     int lim;
{
  BEGIN("Do4Check");
  int val,i,data;
  for(val=0,i=0;i<16;i++)
    {
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + *dptr++ + *eptr++ + 2) >> 2));
      if (data<0) {val-=data;} else {val+=data;}
      if (val COMPARISON lim) return(val+1);
      aptr += (width - 16);
      bptr += (width - 16);
      cptr += (width - 16);
      dptr += (width - 16);
      eptr += (width - 16);
    }
  return(val);
}

static int Do2Check(aptr,bptr,cptr,width,lim)
     unsigned char *aptr;
     unsigned char *bptr;
     unsigned char *cptr;
     int width;
     int lim;
{
  BEGIN("Do2Check");
  int val,i,data;

  for(val=0,i=0;i<16;i++)
    {
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      data=(*(aptr++) - ((*bptr++ + *cptr++ + 1) >> 1));
      if (data<0) {val-=data;} else {val+=data;}
      if (val COMPARISON lim) return(val+1);
      aptr += (width - 16);
      bptr += (width - 16);
      cptr += (width - 16);
    }
  return(val);
}

/*BFUNC

HPFastBME() does a fast brute-force motion estimation with two indexes
into two memory structures. The motion estimation has a short-circuit
abort to speed up calculation.

EFUNC*/

unsigned char LocalX[272];
unsigned char LocalY[272];
unsigned char LocalXY[289];
unsigned char *lptr,*aptr,*dptr,*eptr;

void HPFastBME(rx,ry,rm,cx,cy,cm,ox,oy)
     int rx;
     int ry;
     MEM *rm;
     int cx;
     int cy;
     MEM *cm;
     int ox;
     int oy;
{
  BEGIN("HPFastBME");
  int dx,dy,lx,ly,px,py,incr,xdir,ydir;
  register int i,j,data,val;
  register unsigned char *bptr,*cptr;
  unsigned char *baseptr;

  if ((ox < MinX)||(ox > MaxX))
    {
      WHEREAMI();
      printf("X coord out of bounds [%d,%d]: %d\n",MinX,MaxX,ox);
    }
  if ((oy < MinY)||(oy > MaxY))
    {
      WHEREAMI();
      printf("Y coord out of bounds [%d,%d]: %d\n",MinY,MaxY,oy);
    }
  MX=px=ox; MY=py=oy;                       /* Start search point at offset */
  MV=0;
  bptr=rm->data + (rx+ox) + ((ry+oy) * rm->width);
  baseptr=cm->data + cx + (cy * cm->width);
  cptr=baseptr;
  for(i=0;i<16;i++)                    /* Calculate [ox,oy] compensation */
    {
      for(j=0;j<16;j++)
	{
	  data=(*(bptr++)-*(cptr++));
	  if (data<0) {MV-=data;} else {MV+=data;}
	}
      bptr += (rm->width - 16);
      cptr += (cm->width - 16);
    }
  xdir=ydir=0;
  for(incr=1;incr<CircleLimit;incr++)
    {
      if ((py > (MaxY))||(py < MinY))
	{
	  if (xdir) px += incr;
	  else px -= incr;
	}
      else
	{
	  for(dx=0;dx<incr;dx++)
	    {
	      if (xdir) {px++;} else {px--;}         /* Move search point */
	      if ((px > (MaxX)) || (px < (MinX)))
		continue;                            /* check logical bds */
	      lx = px+rx; ly = py+ry;
	      if (((lx >= 0) && (lx < rm->width-16)) &&  /* check phys. bds */
		  ((ly >= 0) && (ly < rm->height-16)))
		{
		  bptr = rm->data + lx + (ly * rm->width);
		  cptr = baseptr;
		  for(val=i=0;i<16;i++)
		    {
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      if (val COMPARISON MV) break;
		      bptr += (rm->width - 16);
		      cptr += (cm->width - 16);
		    }
		  if (val < MV)
		    {
		      MV = val; 
		      MX = px;
		      MY = py;
		    }
		}
	    }
	}
      xdir = 1-xdir;
      if ((px > (MaxX))||(px < MinX))
	{
	  if (ydir) py += incr;
	  else py -= incr;
	}
      else
	{
	  for(dy=0;dy<incr;dy++)
	    {
	      if (ydir) {py++;} else {py--;}  /* Move search point */
	      if ((py > (MaxY)) || (py < (MinY))) continue;
	      lx = px+rx; ly = py+ry;
	      if (((lx >= 0) && (lx <= rm->width-16)) &&
		  ((ly >= 0) && (ly <= rm->height-16)))
		{
		  bptr = rm->data + lx + (ly * rm->width);
		  cptr = baseptr;
		  for(val=i=0;i<16;i++)
		    {
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      data=(*(bptr++)-*(cptr++));
		      if (data<0) {val-=data;} else	{val+=data;}
		      if (val COMPARISON MV) break;
		      bptr += (rm->width - 16);
		      cptr += (cm->width - 16);
		    }
		  if (val < MV)
		    {
		      MV = val; 
		      MX = px;
		      MY = py;
		    }
		}
	    }
	}
      ydir = 1-ydir;
    }
  /* At this point, MX and MY contain the integer mc vectors. */
  /* Look at nearest neighboring pels; dx and dy hold the offset. */

  dx=dy=0;
  aptr = baseptr;
  bptr = rm->data + (MX + rx) + ((MY + ry) * rm->width);
  if (((2*MX-1)>=BMinX)&&((2*MY-1)>=BMinY)&&
      ((MX+rx)>0) && ((MY+ry)>0))
    {
      cptr = bptr-1;
      dptr = bptr-rm->width;
      eptr = dptr-1;
      val = Do4Check(aptr,bptr,cptr,dptr,eptr,rm->width,MV);
      if (val < MV)
	{
	  MV = val; 
	  dx = dy = -1;
	}
    }
  if (((2*MY-1)>=BMinY)&&((MY+ry)>0))
    {
      cptr = bptr-rm->width;
      val = Do2Check(aptr,bptr,cptr,rm->width,MV);
      if (val < MV)
	{
	  MV = val; 
	  dx = 0; dy = -1;
	}
    }
  if (((2*MX+1)<= BMaxX)&&((2*MY-1)>=BMinY)&&
      ((MX+rx+16)<rm->width) && ((MY+ry)>0))
    {
      cptr = bptr+1;
      dptr = bptr-rm->width;
      eptr = dptr+1;
      val = Do4Check(aptr,bptr,cptr,dptr,eptr,rm->width,MV);
      if (val < MV)
	{
	  MV = val; 
	  dx = 1; dy = -1;
	}
    }
  if (((2*MX-1)>=BMinX)&&((MX+rx) > 0))
    {
      cptr = bptr-1;
      val = Do2Check(aptr,bptr,cptr,rm->width,MV);
      if (val < MV)
	{
	  MV = val; 
	  dx = -1; dy = 0;
	}
    }
  if (((2*MX+1)<=BMaxX)&&((MX+rx+16)<rm->width))
    {
      cptr = bptr+1;
      val = Do2Check(aptr,bptr,cptr,rm->width,MV);
      if (val < MV)
	{
	  MV = val; 
	  dx = 1; dy = 0;
	}
    }
  if (((2*MX-1)>=BMinX)&& ((2*MY+1)<=BMaxY)&&
      ((MX+rx)>0) && ((MY+ry+16)<rm->height))
    {
      cptr = bptr-1;
      dptr = bptr+rm->width;
      eptr = dptr-1;
      val = Do4Check(aptr,bptr,cptr,dptr,eptr,rm->width,MV);
      if (val < MV)
	{
	  MV = val; 
	  dx = -1; dy = 1;
	}
    }
  if (((2*MY+1)<=BMaxY)&&((MY+ry+16)<rm->height))
    {
      cptr = bptr+rm->width;
      val = Do2Check(aptr,bptr,cptr,rm->width,MV);
      if (val < MV)
	{
	  MV = val; 
	  dx = 0; dy = 1;
	}
    }
  if (((2*MY+1)<=BMaxY)&&((2*MX+1)<=BMaxX)&&
      ((MX+rx+16)<rm->width) && ((MY+ry+16) < rm->height))
    {
      cptr = bptr+1;
      dptr = bptr+rm->width;
      eptr = dptr+1;
      val = Do4Check(aptr,bptr,cptr,dptr,eptr,rm->width,MV);
      if (val < MV)
	{
	  MV = val; 
	  dx = dy = 1;
	}
    }
  MX = MX*2 + dx;
  MY = MY*2 + dy;
}

/*BFUNC

BruteMotionEstimation() does a brute-force motion estimation on all
aligned 16x16 blocks in two memory structures. It is presented as a
compatibility-check routine.

EFUNC*/

void BruteMotionEstimation(pmem,fmem)
     MEM *pmem;
     MEM *fmem;
{
  BEGIN("BruteMotionEstimation");
  int x,y;

  CircleLimit=SearchLimit;
  for(MeN=0,y=0;y<fmem->height;y+=16)
    {
      for(x=0;x<fmem->width;x+=16)
	{
	  HPFastBME(x,y,pmem,x,y,fmem,0,0);
	  MeVAR[MeN] = VAR;
	  MeVAROR[MeN] = VAROR;
	  MeMWOR[MeN] = MWOR;
	  MeX[MeN] = MX;
	  MeY[MeN] = MY;
	  MeVal[MeN] = MV;
	  MeN++;
	}
    }
}

/*BFUNC

InterpolativeBME() does the interpolative block motion estimation for
an entire frame interval at once.  Although motion estimation can be
done sequentially with considerable success, the temporal and spatial
locality of doing it all at once is probably better.

EFUNC*/

void InterpolativeBME()
{
  BEGIN("InterpolativeBME");
  int i,dx,dy,rx,ry,x,y,n;

  /* Do first forward predictive frame */
  if (FrameInterval)
    {
      MaxX = MaxY = 7;
      MinX = MinY = -8;
      BMaxX = BMaxY = 15;
      BMinX = BMinY = -16;
      printf("Doing Forward: 1\n");
      for(n=y=0;y<FFS[0]->height;y+=16)
	{
	  /* printf("Y:%d",y); */
	  for(rx=ry=dx=dy=x=0;x<FFS[0]->width;x+=16)
	    {
	      CircleLimit = SearchLimit + MAX(rx,ry)+1;
	      HPFastBME(x,y,FFS[0],x,y,FFS[1],dx/2,dy/2);
	      FMX[1][n] = MX;
	      FMY[1][n] = MY;
	      /* printf("[%d:%d]",MX,MY);*/
	      if (MVPrediction)
		{
		  dx = MX; dy = MY;
		  if (dx < BMinX) dx = BMinX;
		  else if (dx > MaxX) dx = BMaxX;
		  if (dy < BMinY) dy = BMinY;
		  else if (dy > MaxY) dy = BMaxY;
		  rx = abs(dx); ry = abs(dy);
		}
	      n++;
	    }
	  /* printf("\n");*/
	}
      for(i=2;i<=FrameInterval;i++)
	{
	  MaxX = MaxY = (i<<3)-1;
	  MinX = MinY = -(i<<3);
	  BMaxX = BMaxY = (i<<4)-1;
	  BMinX = BMinY =  -(i<<4);
	  printf("Doing Forward: %d\n",i);
	  for(n=0,y=0;y<FFS[0]->height;y+=16)
	    {
	      /* printf("Y:%d",y);*/
	      for(rx=ry=dx=dy=x=0;x<FFS[0]->width;x+=16)
		{
		  CircleLimit = SearchLimit + MAX(rx,ry)+1;
		  HPFastBME(x,y,FFS[0],x,y,FFS[i],dx/2,dy/2);
		  FMX[i][n] = MX;
		  FMY[i][n] = MY;
		  /* printf("[%d:%d]",MX,MY);*/
		  if (MVPrediction)
		    {
		      dx = MX-FMX[i-1][n]+FMX[i-1][n+1];  /* Next pos */
		      dy = MY-FMY[i-1][n]+FMY[i-1][n+1];  /* next pos */
		      if (dx < BMinX) dx = BMinX;
		      else if (dx > MaxX) dx = BMaxX;
		      if (dy < BMinY) dy = BMinY;
		      else if (dy > MaxY) dy = BMaxY;
		      rx = abs(dx - FMX[i-1][n+1]);  /* Distance from 0pred */
		      ry = abs(dy - FMY[i-1][n+1]);
		    }
		  else if (MVTelescope)
		    {
		      dx = FMX[i-1][n+1];  /* Next pos */
		      dy = FMY[i-1][n+1];  /* next pos */
		    }
		  n++;
		}
	    }
	}
    }
  if (FrameInterval>1)
    {
      /* Do first backward predictive frame */
      MaxX = MaxY = 7;
      MinX = MinY = -8;
      BMaxX = BMaxY = 15;
      BMinX = BMinY = -16;
      printf("Doing Backward: %d\n",FrameInterval - 1);
      for(n=0,y=0;y<FFS[FrameInterval]->height;y+=16)
	{
	  for(rx=ry=dx=dy=x=0;x<FFS[FrameInterval]->width;x+=16)
	    {
	      CircleLimit = SearchLimit + MAX(rx,ry)+1;
	      HPFastBME(x,y,FFS[FrameInterval],
			x,y,FFS[FrameInterval-1],dx/2,dy/2);
	      BMX[FrameInterval-1][n] = MX;
	      BMY[FrameInterval-1][n] = MY;
	      if (MVPrediction)
		{
		  dx = MX; dy = MY;
		  if (dx < BMinX) dx = BMinX;
		  else if (dx > MaxX) dx = BMaxX;
		  if (dy < BMinY) dy = BMinY;
		  else if (dy > MaxY) dy = BMaxY;
		  rx = abs(dx); ry = abs(dy);
		}
	      n++;
	    }
	}
      for(i=FrameInterval-2;i>0;i--)
	{
	  MaxX = MaxY = ((FrameInterval-i)<<3)-1;
	  MinX = MinY = - ((FrameInterval-i)<<3);
	  BMaxX = BMaxY = ((FrameInterval-i)<<4)-1;
	  BMinX = BMinY = - ((FrameInterval-i)<<4);
	  printf("Doing Backward: %d\n",i);
	  for(n=0,y=0;y<FFS[FrameInterval]->height;y+=16)
	    {
	      for(rx=ry=dx=dy=x=0;x<FFS[FrameInterval]->width;x+=16)
		{
		  CircleLimit = SearchLimit + MAX(rx,ry)+1;
		  HPFastBME(x,y,FFS[FrameInterval],x,y,FFS[i],dx/2,dy/2);
		  BMX[i][n] = MX;
		  BMY[i][n] = MY;
		  if (MVPrediction)
		    {
		      dx = MX-BMX[i+1][n]+BMX[i+1][n+1];
		      dy = MY-BMY[i+1][n]+BMY[i+1][n+1];
		      if (dx < BMinX) dx = BMinX;
		      else if (dx > MaxX) dx = BMaxX;
		      if (dy < BMinY) dy = BMinY;
		      else if (dy > MaxY) dy = BMaxY;
		      rx = abs(dx - BMX[i+1][n+1]);
		      ry = abs(dy - BMY[i+1][n+1]);
		    }
		  else if (MVTelescope)
		    {
		      dx = BMX[i+1][n+1];
		      dy = BMY[i+1][n+1];
		    }
		  n++;
		}
	    }
	}
    }
}

/*END*/

