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
stat.c

This routine keeps all the statistics handy.

************************************************************
*/

/*LABEL stat.c */

#include <math.h>
#include "globals.h"

/*PUBLIC*/

extern void Statistics();
static void StatisticsMem();

/*PRIVATE*/

extern FRAME *CFrame;
extern FSTORE *CFStore;
extern STAT *CStat;

/*START*/

/*BFUNC 

Statistics() prints to {\tt stdout} all the accumulated statistics on
the memory structures (CFS and Iob).

EFUNC*/

void Statistics(RefFS,NewFS)
     FSTORE * RefFS;
     FSTORE * NewFS;
{
  BEGIN("Statistics");
  int i;

  for(i=0;i<CFrame->NumberComponents;i++)
    {
      StatisticsMem(RefFS->Iob[i]->mem,NewFS->Iob[i]->mem,CStat);
      /*printf("Comp: %d  MRSNR: %2.2f  SNR: %2.2f  PSNR: %2.2f  MSE: %4.2f  Entropy: %1.2f\n",
	     i,CStat->mrsnr,CStat->snr,CStat->psnr,CStat->mse,CStat->entropy);*/
    }
}

/*BFUNC

StatisticsMem() calculates the statistics beween a reference memory
structure and another memory structure, storing it in a statistics
structure.

EFUNC*/

static void StatisticsMem(mref,m,s)
     MEM *mref;
     MEM *m;
     STAT *s;
{
  BEGIN("StatisticsMem");
  int i,top;
  double value,squared,p,mr;
  double rvalue,rsquared;
  int Values[256],*iptr;
  unsigned char *cptr,*rptr;

  top = m->width*m->height;
  for(i=0,iptr=Values;i<256;i++) {*(iptr++)=0;}
  value=0;
  squared=0;
  rsquared=0;
  rvalue=0;
  for(i=0,rptr=mref->data,cptr=m->data;i<top;i++)
    {
      rvalue += (double) *rptr;
      value += (double) *cptr;
      squared += (double) (((*cptr)-*(rptr)) *((*cptr)-(*rptr)));
      rsquared += (double) ((*rptr) * (*rptr));

      Values[*cptr]++;
      cptr++;
      rptr++;
    }
  s->mean = value/(double) top;
  s->mse = squared/(double) top;
  if (squared)
    {
      if (rsquared) s->snr = 10*log10(rsquared/squared);
      else s->snr = -99.99;
      mr = (rsquared-(rvalue*rvalue/((double)top)));
      if (mr) s->mrsnr = 10*log10(mr/squared);
      else s->mrsnr = -99.99;
      if (top) s->psnr = 10*log10((65025.0 * (double) top)/squared);
      else s->psnr = -99.99;
    }
  else
    {
      s->snr = 99.99;
      s->mrsnr = 99.99;
      s->psnr = 99.99;
    }
  for(i=0,s->entropy=0,iptr=Values;i<256;i++)
    {
      if (*iptr)
	{
	  p = (double) *iptr/ (double) top;
	  s->entropy += p * log(p);
	}
      iptr++;
    }
  s->entropy = - s->entropy / log(2.0);
}

/*END*/
