#include "include/nrrd.h"
#include <math.h>

int
nrrdHisto(Nrrd *nin, Nrrd *nout, int bins) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdHisto";
  int type, idx, *hist, size;
  NRRD_BIG_INT i, num;
  float min, max, val;
  char *data;

  if (!(nin && nout && bins > 0)) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  if (!(nout->data)) {
    if (nrrdAlloc(nout, bins, nrrdTypeUInt, 1)) {
      sprintf(err, "%s: nrrdAlloc() failed to allocate histogram array\n", me);
      nrrdSetErr(err); return 1;
    }
  }
  nout->size[0] = bins;
  hist = nout->data;
  data = nin->data;
  type = nin->type;
  size = nrrdTypeSize[nin->type];
  num = nin->num;

  nrrdInitValue();
  /* find first non-NaN value */
  for (i=0; i<=num-1; i++) {
    val = nrrdFValue[type](data + i*size);
    if (NRRD_EXISTS(val)) {
      min = max = val;
      break;
    }
  }
  if (!(NRRD_EXISTS(min))) {
    /* whole volume was NaN, nothing to do */
    return 0;
  }

  /* find min and max */
  for (i=0; i<=num-1; i++) {
    val = nrrdFValue[type](data + i*size);
    if (NRRD_EXISTS(val)) {
      min = NRRD_MIN(min, val);
      max = NRRD_MAX(max, val);
    }
  }
  nout->axisMin[0] = min;
  nout->axisMax[0] = max;

  /* make histogram */
  for (i=0; i<=nin->num-1; i++) {
    val = nrrdFValue[type](data + i*size);
    if (NRRD_EXISTS(val)) {
      NRRD_INDEX(min, val, max, bins, idx);
      ++hist[idx];
    }
  }

  nrrdDescribe(stdout, nout);
  return 0;
}

Nrrd *
nrrdNewHisto(Nrrd *nin, int bins) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdNewHisto";
  Nrrd *nout;

  if (!(nout = nrrdNew())) {
    sprintf(err, "%s: nrrdNew() failed\n", me);
    nrrdAddErr(err);
    return NULL;
  }
  if (nrrdHisto(nin, nout, bins)) {
    sprintf(err, "%s: nrrdHisto() failed\n", me);
    nrrdAddErr(err);
    nrrdNuke(nout);
    return NULL;
  }
  nrrdDescribe(stdout, nout);
  return nout;
}

int
nrrdDrawHisto(Nrrd *nin, Nrrd *nout, int sy) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdDrawHisto", cmt[NRRD_MED_STRLEN];
  int *hist, k, sx, x, y, maxhits, maxhitidx,
    numticks, *Y, *logY, tick, *ticks;
  unsigned char *idata;

  if (!(nin && nout && sy > 0)) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  if (!(1 == nin->dim && nrrdTypeUInt == nin->type)) {
    sprintf(err, "%s: given nrrd can't be a histogram\n", me);
    nrrdSetErr(err); return 1;
  }
  if (!(nout->data)) {
    if (nrrdAlloc(nout, nin->size[0]*sy, nrrdTypeUChar, 2)) {
      sprintf(err, "%s: nrrdAlloc() failed to allocate histogram image\n", me);
      nrrdSetErr(err); return 1;
    }
  }
  idata = nout->data;
  nout->size[0] = sx = nin->size[0];
  nout->size[1] = sy;
  hist = nin->data;
  maxhits = maxhitidx = 0;
  for (x=0; x<=sx-1; x++) {
    if (maxhits < hist[x]) {
      maxhits = hist[x];
      maxhitidx = x;
    }
  }
  numticks = log10(maxhits + 1);
  ticks = (int*)calloc(numticks, sizeof(int));
  Y = (int*)calloc(sx, sizeof(int));
  logY = (int*)calloc(sx, sizeof(int));
  for (k=0; k<=numticks-1; k++) {
    NRRD_INDEX(0, log10(pow(10,k+1) + 1), log10(maxhits+1), sy, ticks[k]);
  }
  for (x=0; x<=sx-1; x++) {
    NRRD_INDEX(0, hist[x], maxhits, sy, Y[x]);
    NRRD_INDEX(0, log10(hist[x]+1), log10(maxhits+1), sy, logY[x]);
    printf("%d -> %d,%d\n", x, Y[x], logY[x]);
  }
  for (y=0; y<=sy-1; y++) {
    tick = 0;
    for (k=0; k<=numticks-1; k++)
      tick |= ticks[k] == y;
    for (x=0; x<=sx-1; x++) {
      idata[x + sx*(sy-1-y)] = 
	(y >= logY[x]       /* above log curve                       */
	 ? (!tick ? 0       /*                    not on tick mark   */
	    : 255)          /*                    on tick mark       */
	 : (y >= Y[x]       /* below log curve, above normal curve   */
	    ? (!tick ? 128  /*                    not on tick mark   */
	       : 0)         /*                    on tick mark       */
	    :255            /* below log curve, below normal curve */
	    )
	 );
    }
  }
  sprintf(cmt, "min value: %g\n", nin->axisMin[0]);
  nrrdAddComment(nout, cmt);
  sprintf(cmt, "max value: %g\n", nin->axisMax[0]);
  nrrdAddComment(nout, cmt);
  sprintf(cmt, "max hits: %d, around value %g\n", maxhits, 
	  NRRD_AFFINE(0, maxhitidx, sx-1, nin->axisMin[0], nin->axisMax[0]));
  nrrdAddComment(nout, cmt);
  free(Y);
  free(logY);
  free(ticks);
  return 0;
}

Nrrd *
nrrdNewDrawHisto(Nrrd *nin, int sy) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdNewDrawHisto";
  Nrrd *nout;

  if (!(nout = nrrdNew())) {
    sprintf(err, "%s: nrrdNew() failed\n", me);
    nrrdAddErr(err);
    return NULL;
  }
  if (nrrdDrawHisto(nin, nout, sy)) {
    sprintf(err, "%s: nrrdDrawHisto() failed\n", me);
    nrrdAddErr(err);
    nrrdNuke(nout);
    return NULL;
  }
  return nout;
}
