#include "nrrd.h"

double DvalueC(char *v)                      {return(*v);}
double DvalueUC(unsigned char *v)            {return(*v);}
double DvalueS(short *v)                     {return(*v);}
double DvalueUS(unsigned short *v)           {return(*v);}
double DvalueI(int *v)                       {return(*v);}
double DvalueUI(unsigned int *v)             {return(*v);}
double DvalueLLI(long long int *v)           {return(*v);}
double DvalueULLI(unsigned long long int *v) {return(*v);}
double DvalueF(float *v)                     {return(*v);}
double DvalueD(double *v)                    {return(*v);}
double DvalueLD(long double *v)              {return(*v);}
double (*nrrdDValue[13])(void *v);

double DlookupC(char *v, int i)                      {return(v[i]);}
double DlookupUC(unsigned char *v, int i)            {return(v[i]);}
double DlookupS(short *v, int i)                     {return(v[i]);}
double DlookupUS(unsigned short *v, int i)           {return(v[i]);}
double DlookupI(int *v, int i)                       {return(v[i]);}
double DlookupUI(unsigned int *v, int i)             {return(v[i]);}
double DlookupLLI(long long int *v, int i)           {return(v[i]);}
double DlookupULLI(unsigned long long int *v, int i) {return(v[i]);}
double DlookupF(float *v, int i)                     {return(v[i]);}
double DlookupD(double *v, int i)                    {return(v[i]);}
double DlookupLD(long double *v, int i)              {return(v[i]);}
double (*nrrdDLookup[13])(void *v, int i);

float FvalueC(char *v)                      {return(*v);}
float FvalueUC(unsigned char *v)            {return(*v);}
float FvalueS(short *v)                     {return(*v);}
float FvalueUS(unsigned short *v)           {return(*v);}
float FvalueI(int *v)                       {return(*v);}
float FvalueUI(unsigned int *v)             {return(*v);}
float FvalueLLI(long long int *v)           {return(*v);}
float FvalueULLI(unsigned long long int *v) {return(*v);}
float FvalueF(float *v)                     {return(*v);}
float FvalueD(double *v)                    {return(*v);}
float FvalueLD(long double *v)              {return(*v);}
float (*nrrdFValue[13])(void *v);

float FlookupC(char *v, int i)                      {return(v[i]);}
float FlookupUC(unsigned char *v, int i)            {return(v[i]);}
float FlookupS(short *v, int i)                     {return(v[i]);}
float FlookupUS(unsigned short *v, int i)           {return(v[i]);}
float FlookupI(int *v, int i)                       {return(v[i]);}
float FlookupUI(unsigned int *v, int i)             {return(v[i]);}
float FlookupLLI(long long int *v, int i)           {return(v[i]);}
float FlookupULLI(unsigned long long int *v, int i) {return(v[i]);}
float FlookupF(float *v, int i)                     {return(v[i]);}
float FlookupD(double *v, int i)                    {return(v[i]);}
float FlookupLD(long double *v, int i)              {return(v[i]);}
float (*nrrdFLookup[13])(void *v, int i);

int IvalueC(char *v)                      {return(*v);}
int IvalueUC(unsigned char *v)            {return(*v);}
int IvalueS(short *v)                     {return(*v);}
int IvalueUS(unsigned short *v)           {return(*v);}
int IvalueI(int *v)                       {return(*v);}
int IvalueUI(unsigned int *v)             {return(*v);}
int IvalueLLI(long long int *v)           {return(*v);}
int IvalueULLI(unsigned long long int *v) {return(*v);}
int IvalueF(float *v)                     {return(*v);}
int IvalueD(double *v)                    {return(*v);}
int IvalueLD(long double *v)              {return(*v);}
int (*nrrdIValue[13])(void *v);

int IlookupC(char *v, int i)                      {return(v[i]);}
int IlookupUC(unsigned char *v, int i)            {return(v[i]);}
int IlookupS(short *v, int i)                     {return(v[i]);}
int IlookupUS(unsigned short *v, int i)           {return(v[i]);}
int IlookupI(int *v, int i)                       {return(v[i]);}
int IlookupUI(unsigned int *v, int i)             {return(v[i]);}
int IlookupLLI(long long int *v, int i)           {return(v[i]);}
int IlookupULLI(unsigned long long int *v, int i) {return(v[i]);}
int IlookupF(float *v, int i)                     {return(v[i]);}
int IlookupD(double *v, int i)                    {return(v[i]);}
int IlookupLD(long double *v, int i)              {return(v[i]);}
int (*nrrdILookup[13])(void *v, int i);

void
nrrdInitValue() {

  printf("nrrdInitValue: hello\n");
  nrrdIValue[nrrdTypeChar] = (int (*)(void*))IvalueC;
  nrrdIValue[nrrdTypeUChar] = (int (*)(void*))IvalueUC;
  nrrdIValue[nrrdTypeShort] = (int (*)(void*))IvalueS;
  nrrdIValue[nrrdTypeUShort] = (int (*)(void*))IvalueUS;
  nrrdIValue[nrrdTypeInt] = (int (*)(void*))IvalueI;
  nrrdIValue[nrrdTypeUInt] = (int (*)(void*))IvalueUI;
  nrrdIValue[nrrdTypeLLong] = (int (*)(void*))IvalueLLI;
  nrrdIValue[nrrdTypeULLong] = (int (*)(void*))IvalueULLI;
  nrrdIValue[nrrdTypeFloat] = (int (*)(void*))IvalueF;
  nrrdIValue[nrrdTypeDouble] = (int (*)(void*))IvalueD;
  nrrdIValue[nrrdTypeLDouble] = (int (*)(void*))IvalueLD;

  nrrdFValue[nrrdTypeChar] = (float (*)(void*))FvalueC;
  nrrdFValue[nrrdTypeUChar] = (float (*)(void*))FvalueUC;
  nrrdFValue[nrrdTypeShort] = (float (*)(void*))FvalueS;
  nrrdFValue[nrrdTypeUShort] = (float (*)(void*))FvalueUS;
  nrrdFValue[nrrdTypeInt] = (float (*)(void*))FvalueI;
  nrrdFValue[nrrdTypeUInt] = (float (*)(void*))FvalueUI;
  nrrdFValue[nrrdTypeLLong] = (float (*)(void*))FvalueLLI;
  nrrdFValue[nrrdTypeULLong] = (float (*)(void*))FvalueULLI;
  nrrdFValue[nrrdTypeFloat] = (float (*)(void*))FvalueF;
  nrrdFValue[nrrdTypeDouble] = (float (*)(void*))FvalueD;
  nrrdFValue[nrrdTypeLDouble] = (float (*)(void*))FvalueLD;

  nrrdDValue[nrrdTypeChar] = (double (*)(void*))DvalueC;
  nrrdDValue[nrrdTypeUChar] = (double (*)(void*))DvalueUC;
  nrrdDValue[nrrdTypeShort] = (double (*)(void*))DvalueS;
  nrrdDValue[nrrdTypeUShort] = (double (*)(void*))DvalueUS;
  nrrdDValue[nrrdTypeInt] = (double (*)(void*))DvalueI;
  nrrdDValue[nrrdTypeUInt] = (double (*)(void*))DvalueUI;
  nrrdDValue[nrrdTypeLLong] = (double (*)(void*))DvalueLLI;
  nrrdDValue[nrrdTypeULLong] = (double (*)(void*))DvalueULLI;
  nrrdDValue[nrrdTypeFloat] = (double (*)(void*))DvalueF;
  nrrdDValue[nrrdTypeDouble] = (double (*)(void*))DvalueD;
  nrrdDValue[nrrdTypeLDouble] = (double (*)(void*))DvalueLD;

  nrrdILookup[nrrdTypeChar] = (int (*)(void*, int))IlookupC;
  nrrdILookup[nrrdTypeUChar] = (int (*)(void*, int))IlookupUC;
  nrrdILookup[nrrdTypeShort] = (int (*)(void*, int))IlookupS;
  nrrdILookup[nrrdTypeUShort] = (int (*)(void*, int))IlookupUS;
  nrrdILookup[nrrdTypeInt] = (int (*)(void*, int))IlookupI;
  nrrdILookup[nrrdTypeUInt] = (int (*)(void*, int))IlookupUI;
  nrrdILookup[nrrdTypeLLong] = (int (*)(void*, int))IlookupLLI;
  nrrdILookup[nrrdTypeULLong] = (int (*)(void*, int))IlookupULLI;
  nrrdILookup[nrrdTypeFloat] = (int (*)(void*, int))IlookupF;
  nrrdILookup[nrrdTypeDouble] = (int (*)(void*, int))IlookupD;
  nrrdILookup[nrrdTypeLDouble] = (int (*)(void*, int))IlookupLD;

  nrrdFLookup[nrrdTypeChar] = (float (*)(void*, int))FlookupC;
  nrrdFLookup[nrrdTypeUChar] = (float (*)(void*, int))FlookupUC;
  nrrdFLookup[nrrdTypeShort] = (float (*)(void*, int))FlookupS;
  nrrdFLookup[nrrdTypeUShort] = (float (*)(void*, int))FlookupUS;
  nrrdFLookup[nrrdTypeInt] = (float (*)(void*, int))FlookupI;
  nrrdFLookup[nrrdTypeUInt] = (float (*)(void*, int))FlookupUI;
  nrrdFLookup[nrrdTypeLLong] = (float (*)(void*, int))FlookupLLI;
  nrrdFLookup[nrrdTypeULLong] = (float (*)(void*, int))FlookupULLI;
  nrrdFLookup[nrrdTypeFloat] = (float (*)(void*, int))FlookupF;
  nrrdFLookup[nrrdTypeDouble] = (float (*)(void*, int))FlookupD;
  nrrdFLookup[nrrdTypeLDouble] = (float (*)(void*, int))FlookupLD;

  nrrdDLookup[nrrdTypeChar] = (double (*)(void*, int))DlookupC;
  nrrdDLookup[nrrdTypeUChar] = (double (*)(void*, int))DlookupUC;
  nrrdDLookup[nrrdTypeShort] = (double (*)(void*, int))DlookupS;
  nrrdDLookup[nrrdTypeUShort] = (double (*)(void*, int))DlookupUS;
  nrrdDLookup[nrrdTypeInt] = (double (*)(void*, int))DlookupI;
  nrrdDLookup[nrrdTypeUInt] = (double (*)(void*, int))DlookupUI;
  nrrdDLookup[nrrdTypeLLong] = (double (*)(void*, int))DlookupLLI;
  nrrdDLookup[nrrdTypeULLong] = (double (*)(void*, int))DlookupULLI;
  nrrdDLookup[nrrdTypeFloat] = (double (*)(void*, int))DlookupF;
  nrrdDLookup[nrrdTypeDouble] = (double (*)(void*, int))DlookupD;
  nrrdDLookup[nrrdTypeLDouble] = (double (*)(void*, int))DlookupLD;
  printf("nrrdInitValue: nrrdFValue = %lu, nrrdFValue[2] = %lu\n", 
	 (unsigned long)nrrdFValue, (unsigned long)(nrrdFValue[2]));
  printf("nrrdInitValue: bye bye\n");
}

int
elementSize(Nrrd *nrrd) {

  if (!(nrrd->type > nrrdTypeUnknown &&
	nrrd->type < nrrdTypeLast)) {
    return -1;
  }
  if (nrrdTypeBlock != nrrd->type) {
    return nrrdTypeSize[nrrd->type];
  }
  else {
    return nrrd->blockSize;
  }
}

void
select(void *dataIn, void *dataOut, int size,
       NRRD_BIG_INT *idx, NRRD_BIG_INT num) {
  NRRD_BIG_INT i, o;

  o = 0;
  for (i=0; i<=num-1; i++) {
    memcpy((char*)dataOut + (o++)*size, 
	   (char*)dataIn + idx[i]*size, size);
  }
}

/*
******** nrrdSample()
**
** given coordinates within a nrrd, copies the 
** single element into given *val
*/
int
nrrdSample(Nrrd *nrrd, int *coord, void *val) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdSample";
  int size, dim, i, idx;
  
  if (!(nrrd && coord && val && 
	nrrd->type > nrrdTypeUnknown &&
	nrrd->type < nrrdTypeLast)) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  size = elementSize(nrrd);
  dim = nrrd->dim;
  for (i=0; i<=dim-1; i++) {
    if (!(NRRD_INSIDE(0, coord[i], nrrd->size[i]-1))) {
      sprintf(err, "%s: coordinate %d on axis %d out of bounds (0 to %d)\n", 
	      me, coord[i], i, nrrd->size[i]-1);
      nrrdSetErr(err); return 1;
    }
  }
  idx = coord[dim-1];
  for (i=dim-2; i>=0; i--) {
    idx = coord[i] + nrrd->size[i]*idx;
  }
  memcpy((char*)val, (char*)(nrrd->data) + idx*size, size);
  return 0;
}

/*
******** nrrdSlice()
**
** slices a nrrd along a given axis, at a given position.
**
** will allocate memory for the new slice only if NULL==nrrdOut->data,
** otherwise assumes that the pointer there is pointing to something
** big enough to hold the slice
*/
int
nrrdSlice(Nrrd *nrrdIn, Nrrd *nrrdOut, int axis, int pos) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdSlice";
  NRRD_BIG_INT num,          /* number of elements in slice */
    i, j,                    /* index through elements in slice */
    *idx;                    /* indices into original array of the 
				elements which comprise the slice */
  int tmp,                   /* divided and mod'ed to produce coords */
    d,                       /* running index along dimensions */
    dimIn,                   /* dimension of original array  */
    dimOut,                  /* dimension of slice array  */
    elSize,                  /* size of one element */
    map[NRRD_MAX_DIM],       /* map from dimension index in slice to 
				dimension index in original array */
    *sizeIn,                 /* dimensions of original array */
    *coord;                  /* array of original array coordinates 
				comprising slice */
  void *dataIn, *dataOut;

  if (!(nrrdIn && nrrdOut)) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  if (!(NRRD_INSIDE(0, axis, nrrdIn->dim-1))) {
    sprintf(err, "%s: slice axis %d out of bounds (0 to %d)\n", 
	    me, axis, nrrdIn->dim-1);
    nrrdSetErr(err); return 1;
  }
  if (!(NRRD_INSIDE(0, pos, nrrdIn->size[axis]-1) )) {
    sprintf(err, "%s: position %d out of bounds (0 to %d)\n", 
	    me, axis, nrrdIn->size[axis]-1);
    nrrdSetErr(err); return 1;
  }
  dimIn = nrrdIn->dim;
  dimOut = dimIn - 1;
  sizeIn = nrrdIn->size;
  dataIn = nrrdIn->data;
  /* 
  ** (put here optimized functions for slicing of 1-D, 2-D and 3-D array)
  */
  for (d=0; d<=dimOut-1; d++) {
    map[d] = d < axis ? d : d+1;
  }

  /* # elements in slice is product of dimensions along non-slice axes */
  num = 1;
  for (d=0; d<=dimOut-1; d++)
    num *= sizeIn[map[d]];
  coord = malloc(num*dimIn*sizeof(int));
  idx = malloc(num*sizeof(NRRD_BIG_INT));
  if (!(coord && idx)) {
    sprintf(err, "%s: couldn't alloc coord and idx arrays\n", me);
    nrrdSetErr(err); return 1;
  }
  if (!(nrrdOut->data)) {
    if (nrrdAlloc(nrrdOut, num, nrrdIn->type, dimOut)) {
      sprintf(err, "%s: nrrdAlloc() failed to create slice\n", me);
      nrrdSetErr(err); return 1;
    }
  }
  dataOut = nrrdOut->data;

  /* the coordinate in the slice axis is fixed */
  for (i=0; i<=num-1; i++)
    coord[dimIn*i + axis] = pos;

  /* produce array of coordinates inside original array of the
  ** elements that comprise the slice.  We go linearly through
  ** the indices of the _slice_, and then div and mod this to
  ** produce the necessary coordinates 
  */
  for (i=0; i<=num-1; i++) {
    tmp = i;
    for (d=0; d<=dimOut-1; d++) {
      coord[map[d] + dimIn*i] = tmp % sizeIn[map[d]];
      tmp /= sizeIn[map[d]];
    }
  }

  /* map coordinates to indices by using something reminiscent
  ** of Horner's rule, starting with the last coordinate, and
  ** working to the first one, adding the current coordinate,
  ** and the product of the current size with the last tmp value (j)
  **
  ** detecting big sections of contiguous numbers should be done
  ** here at some point, to facilitate better use of memcpy
  */
  for (i=0; i<=num-1; i++) {
    j = coord[(dimIn-1) + dimIn*i]; /* last coordinate for sample # i */
    for (d=dimIn-2; d>=0; d--)
      j = coord[d + dimIn*i] + sizeIn[d]*j;
    idx[i] = j;
    /* (testing index generation)
    printf("%s; i=" NRRD_BIG_INT_PRINTF " -->", me, i);
    for (d=0; d<=dimIn-1; d++)
      printf(" %d:%d", d, coord[dimIn*i + d]);
    printf(" --> " NRRD_BIG_INT_PRINTF "\n", j);
    */
  }

  /* set information in slice */
  elSize = elementSize(nrrdIn);
  select(dataIn, dataOut, elSize, idx, num);
  for (d=0; d<=dimOut-1; d++) {
    nrrdOut->size[d] = nrrdIn->size[map[d]];
    nrrdOut->spacing[d] = nrrdIn->spacing[map[d]];
    nrrdOut->axisMin[d] = nrrdIn->axisMin[map[d]];
    nrrdOut->axisMax[d] = nrrdIn->axisMax[map[d]];
    strcpy(nrrdOut->label[d], nrrdIn->label[map[d]]);
  }
  sprintf(nrrdOut->content, "slice(%s,%d,%d)", 
	  nrrdIn->content, axis, pos);
  nrrdOut->blockSize = nrrdIn->blockSize;
  nrrdIn->min = nrrdNand();
  nrrdIn->max = nrrdNand();

  /* bye */
  free(coord);
  free(idx);
  return 0;
}

/*
******** nrrdNewSlice()
**
** slicer which calls nrrdNew first
*/
Nrrd *
nrrdNewSlice(Nrrd *nrrdIn, int axis, int pos) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdNewSlice";
  Nrrd *nrrdOut;

  if (!(nrrdOut = nrrdNew())) {
    sprintf(err, "%s: nrrdNew() failed\n", me);
    nrrdAddErr(err);
    return NULL;
  }
  if (nrrdSlice(nrrdIn, nrrdOut, axis, pos)) {
    sprintf(err, "%s: nrrdSlice() failed\n", me);
    nrrdAddErr(err);
    nrrdNuke(nrrdOut);
    return NULL;
  }
  return nrrdOut;
}


/*
******** nrrdPermuteAxes
**
*/
int
nrrdPermuteAxes(Nrrd *nin, Nrrd *nout, int *axes) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdPermuteAxes", tmpstr[512];
  NRRD_BIG_INT num,          /* number of elements in volume */
    i, j,                    /* index through elements in slice */
    *idx;                    /* indices into original array of the 
				elements which comprise the slice */
  void *dataIn, *dataOut;
  int used[NRRD_MAX_DIM],    /* records times a given axis is listed
				in permuted order (should only be once) */
    dim,                     /* dimension of volume */
    *size,                   /* sizes along axes of original volume */
    tmp,                     /* divided and mod'ed to produce coords */
    d,                       /* running index along dimensions */
    elSize,                  /* size of one element */
    *coord;                  /* array of original array coordinates 
				comprising slice */

  if (!(nin && nout && axes)) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  num = nin->num;
  dim = nin->dim;
  size = nin->size;
  dataIn = nin->data;
  memset(used, 0, NRRD_MAX_DIM*sizeof(int));
  for (d=0; d<=dim-1; d++) {
    if (!NRRD_INSIDE(0, axes[d], dim-1)) {
      sprintf(err, "%s: axis#%d == %d out of bounds\n", me, d, axes[d]);
      nrrdSetErr(err); return 1;
    }
    used[axes[d]] += 1;
  }
  for (d=0; d<=dim-1; d++) {
    if (1 != used[d]) {
      sprintf(err, "%s: axis %d used %d times, not once\n", me, d, used[d]);
      nrrdSetErr(err); return 1;
    }
  }

  coord = malloc(num*dim*sizeof(int));
  idx = malloc(num*sizeof(NRRD_BIG_INT));
  if (!(coord && idx)) {
    sprintf(err, "%s: couldn't alloc coord and idx arrays\n", me);
    nrrdSetErr(err); return 1;
  }
  if (!(nout->data)) {
    if (nrrdAlloc(nout, num, nin->type, dim)) {
      sprintf(err, "%s: nrrdAlloc() failed to create slice\n", me);
      nrrdSetErr(err); return 1;
    }
  }
  dataOut = nout->data;

  /* produce array of coordinates inside original array of the
  ** elements that comprise the slice.  We go linearly through the
  ** indices of the permuted volume, and then div and mod this to
  ** produce the necessary coordinates
  */
  for (i=0; i<=num-1; i++) {
    tmp = i;
    for (d=0; d<=dim-1; d++) {
      coord[axes[d] + dim*i] = tmp % size[axes[d]];
      tmp /= size[axes[d]];
    }
  }

  /* map coordinates to indices by using something reminiscent
  ** of Horner's rule, starting with the last coordinate, and
  ** working to the first one, adding the current coordinate,
  ** and the product of the current size with the last tmp value (j)
  **
  ** detecting big sections of contiguous numbers should be done
  ** here at some point, to facilitate better use of memcpy
  */
  for (i=0; i<=num-1; i++) {
    j = coord[(dim-1) + dim*i]; /* last coordinate for sample # i */
    for (d=dim-2; d>=0; d--)
      j = coord[d + dim*i] + size[d]*j;
    idx[i] = j;
  }

  /* set information in slice */
  elSize = elementSize(nin);
  select(dataIn, dataOut, elSize, idx, num);
  for (d=0; d<=dim-1; d++) {
    nout->size[d] = nin->size[axes[d]];
    nout->spacing[d] = nin->spacing[axes[d]];
    nout->axisMin[d] = nin->axisMin[axes[d]];
    nout->axisMax[d] = nin->axisMax[axes[d]];
    strcpy(nout->label[d], nin->label[axes[d]]);
  }
  sprintf(nout->content, "permute(%s,", nin->content);
  for (d=0; d<=dim-1; d++) {
    sprintf(tmpstr, "%d%c", axes[d], d == dim-1 ? ')' : ',');
    strcat(nout->content, tmpstr);
  }
  nout->blockSize = nin->blockSize;
  nin->min = nrrdNand();
  nin->max = nrrdNand();

  /* bye */
  free(coord);
  free(idx);
  return 0;
}

/*
******** nrrdNewPermuteAxes
**
*/
Nrrd *
nrrdNewPermuteAxes(Nrrd *nin, int *axes) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdNewPermuteAxes";
  Nrrd *nout;

  if (!(nout = nrrdNew())) {
    sprintf(err, "%s: nrrdNew() failed\n", me);
    nrrdAddErr(err);
    return NULL;
  }
  if (nrrdPermuteAxes(nin, nout, axes)) {
    sprintf(err, "%s: nrrdPermuteAxes() failed\n", me);
    nrrdAddErr(err);
    nrrdNuke(nout);
    return NULL;
  }
  return nout;
}


/*
******** nrrdCrop
**
*/
int
nrrdCrop(Nrrd *nin, Nrrd *nout, int *min, int *max) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdCrop", tmpstr[512];
  NRRD_BIG_INT num,          /* number of elements in volume */
    i, j,                    /* index through elements in slice */
    *idx;                    /* indices into original array of the 
				elements which comprise the slice */
  void *dataIn, *dataOut;
  int len[NRRD_MAX_DIM],
    dim,                     /* dimension of volume */
    tmp,                     /* divided and mod'ed to produce coords */
    d,                       /* running index along dimensions */
    elSize,                  /* size of one element */
    *coord;                  /* array of original array coordinates */

  if (!(nin && nout && min && max)) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  dim = nin->dim;
  dataIn = nin->data;
  num = 1;
  for (d=0; d<=dim-1; d++) {
    if (!NRRD_INSIDE(0, min[d], nin->size[d])) {
      sprintf(err, "%s: min#%d == %d out of bounds\n", me, d, min[d]);
      nrrdSetErr(err); return 1;
    }
    if (!NRRD_INSIDE(0, max[d], nin->size[d])) {
      sprintf(err, "%s: max#%d == %d out of bounds\n", me, d, max[d]);
      nrrdSetErr(err); return 1;
    }
    len[d] = NRRD_ABS(max[d] - min[d]) + 1;
    num *= len[d];
  }
  coord = malloc(num*dim*sizeof(int));
  idx = malloc(num*sizeof(NRRD_BIG_INT));
  if (!(coord && idx)) {
    sprintf(err, "%s: couldn't alloc coord and idx arrays\n", me);
    nrrdSetErr(err); return 1;
  }
  if (!(nout->data)) {
    if (nrrdAlloc(nout, num, nin->type, dim)) {
      sprintf(err, "%s: nrrdAlloc() failed to create slice\n", me);
      nrrdSetErr(err); return 1;
    }
  }
  dataOut = nout->data;

  /* produce array of coordinates inside original array of the
  ** elements that comprise the slice.  We go linearly through the
  ** indices of the cropped volume, and then div and mod this to
  ** produce the necessary coordinates
  */
  for (i=0; i<=num-1; i++) {
    tmp = i;
    for (d=0; d<=dim-1; d++) {
      coord[d + dim*i] = tmp % len[d];
      if (max[d] < min[d]) {
	/* flip along this axis */
	coord[d + dim*i] = len[d] - 1 - coord[d + dim*i];
      }
      coord[d + dim*i] += NRRD_MIN(max[d], min[d]);
      tmp /= len[d];
    }
  }

  /* map coordinates to indices by using something reminiscent
  ** of Horner's rule, starting with the last coordinate, and
  ** working to the first one, adding the current coordinate,
  ** and the product of the current size with the last tmp value (j)
  **
  ** detecting big sections of contiguous numbers should be done
  ** here at some point, to facilitate better use of memcpy
  */
  for (i=0; i<=num-1; i++) {
    j = coord[(dim-1) + dim*i]; /* last coordinate for sample # i */
    for (d=dim-2; d>=0; d--)
      j = coord[d + dim*i] + nin->size[d]*j;
    idx[i] = j;
  }

  /* set information in slice */
  elSize = elementSize(nin);
  select(dataIn, dataOut, elSize, idx, num);
  for (d=0; d<=dim-1; d++) {
    nout->size[d] = len[d];
    nout->spacing[d] = nin->spacing[d];
    nout->axisMin[d] = NRRD_AFFINE(0, min[d], len[d]-1,
				   nin->axisMin[d], nin->axisMax[d]);
    nout->axisMax[d] = NRRD_AFFINE(0, max[d], len[d]-1,
				   nin->axisMin[d], nin->axisMax[d]);
    strcpy(nout->label[d], nin->label[d]);
  }
  sprintf(nout->content, "crop(%s,", nin->content);
  for (d=0; d<=dim-1; d++) {
    sprintf(tmpstr, "%d-%d%c", min[d], max[d], d == dim-1 ? ')' : ',');
    strcat(nout->content, tmpstr);
  }
  nout->blockSize = nin->blockSize;
  nin->min = nrrdNand();
  nin->max = nrrdNand();

  /* bye */
  free(coord);
  free(idx);
  return 0;
}

/*
******** nrrdNewPermuteAxes
**
*/
Nrrd *
nrrdNewCrop(Nrrd *nin, int *min, int *max) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdNewCrop";
  Nrrd *nout;

  if (!(nout = nrrdNew())) {
    sprintf(err, "%s: nrrdNew() failed\n", me);
    nrrdAddErr(err);
    return NULL;
  }
  if (nrrdCrop(nin, nout, min, max)) {
    sprintf(err, "%s: nrrdCrop() failed\n", me);
    nrrdAddErr(err);
    nrrdNuke(nout);
    return NULL;
  }
  return nout;
}
