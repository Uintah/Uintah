#include "include/nrrd.h"

/*
** initnrrd
**
** initializes a nrrd to default state.  Mostly just sets values to 
** -1, Nan, or "", but if NrrdInfoFree is set, will free the info block,
** and if NrrdInfoNew is set, will create a new one
*/
void
initnrrd(Nrrd *nrrd) {
  int i;

  nrrd->data = NULL;
  nrrd->num = -1;
  nrrd->type = nrrdTypeUnknown;
  nrrd->dim = -1;
  nrrd->encoding = nrrdEncodingUnknown;
  for (i=0; i<=NRRD_MAX_DIM-1; i++) {
    nrrd->size[i] = -1;
    nrrd->spacing[i] = nrrdNand();
    nrrd->axisMin[i] = nrrd->axisMax[i] = nrrdNand();
    strcpy(nrrd->label[i], "");
  }
  strcpy(nrrd->content, "");
  nrrd->blockSize = -1;
  nrrd->min = nrrd->max = nrrdNand();
  nrrd->dataFile = NULL;
  nrrd->dataSkip = 0;    /* this is a reasonable default value */
  nrrdClearComments(nrrd);
}

/*
******** nrrdNew()
**
** creates and initializes a Nrrd
*/
Nrrd *
nrrdNew(void) {
  Nrrd *nrrd;

  if (!(nrrd = (Nrrd*)(calloc(1, sizeof(Nrrd))))) {
    nrrdSetErr("nrrdNew: calloc() for Nrrd failed!\n");
    return(NULL);
  }
  initnrrd(nrrd);
  return(nrrd);
}

/*
******** nrrdInit()
**
** puts nrrd back into state following initialization
*/
void
nrrdInit(Nrrd *nrrd) {

  if (nrrd) {
    initnrrd(nrrd);
  }
}

/*
******** nrrdNix()
**
** does nothing with the array, just does whatever is needed
** to free the nrrd itself
*/
Nrrd *
nrrdNix(Nrrd *nrrd) {
  
  if (nrrd) {
    initnrrd(nrrd);
    free(nrrd);
  }
  return(NULL);
}

/*
******** nrrdEmpty()
**
** frees data inside nrrd AND resets all its state, so its the
** same as what comes from nrrdNew()
*/
void 
nrrdEmpty(Nrrd *nrrd) {
  
  if (nrrd) {
    if (nrrd->data)
      free(nrrd->data);
    initnrrd(nrrd);
  }
}

/*
******** nrrdNuke()
**
** blows away the nrrd and everything inside
**
** always returns NULL
*/
Nrrd *
nrrdNuke(Nrrd *nrrd) {
  
  if (nrrd) {
    nrrdEmpty(nrrd);
    nrrdNix(nrrd);
  }
  return(NULL);
}

/*
******** nrrdWrap()
**
** wraps a given Nrrd around a given array
*/
void
nrrdWrap(Nrrd *nrrd, void *data, NRRD_BIG_INT num, int type, int dim) {

  if (nrrd) {
    nrrd->data = data;
    nrrd->num = num;
    nrrd->type = type;
    nrrd->dim = dim;
  }
}

/*
******** nrrdNewWrap()
**
** wraps a new Nrrd around a given array
*/
Nrrd *
nrrdNewWrap(void *data, NRRD_BIG_INT num, int type, int dim) {
  Nrrd *nrrd;

  if (!(nrrd = nrrdNew())) {
    nrrdAddErr("nrrdNewWrap: nrrdNew() failed\n");
    return(NULL);
  }
  nrrdWrap(nrrd, data, num, type, dim);
  return(nrrd);
}

/*
******** nrrdSetInfo()
**
** does anyone really need this?
*/
void
nrrdSetInfo(Nrrd *nrrd, NRRD_BIG_INT num, int type, int dim) {

  if (nrrd) {
    nrrd->num = num;
    nrrd->type = type;
    nrrd->dim = dim;
  }
}

/*
******** nrrdNewSetInfo()
**
** does anyone really need this?
*/
Nrrd *
nrrdNewSetInfo(NRRD_BIG_INT num, int type, int dim) {
  Nrrd *nrrd;

  if (!(nrrd = nrrdNew())) {
    nrrdAddErr("nrrdNewSetInfo: nrrdNew() failed\n");
    return(NULL);
  }
  nrrdSetInfo(nrrd, num, type, dim);
  return(nrrd);
}

/*
******** nrrdAlloc()
**
** allocates data array and sets information 
*/
int 
nrrdAlloc(Nrrd *nrrd, NRRD_BIG_INT num, int type, int dim) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdAlloc";

  if (!nrrd)
    return(1);
  if (!(nrrd->data = calloc(num, nrrdTypeSize[type]))) {
    sprintf(err, "%s: calloc(" NRRD_BIG_INT_PRINTF ",%d) failed\n", 
	    me, num, nrrdTypeSize[type]);
    nrrdSetErr(err);
    return(1);
  }
  nrrd->num = num;
  nrrd->type = type;
  nrrd->dim = dim;
  return(0);
}

/*
******** nrrdNewAlloc()
**
** creates the nrrd AND the array inside
*/
Nrrd *
nrrdNewAlloc(NRRD_BIG_INT num, int type, int dim) {
  Nrrd *nrrd;
  
  if (!(nrrd = nrrdNew())) {
    nrrdAddErr("nrrdNewAlloc: nrrdNew() failed\n");
    return(NULL);
  }
  if (nrrdAlloc(nrrd, num, type, dim)) {
    nrrdAddErr("nrrdNewAlloc: nrrdAlloc() failed\n");
    nrrdNix(nrrd);
    return(NULL);
  }    
  return(nrrd);
}

/*
******** nrrdAddComment()
**
** adds a given string to the list of comments
*/
void
nrrdAddComment(Nrrd *nrrd, char *cmt) {
  char *newcmt, **newcmts;
  int i, len, num;
  
  if (nrrd && cmt) {
    newcmt = strdup(cmt);
    len = strlen(newcmt);
    /* clean out carraige returns that would screw up reader */
    for (i=0; i<=len-1; i++) {
      if ('\n' == newcmt[i]) {
	if (i < len-1) {
	  newcmt[i] = ' ';
	}
	else {
	  newcmt[i] = 0;
	}
      }
    }
    num = nrrd->numComments+1;
    newcmts = malloc((num+1)*sizeof(char *));
    for (i=0; i<=nrrd->numComments-1; i++)
      newcmts[i] = nrrd->comment[i];
    newcmts[i] = newcmt;
    newcmts[i+1] = NULL;
    if (nrrd->comment) 
      free(nrrd->comment);
    nrrd->comment = newcmts;
    nrrd->numComments += 1;
  }
}

/*
******** nrrdClearComments()
**
** blows away comments only
*/
void
nrrdClearComments(Nrrd *nrrd) {
  char *cmt;
  int i;

  i = 0;
  if (nrrd->comment) {
    while (cmt = nrrd->comment[i]) {
      free(cmt);
      i++;
    }
    free(nrrd->comment);
  }
  nrrd->numComments = 0;
}

/*
******** nrrdDescribe
** 
** writes verbose description of nrrd to given file
*/
void
nrrdDescribe(FILE *file, Nrrd *nrrd) {
  int i;
  char *cmt;

  if (file && nrrd) {
    fprintf(file, "Data at 0x%lx is " NRRD_BIG_INT_PRINTF 
	    " elements of type %s.\n",
	    (unsigned long)nrrd->data, nrrd->num, nrrdType2Str[nrrd->type]);
    if (nrrdTypeBlock == nrrd->type) 
      fprintf(file, "The blocks have size %d\n", nrrd->blockSize);
    if (strlen(nrrd->content))
      fprintf(file, "It is \"%s\"\n", nrrd->content);
    fprintf(file, "It is a %d-dimensional array, with axes:\n", nrrd->dim);
    for (i=0; i<=nrrd->dim-1; i++) {
      if (strlen(nrrd->label[i]))
	fprintf(file, "%d: (\"%s\") ", i, nrrd->label[i]);
      else
	fprintf(file, "%d: ", i);
      fprintf(file, "size=%d, spacing=%lg, \n    axis(Min,Max) = (%lg,%lg)\n",
	      nrrd->size[i], nrrd->spacing[i], 
	      nrrd->axisMin[i], nrrd->axisMax[i]);
    }
    fprintf(file, "The min and max values are %lg, %lg\n", 
	    nrrd->min, nrrd->max);
    if (nrrd->comment) {
      fprintf(file, "Comments:\n");
      i = 0;
      while (cmt = nrrd->comment[i]) {
	fprintf(file, "%s\n", cmt);
	i++;
      }
    }
    fprintf(file, "\n");
  }
}

int
nrrdCheck(Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdCheck";
  NRRD_BIG_INT mult;
  int i;

  if (!(nrrd->num >= 1)) {
    sprintf(err, "%s: number of elements is %d\n", me, (int)nrrd->num);
    nrrdSetErr(err); return(1);
  }
  if (nrrdTypeUnknown == nrrd->type) {
    sprintf(err, "%s: type of array is unknown\n", me);
    nrrdSetErr(err); return(1);
  }
  if (nrrdTypeBlock == nrrd->type && -1 == nrrd->blockSize) {
    sprintf(err, "%s: type is \"block\" but no blocksize given\n", me);
    nrrdSetErr(err); return(1);
  }
  mult = 1;
  for (i=0; i<=nrrd->dim-1; i++) {
    if (-1 == nrrd->size[i])
      mult = -1;
    else
      mult *= nrrd->size[i];
  }
  if (mult != -1 && mult != nrrd->num) {
    sprintf(err, "%s: number of elements != product of axes sizes\n", me);
    nrrdSetErr(err); return(1);
  }
  return(0);
}

void
nrrdRange(Nrrd *nrrd) {
  double min, max, val;
  int type, size;
  NRRD_BIG_INT i, num;
  char *v;

  type = nrrd->type;
  if (nrrdTypeChar == type) {
    min = -128.0;
    max = 127.0;
  }
  else if (nrrdTypeUChar == type) {
    min = 0.0;
    max = 255.0;
  }
  else {
    nrrdInitValue();
    v = nrrd->data;
    num = nrrd->num;
    size = nrrdTypeSize[type];
    min = max = nrrdDValue[type](v);
    for (i=0; i<=num-1; i++) {
      val = nrrdDValue[type](v);
      min = NRRD_MIN(min, val);
      max = NRRD_MAX(max, val);
      v += size;
    }
  }
  nrrd->min = min;
  nrrd->max = max;
}  

