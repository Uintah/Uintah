#include "include/nrrd.h"

/* Ernesto "Che" Guevara  */

int numFields = 15;   /* number of field identifiers recognized */
Nrrd dummy;           /* for calculating offsets into struct */
char magicstr[NRRD_BIG_STRLEN];  /* hack: holder for magic string */

/* field identifier string */
char fieldStr[][NRRD_SMALL_STRLEN] = {  
  "content",
  "number",
  "type",
  "dimension",
  "encoding",
  "sizes",
  "spacings",
  "axis mins",
  "axis maxs",
  "labels",
  "min",
  "max",
  "blocksize",
  "data file",
  "data skip"
};

/* conversion character for info in the line */
char fieldConv[][NRRD_SMALL_STRLEN] = {  
  "%s",
  NRRD_BIG_INT_PRINTF,
  "%s",
  "%d",
  "%s",
  "%d",
  "%lf",
  "%lf",
  "%lf",
  "%s",
  "%lf",
  "%lf",
  "%d",
  "%s",
  "%d"
};

/* whether or not we parse as many items as the dimension */
int fieldMultiple[] = { 
  0,
  0,
  0,
  0,
  0,
  1,
  1,
  1,
  1,
  1,
  0,
  0,
  0,
  0,
  0
};

/*
******** nrrdOneLine()
** 
** gets one line from "file", putting it into an array if given size.
** "size" must be the size of line buffer "line".  Always null-terminates
** the contents of the array (except if the arguments are invalid).
**
** -1: if arguments are invalid
** 0: if saw EOF before seeing a newline
** 1: if line was a single newline
** n, where n <= size: if line was n-1 characters followed by newline
** size+1: if didn't see a newline within size-1 characters
**
** So except for returns of -1 and size+1, the return is the number of
** characters comprising the line, including the newline character.
*/
int
nrrdOneLine(FILE *file, char *line, int size) {
  int c, i;
  
  if (!(size >= 2 && line && file))
    return -1;
  line[0] = 0;
  for (i=0; (i <= size-2 && 
	     EOF != (c=getc(file)) && 
	     c != '\n'); ++i)
    line[i] = c;
  if (EOF == c)
    return 0;
  if (0 == i) {
    /* !EOF ==> we stopped because of newline */
    return 1;
  }
  if (i == size-1) {
    line[size-1] = 0;
    if ('\n' != c) {
      /* we got to the end of the array and still no newline */
      return size+1;
    }
    else {
      /* we got to end of array just as we saw a newline, all's well */
      return size;
    }
  }
  /* i < size-1 && EOF != c ==> '\n' == c */
  line[i] = 0;
  return i+1;
}

int
getnums(char *data, double *array, int num) {
  char err[NRRD_MED_STRLEN], me[] = "getnums";
  int i;

  if (1 != sscanf(data, "%lg", &(array[0]))) {
    sprintf(err, "%s: couldn't get first num of %s\n", me, data);
    nrrdSetErr(err); return 1;
  }
  i = 1;
  while (i < num) {
    if (!(data = strstr(data, " "))) {
      sprintf(err, "%s: didn't see space after number %d of %d\n", me, i, num);
      nrrdSetErr(err); return 1;
    }
    data = &(data[1]);
    if (1 != sscanf(data, "%lg", &(array[i]))) {
      sprintf(err, "%s: couldn't parse %s for num %d\n", me, data, i+1);
      nrrdSetErr(err); return 1;
    }
    i++;
  }
  return 0;
}

char *
getstring(char *data, char *str, int size) {
  char err[NRRD_MED_STRLEN], me[] = "getstring";
  int i;

  if (!(data && str && size >= 2)) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return NULL;
  }
  if (!('"' == data[0])) {
    sprintf(err, "%s: \"%s\" doesn't start with '\"'\n", me, data);
    nrrdSetErr(err); return NULL;
  }
  data += 1;
  for (i=0; (i<=size-2 && 
	     !('"' == data[i] && '\\' != data[i-1])); i++)
    str[i] = data[i];
  if (i == size-1) {
    sprintf(err, "%s: didn't see end of string \"%s\" soon enough", me, data);
    nrrdSetErr(err); return NULL;
  }
  str[i] = 0;
  return data+i;
}

int
parseline(Nrrd *nrrd, char *line) {
  int f, i, type, *ivals;
  char *next, *tmp, str[NRRD_MED_STRLEN], 
    err[NRRD_MED_STRLEN], me[] = "parseline";
  void *field[15];
  double *dvals, invals[NRRD_MAX_DIM];

  field[0] = &nrrd->content;
  field[1] = &nrrd->num;
  field[2] = &nrrd->type;      /* code below assumes field 2 is type */
  field[3] = &nrrd->dim;
  field[4] = &nrrd->encoding;  /* code below assumes field 4 is encoding */
  field[5] = &nrrd->size;
  field[6] = &nrrd->spacing;
  field[7] = &nrrd->axisMin;
  field[8] = &nrrd->axisMax;
  field[9] = &nrrd->label;
  field[10] = &nrrd->min;
  field[11] = &nrrd->max;
  field[12] = &nrrd->blockSize;
  field[13] = NULL;           /* there is no actual field in the nrrd for
				 the string holding the name of the datafile,
				 we're just using this element of the local
				 "field" array for this purpose */
  field[14] = &nrrd->dataSkip;

  for (f=0; f<=numFields-1; f++) {
    if (!(strncmp(line, fieldStr[f], strlen(fieldStr[f])))) {
      next = line + strlen(fieldStr[f]);
      if (!(':' == next[0] && ' ' == next[1])) {
	sprintf(err, "%s: didn't see \": \" after %s\n", me, fieldStr[f]);
	nrrdSetErr(err); return 1;
      }
      next += 2;
      break;
    }
  }
  if (NRRD_INSIDE(0, f, numFields-1)) {
    /*
    printf("parseline: saw field %d head, to parse \"%s\"\n", f, next);
    */
    if (!(fieldMultiple[f])) {
      if (!(strcmp(fieldConv[f], "%s"))) {
	if (4 == f) {
	  /* need to interpret encoding string */
	  for (i=0; i<=nrrdEncodingLast-1; i++) {
	    if (!(strcmp(next, nrrdEncoding2Str[i]))) {
	      nrrd->encoding = i;
	      break;
	    }
	  }
	  if (nrrdEncodingLast == i) {
	    sprintf(err, "%s: didn't recognize encoding \"%s\"\n", me, next);
	    nrrdSetErr(err); return 1;
	  }
	  /*
	  printf("got encoding %d\n", nrrd->encoding);
	  */
	}
	else if (2 == f) {
	  /* need to interpret type string */
	  if (nrrdTypeUnknown == (type = nrrdStr2Type(next))) {
	    sprintf(err, "%s: didn't recognize type \"%s\"\n", me, next);
	    nrrdSetErr(err); return 1;
	  }
	  nrrd->type = type;
	  /*
	  printf("got type %d\n", nrrd->type);
	  */
	}
	else if (13 == f) {
	  /* have to see if we can open the indicated data file */
	  if (!(nrrd->dataFile = fopen(next, "r"))) {
	    sprintf(err, "%s: can't open seperate data file \"%s\" for reading!\n",
		    me, next);
	    nrrdSetErr(err); return 1;
	  }
	}
	else 
	  strcpy(field[f], next);
      }
      else {
	if (1 != (sscanf(next, fieldConv[f], field[f]))) {
	  sprintf(err, "%s: couldn't sscanf %s \"%s\" as %s\n",
		  me, fieldStr[f], next, fieldConv[f]);
	  nrrdSetErr(err);
	  return 1;
	}
      }
    }
    else {
      /* we've got multiple items to parse */
      if (-1 == nrrd->dim) {
	sprintf(err, "%s: can't parse \"%s\" until dimension is known\n",
		me, fieldStr[f]);
	nrrdSetErr(err); return 1;
      }
      if (!(strcmp(fieldConv[f], "%s"))) {
	for (i=0; i<=nrrd->dim-1; i++) {
	  if (!(tmp = getstring(next, str, NRRD_MED_STRLEN))) {
	    sprintf(err, "%s: couldn't get 1st string from \"%s\"\n", 
		    me, next);
	    nrrdAddErr(err); return 1;
	  }
	  strcpy(nrrd->label[i], str);
	  /* value for "next" set on last iteration is never dereferenced */
	  next = tmp + 2;  
	}
      }
      else {
	if (getnums(next, invals, nrrd->dim)) {
	  sprintf(err, "%s: couldn't parse %d numbers in \"%s\"\n",
		  me, nrrd->dim, next);
	  nrrdSetErr(err); return 1;
	}
	for (i=0; i<=nrrd->dim-1; i++)
	ivals = field[f];
	dvals = field[f];
	for (i=0; i<=nrrd->dim-1; i++) {
	  if (!(strcmp(fieldConv[f], "%lf"))) {
	    dvals[i] = invals[i];
	    /*
	    printf("got %d: %lf\n", i, dvals[i]);
	    */
	  }
	  else {
	    ivals[i] = invals[i];
	    /*
	    printf("got %d: %d\n", i, ivals[i]);
	    */
	  }
	}
      }
    }
  }
  else {
    /* didn't recognize any fields at the beginning of the line */
    sprintf(err, "%s: no recognized field identifiers in \"%s\"\n", me, line);
    nrrdSetErr(err);
    return 1;
  }
  return 0;
}

int
nonCmtLine(FILE *file, char *line, int size, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nonCmtLine";
  int len;

  do {
    if (!(len = nrrdOneLine(file, line, size))) {
      sprintf(err, "%s: hit EOF\n", me); 
      nrrdSetErr(err); 
      return 0;
    }
    if (NRRD_COMMENT_CHAR == line[0]) {
      nrrdAddComment(nrrd, line+1);
    }
  } while (NRRD_COMMENT_CHAR == line[0]);
  return len;
}

Nrrd *
nrrdNewRead(FILE *file) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdNewRead";
  Nrrd *nrrd;
  
  if (!file) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return NULL;
  }
  if (!(nrrd = nrrdNew())) {
    sprintf(err, "%s: nrrdNew() failed\n", me);
    nrrdAddErr(err); return NULL;
  }
  if (nrrdRead(file, nrrd)) {
    sprintf(err, "%s: nrrdRead() failed\n", me);
    nrrdNuke(nrrd);
    nrrdAddErr(err); return NULL;
  }
  return nrrd;
}

int
nrrdRead(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdRead";

  if (nrrdReadHeader(file, nrrd)) {
    sprintf(err, "%s: nrrdReadHeader() failed\n", me);
    nrrdAddErr(err); return 1;
  }
  if (nrrdAlloc(nrrd, nrrd->num, nrrd->type, nrrd->dim)) {
    sprintf(err, "%s: nrrdAlloc() failed\n", me);
    nrrdAddErr(err); return 1;
  }
  if (nrrdReadData(file, nrrd)) {
    sprintf(err, "%s: nrrdReadData() failed\n", me);
    nrrdAddErr(err); return 1;
  }
  return 0;
}

nrrdMagic
nrrdReadMagic(FILE *file) {
  char line[NRRD_BIG_STRLEN], err[NRRD_MED_STRLEN], 
    me[] = "nrrdReadMagic";
  int i, len;

  magicstr[0] = 0;
  do {
    if (!(len = nrrdOneLine(file, line, NRRD_BIG_STRLEN))) {
      sprintf(err, "%s: initial nonCmtLine() hit EOF\n", me);
      nrrdAddErr(err); return nrrdMagicUnknown;
    }
  } while (1 == len);

  /* --- got to first non-trivial line */
  strcpy(magicstr, line);
  for (i=nrrdMagicUnknown+1; i<nrrdMagicLast; i++) {
    if (!strcmp(line, nrrdMagic2Str[i])) {
      break;
    }
  }
  if (i < nrrdMagicLast) {
    return (nrrdMagic)i;
  }
  else {
    return nrrdMagicUnknown;
  }
}

int
nrrdReadHeader(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdReadHeader";
  nrrdMagic magic;

  magic = nrrdReadMagic(file);
  switch (magic) {
  case nrrdMagicUnknown:
    sprintf(err, "%s: unknown magic \"%s\"\n", me, magicstr);
    nrrdAddErr(err);
    return 1;
  case nrrdMagicNrrd0001:
    if (nrrdReadNrrdHeader(file, nrrd)) {
      sprintf(err, "%s: nrrdReadNrrdHeader failed\n", me);
      nrrdAddErr(err); return 1;
    }
    break;
  case nrrdMagicP1:
  case nrrdMagicP2:
  case nrrdMagicP3:
  case nrrdMagicP4:
  case nrrdMagicP5:
  case nrrdMagicP6:
    if (nrrdReadPNMHeader(file, nrrd, magic)) {
      sprintf(err, "%s: nrrdReadPNMHeader failed\n", me);
      nrrdAddErr(err); return 1;
    }
    break;
  }
  return 0;
}  
  
int
nrrdReadNrrdHeader(FILE *file, Nrrd *nrrd) {
  char line[NRRD_BIG_STRLEN], err[NRRD_MED_STRLEN], 
    me[] = "nrrdReadNrrdHeader";
  int len;

  do {
    if (!(len = nonCmtLine(file, line, NRRD_BIG_STRLEN, nrrd))) {
      /* nonCmtLine returned 0, meaning it hit EOF.  This is an
	 error in most cases, but in the case that there's a seperate
	 data file (nrrd->dataFile != NULL), we gracefully permit
	 the situation and pretend that we just hit the single newline
	 signalling the end of the header */
      if (nrrd->dataFile) {
	len = 1;
      }
      else {
	sprintf(err, "%s: hit EOF parsing header\n", me);
	nrrdAddErr(err); return 1;
      }
    }
    if (len > 1) {
      if (parseline(nrrd, line)) {
	sprintf(err, "%s: parseline(\"%s\") failed\n", me, line);
	nrrdAddErr(err); return 1;
      }
    }
  } while (len > 1);

  /* we've hit a newline-only line; header is done.  Is nrrd finished? */
  if (nrrdCheck(nrrd)) {
    sprintf(err, "%s: nrrdCheck() failed\n", me);
    nrrdAddErr(err); return 1;
  }
  if (nrrdEncodingUnknown == nrrd->encoding) {
    sprintf(err, "%s: Encoding method has not been set.\n", me);
    nrrdSetErr(err); return 1;
  }
  /*
  printf("len = %d\n", len);
  */
  return 0;
}

int
nrrdReadData(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdReadData", line[NRRD_BIG_STRLEN];
  NrrdReadDataType fptr;
  FILE *dataFile;
  int ret, i;
  
  if (!(file && nrrd && nrrd->num > 0)){
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  if (!(NRRD_INSIDE(nrrdEncodingUnknown+1,
		    nrrd->encoding,
		    nrrdEncodingLast-1))) {
    sprintf(err, "%s: invalid encoding: %d\n", me, nrrd->encoding);
    nrrdSetErr(err); return 1;
  }
  if (!(fptr = nrrdReadDataFptr[nrrd->encoding])) {
    sprintf(err, "%s: NULL function pointer for %s reader\n", 
	    me, nrrdEncoding2Str[nrrd->encoding]);
    nrrdSetErr(err); return 1;
  }
  if (nrrd->dataFile) {
    /* we have a seperate data file, and we may need to skip
       some of the lines */
    dataFile = nrrd->dataFile;
    if (nrrd->dataSkip) {
      for (i=0; i<=nrrd->dataSkip-1; i++) {
	ret = nrrdOneLine(dataFile, line, NRRD_BIG_STRLEN);
	if (!(ret > 0 && ret <= NRRD_BIG_STRLEN)) {
	  sprintf(err, "%s: nrrdOneLine returned %d trying to skip line %d (of %d) in seperate data file\n",
		  me, ret, i+1, nrrd->dataSkip);
	  nrrdSetErr(err); return 1;
	}
      }
    }
  }
  else {
    /* there is no seperate data file, we're going to read from the
       present (given) file */
    dataFile = file;
  }
  printf("%s: calling reader for encoding %s (%d)\n",
	 me, nrrdEncoding2Str[nrrd->encoding], nrrd->encoding);
  if ((*fptr)(dataFile, nrrd)) {
    sprintf(err, "%s: data reader for %s encoding failed.\n", 
	    me, nrrdEncoding2Str[nrrd->encoding]);
    nrrdAddErr(err); return 1;
  }
  if (nrrd->dataFile) {
    /* if there was a seperate data file, close it now */
    fclose(dataFile);
  }
  printf("nrrdReadData: reader done\n");
  return 0;
}

int
nrrdReadDataRaw(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdReadDataRaw";
  int num, size;
  
  if (!(file && nrrd && nrrd->num > 0)){
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  num = nrrd->num;
  size = nrrdTypeSize[nrrd->type];
  if (nrrdAlloc(nrrd, nrrd->num, nrrd->type, nrrd->dim)) {
    /* admittedly weird to be calling this since we already have num, 
       type, dim set-- just a way to get error reporting on calloc */
    sprintf(err, "%s: nrrdAlloc() failed\n", me);
    nrrdAddErr(err); return 1;
  }
  if (num != fread(nrrd->data, size, num, file)) {
    sprintf(err, "%s: unable to read %d objects of size %d\n", me, num, size);
    nrrdSetErr(err); return 1;
  }
  return 0;
}

int
nrrdReadDataZlib(FILE *file, Nrrd *nrrd) {

  fprintf(stderr, "nrrdReadDataZlib: NOT IMPLEMENTED\n");
  return 0;
}

int
nrrdReadDataAscii(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdReadDataAscii", 
    numstr[NRRD_MED_STRLEN];
  NRRD_BIG_INT i;
  int type, tmp, size;
  char *data;
  unsigned char *udata;

  if (!(file && nrrd)) {
    sprintf(err, "%s: got NULL pointer\n", me);
    nrrdAddErr(err); return 1;
  }
  type = nrrd->type;
  data = nrrd->data;
  if (!(NRRD_INSIDE(nrrdTypeUnknown+1, type, nrrdTypeLast-1))) {
    sprintf(err, "%s: got bogus type %d\n", me, type);
    nrrdAddErr(err); return 1;
  }
  if (nrrdTypeBlock == type) {
    sprintf(err, "%s: can't read into blocks from ascii\n", me);
    nrrdAddErr(err); return 1;
  }
  size = nrrdTypeSize[type];
  for (i=0; i<=nrrd->num-1; i++) {
    if (1 != fscanf(file, "%s", numstr)) {
      sprintf(err, 
	      "%s: couldn't get element "NRRD_BIG_INT_PRINTF"\n", me, i);
      nrrdAddErr(err); return 1;
    }
    if (type > nrrdTypeShort) {
      /* sscanf supports putting value directly into this type */
      if (1 != sscanf(numstr, nrrdType2Conv[type], (void*)data)) {
	sprintf(err, 
		"%s: couldn't parse element "NRRD_BIG_INT_PRINTF"\n", me, i);
	nrrdAddErr(err); return 1;
      }
    }
    else {
      /* type is nrrdTypeChar or nrrdTypeUChar: sscanf into int first */
      if (1 != sscanf(numstr, nrrdType2Conv[type], &tmp)) {
	sprintf(err, 
		"%s: couldn't parse element "NRRD_BIG_INT_PRINTF"\n", me, i);
	nrrdAddErr(err); return 1;
      }
      if (nrrdTypeChar == type) {
	*data = tmp;
      }
      if (nrrdTypeUChar == type) {
	udata = (unsigned char *)data;
	*udata = tmp;
      }
    }
    data += size;
  }
  return 0;
}

int
nrrdReadDataHex(FILE *file, Nrrd *nrrd) {

  fprintf(stderr, "nrrdReadDataHex: NOT IMPLEMENTED\n");
  return 0;
}

int
nrrdReadDataBase85(FILE *file, Nrrd *nrrd) {

  fprintf(stderr, "nrrdReadDataBase85: NOT IMPLEMENTED\n");
  return 0;
}

int
nrrdWrite(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdWrite";

  if (!(file && nrrd && nrrd->data)){
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  if (nrrdWriteHeader(file, nrrd)) {
    sprintf(err, "%s: nrrdWriteHeader() failed\n", me);
    nrrdAddErr(err); return 1;
  }
  if (nrrdWriteData(file, nrrd)) {
    sprintf(err, "%s: nrrdWriteData() failed\n", me);
    nrrdAddErr(err); return 1;
  }
  return 0;
}

int
nrrdWriteHeader(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdWriteHeader", *cmt;
  int i, doit;

  if (!(file && nrrd)){
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  if (nrrdCheck(nrrd)) {
    sprintf(err, "%s: nrrdCheck() failed\n", me);
    nrrdAddErr(err); return 1;
  }
  if (!(NRRD_INSIDE(nrrdEncodingUnknown+1,
		    nrrd->encoding,
		    nrrdEncodingLast-1))) {
    sprintf(err, "%s: invalid encoding: %d\n", me, nrrd->encoding);
    nrrdSetErr(err); return 1;
  }
  fprintf(file, "%s\n", NRRD_HEADER);
  if (strlen(nrrd->content))
    fprintf(file, "content: %s\n", nrrd->content);
  fprintf(file, "number: " NRRD_BIG_INT_PRINTF "\n", nrrd->num);
  fprintf(file, "type: %s\n", nrrdType2Str[nrrd->type]);
  fprintf(file, "dimension: %d\n", nrrd->dim);
  fprintf(file, "encoding: %s\n", nrrdEncoding2Str[nrrd->encoding]);
  if (nrrdEncodingUser == nrrd->encoding)
    fprintf(file, "blocksize: %d\n", nrrd->blockSize);
  fprintf(file, "sizes:");
  for (i=0; i<=nrrd->dim-1; i++)
    fprintf(file, " %d", nrrd->size[i]);
  fprintf(file, "\n");
  doit = 0;
  for (i=0; i<=nrrd->dim-1; i++)
    if (NRRD_EXISTS(nrrd->spacing[i]))
      doit = 1;
  if (doit) {
    fprintf(file, "spacings:");
    for (i=0; i<=nrrd->dim-1; i++)
      fprintf(file, " %lf", nrrd->spacing[i]);
    fprintf(file, "\n");
  }
  doit = 0;
  for (i=0; i<=nrrd->dim-1; i++)
    if (NRRD_EXISTS(nrrd->axisMin[i]))
      doit = 1;
  if (doit) {
    fprintf(file, "axis mins:");
    for (i=0; i<=nrrd->dim-1; i++)
      fprintf(file, " %lf", nrrd->axisMin[i]);
    fprintf(file, "\n");
  }
  doit = 0;
  for (i=0; i<=nrrd->dim-1; i++)
    if (NRRD_EXISTS(nrrd->axisMax[i]))
      doit = 1;
  if (doit) {
    fprintf(file, "axis maxs:");
    for (i=0; i<=nrrd->dim-1; i++)
      fprintf(file, " %lf", nrrd->axisMax[i]);
    fprintf(file, "\n");
  }
  doit = 0;
  for (i=0; i<=nrrd->dim-1; i++)
    if (strlen(nrrd->label[i]))
      doit = 1;
  if (doit) {
    fprintf(file, "labels:");
    for (i=0; i<=nrrd->dim-1; i++)
      fprintf(file, " \"%s\"", nrrd->label[i]);
    fprintf(file, "\n");
  }
  if (nrrd->comment) {
    i = 0;
    while (cmt = nrrd->comment[i]) {
      fprintf(file, "#%s\n", cmt);
      i++;
    }
  }
  fprintf(file, "\n");
  return 0;
}

int
nrrdWriteData(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdWriteData";
  NrrdWriteDataType fptr;
  
  if (!(file && nrrd && nrrd->num > 0)){
    printf("%lu %lu %d\n", 
	   file, nrrd, (int)nrrd->num);
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  if (!(NRRD_INSIDE(nrrdEncodingUnknown+1,
		    nrrd->encoding,
		    nrrdEncodingLast-1))) {
    sprintf(err, "%s: invalid encoding: %d\n", me, nrrd->encoding);
    nrrdSetErr(err); return 1;
  }
  if (!(fptr = nrrdWriteDataFptr[nrrd->encoding])) {
    sprintf(err, "%s: NULL function pointer for %s writer\n", 
	    me, nrrdEncoding2Str[nrrd->encoding]);
    nrrdSetErr(err); return 1;
  }
  if ((*fptr)(file, nrrd)) {
    sprintf(err, "%s: data writer for %s encoding failed.\n", 
	    me, nrrdEncoding2Str[nrrd->encoding]);
    nrrdAddErr(err); return 1;
  }
  return 0;
}

int
nrrdWriteDataRaw(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdWriteDataRaw";
  int num, size;
  
  if (!(file && nrrd && nrrd->data && nrrd->num > 0)){
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  num = nrrd->num;
  size = nrrdTypeSize[nrrd->type];
  if (num != fwrite(nrrd->data, size, num, file)) {
    sprintf(err, 
	    "%s: unable to write %d objects of size %d (data,file=%lu,%lu)\n", 
	    me, num, size, (unsigned long)(nrrd->data), (unsigned long)(file));
    nrrdSetErr(err); 
    if (ferror(file)) {
      sprintf(err, "%s: ferror returned non-zero\n", me);
      nrrdAddErr(err);
    }
    return 1;
  }
  return 0;
}

int
nrrdWriteDataZlib(FILE *file, Nrrd *nrrd) {

  fprintf(stderr, "nrrdWriteDataZlib: NOT IMPLEMENTED\n");
  return 0;
}

#define DOIT(f, t) fprintf(file, f, *((t*)v))
int printC(FILE *file, void *v)    {return(DOIT("%d", char));}
int printUC(FILE *file, void *v)   {return(DOIT("%u", unsigned char));}
int printS(FILE *file, void *v)    {return(DOIT("%d", short));}
int printUS(FILE *file, void *v)   {return(DOIT("%u", unsigned short));}
int printI(FILE *file, void *v)    {return(DOIT("%d", int));}
int printUI(FILE *file, void *v)   {return(DOIT("%u", unsigned int));}
int printLLI(FILE *file, void *v)  {return(DOIT("%lld", long long));}
int printULLI(FILE *file, void *v) {return(DOIT("%llu", unsigned long long));}
int printF(FILE *file, void *v)    {return(DOIT("%f", float));}
int printD(FILE *file, void *v)    {return(DOIT("%f", double));}
int printLD(FILE *file, void *v)   {return(DOIT("%Lf", long double));}

int
nrrdWriteDataAscii(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdWriteDataAscii";
  NRRD_BIG_INT i;
  int type, size;
  char *data;
  int (*print[13])(FILE*, void*);
  
  if (!(file && nrrd)) {
    sprintf(err, "%s: got NULL pointer\n", me);
    nrrdAddErr(err); return 1;
  }
  type = nrrd->type;
  data = nrrd->data;
  print[nrrdTypeUnknown] = NULL;
  print[nrrdTypeChar] = printC;
  print[nrrdTypeUChar] = printUC;
  print[nrrdTypeShort] = printS;
  print[nrrdTypeUShort] = printUS;
  print[nrrdTypeInt] = printI;
  print[nrrdTypeUInt] = printUI;
  print[nrrdTypeLLong] = printLLI;
  print[nrrdTypeULLong] = printULLI;
  print[nrrdTypeFloat] = printF;
  print[nrrdTypeDouble] = printD;
  print[nrrdTypeLDouble] = printLD;
  if (!(NRRD_INSIDE(nrrdTypeUnknown+1, type, nrrdTypeLast-1))) {
    sprintf(err, "%s: got bogus type %d\n", me, type);
    nrrdAddErr(err); return 1;
  }
  if (nrrdTypeBlock == type) {
    sprintf(err, "%s: can't write blocks to ascii\n", me);
    nrrdAddErr(err); return 1;
  }
  size = nrrdTypeSize[type];
  for (i=0; i<=nrrd->num-1; i++) {
    print[type](file, data);
    fprintf(file, "\n");
    data += size;
  }  
  return 0;
}

int
nrrdWriteDataHex(FILE *file, Nrrd *nrrd) {

  fprintf(stderr, "nrrdWriteDataHex: NOT IMPLEMENTED\n");
  return 0;
}

int
nrrdWriteDataBase85(FILE *file, Nrrd *nrrd) {

  fprintf(stderr, "nrrdWriteDataBase85: NOT IMPLEMENTED\n");
  return 0;
}

int
nrrdReadPNMHeader(FILE *file, Nrrd *nrrd, nrrdMagic magic) {
  char err[NRRD_MED_STRLEN], line[NRRD_BIG_STRLEN], 
    me[] = "nrrdReadPNMHeader";
  int ascii, color, bitmap, 
    size, sx, sy, max, *num[5], want, got, dumb;

  if (!(file && nrrd)) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  size = NRRD_BIG_STRLEN;
  switch(magic) {
  case nrrdMagicP6:
    color = 1; ascii = 0; bitmap = 0;
    break;
  case nrrdMagicP5:
    color = 0; ascii = 0; bitmap = 0;
    break;
  case nrrdMagicP4:
    color = 0; ascii = 0; bitmap = 1;
    break;
  case nrrdMagicP3:
    color = 1; ascii = 1; bitmap = 0;
    break;
  case nrrdMagicP2:
    color = 0; ascii = 1; bitmap = 0;
    break;
  case nrrdMagicP1:
    color = 0; ascii = 1; bitmap = 1;
    break;
  default:
    sprintf(err, "%s: sorry, (%d) unsupported PNM format file\n", me, magic);
    nrrdSetErr(err); return 1;
  }
  sx = sy = max = 0;
  num[0] = &sx;
  num[1] = &sy;
  num[2] = &max;
  num[3] = num[4] = &dumb;
  got = 0;
  want = bitmap ? 2 : 3;
  while (got < want) {
    /* eventually, at worst, this will go to the end of the file */
    if (!(0 < nrrdOneLine(file, line, size))) {
      sprintf(err, "%s: line read failed\n", me);
      nrrdSetErr(err); return 1;
    }
    printf("%s: got line: |%s|\n", me, line);
    if ('#' == line[0]) {
      nrrdAddComment(nrrd, line);
      continue;
    }
    if (3 == sscanf(line, "%d%d%d", num[got], num[got+1], num[got+2])) {
      printf("%s: got 3\n", me);
      got += 3;
      continue;
    }
    if (2 == sscanf(line, "%d%d", num[got], num[got+1])) {
      printf("%s: got 2\n", me);
      got += 2;
      continue;
    }
    if (1 == sscanf(line, "%d", num[got])) {
      printf("%s: got 1\n", me);
      got += 1;
      continue;
    }
  }
  sx = NRRD_MAX(0, sx);
  sy = NRRD_MAX(0, sy);
  max = NRRD_MAX(0, max);
  printf("%s: image is %dx%d, maxval=%d\n", me, sx, sy, max);
  nrrd->num = (color ? 3 : 1)*sx*sy;
  /* we do not support binary bit arrays; a binary PGM will get
     put into an arrays of uchars */
  nrrd->type = ascii && max > 255 ? nrrdTypeUInt : nrrdTypeUChar;
  nrrd->dim = color ? 3 : 2;
  if (color) {
    nrrd->size[0] = 3;
    nrrd->size[1] = sx;
    nrrd->size[2] = sy;
  }
  else {
    nrrd->size[0] = sx;
    nrrd->size[1] = sy;
  }
  nrrd->encoding = (ascii 
		    ? nrrdEncodingAscii
		    : nrrdEncodingRaw);
  printf("%s: ascii = %d, encoding = %d\n", me, ascii, nrrd->encoding);
  return 0;
}

int
nrrdWritePNM(FILE *file, Nrrd *nrrd) {
  char err[NRRD_MED_STRLEN], me[] = "nrrdWritePNM", *cmt;
  int i, ascii, color;
  
  if (!(file && nrrd)) {
    sprintf(err, "%s: invalid args\n", me);
    nrrdSetErr(err); return 1;
  }
  /* HEY!! should probably allow something larger that 8-bit values
     if the user wants ASCII encoding; PNM does support that */
  color = (3 == nrrd->dim &&
	   3 == nrrd->size[0]);
  if (!(nrrdTypeUChar == nrrd->type &&
	(2 == nrrd->dim || color) )) {
    sprintf(err, "%s: not given proper 2-D array of unsigned chars\n", me);
    nrrdSetErr(err); return 1;
  }
  ascii = nrrdEncodingAscii == nrrd->encoding;
  if (!( ascii || nrrdEncodingRaw == nrrd->encoding )) {
    sprintf(err, "%s: PNM only supports %s or %s encoding\n", me,
	    nrrdEncoding2Str[nrrdEncodingRaw], 
	    nrrdEncoding2Str[nrrdEncodingAscii]);
    nrrdSetErr(err); return 1;
  }
  fprintf(file, "P%c\n", (color 
			  ? (ascii ? '3' : '6')
			  : (ascii ? '2' : '5')));
  if (nrrd->comment) {
    i = 0;
    while (cmt = nrrd->comment[i]) {
      fprintf(file, "#%s\n", cmt);
      i++;
    }
  }
  fprintf(file, "%d %d\n255\n", 
	  color ? nrrd->size[1] : nrrd->size[0], 
	  color ? nrrd->size[2] : nrrd->size[1]);
  if (nrrdWriteData(file, nrrd)) {
    sprintf(err, "%s: nrrdWriteData() failed\n", me);
    nrrdAddErr(err); return 1;
  }
  return 0;
}
