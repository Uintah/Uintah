#ifndef NRRD_HAS_BEEN_INCLUDED
#define NRRD_HAS_BEEN_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nrrdDefines.h"
#include "nrrdMacros.h"   /* tested in macrotest.c */

/*
******** Nrrd struct
**
** The struct used to wrap around the raw data array
*/
typedef struct {
  /* 
  ** NECESSARY information describing the main array, how it is 
  ** represented and stored.  This is generally set at the same time 
  ** that either the nrrd is created, or at the time that an existing 
  ** nrrd is wrapped around an array 
  */
  void *data;                    /* the data in memory */
  NRRD_BIG_INT num;              /* number of elements */
  int type;                      /* a value from the nrrdType enum */
  int dim;                       /* what is dimension of data */
  int encoding;                  /* how the data will be read or should be
				    written from file */

  /* 
  ** Information about the individual axes of the data's domain 
  */
  int size[NRRD_MAX_DIM];        /* number of elements along each axis */
  double spacing[NRRD_MAX_DIM];  /* if non-NaN, distance between samples */
  double axisMin[NRRD_MAX_DIM];  /* if non-NaN, value associated with
				    lowest index */
  double axisMax[NRRD_MAX_DIM];  /* if non-NaN, value associated with
				    highest index */
                                 /* obviously one can set spacing to something 
				    incompatible with bot and top.  If this is
				    detected, bot and top will be used to 
				    set a new spacing value (blowing away its
				    old value) */
  char label[NRRD_MAX_DIM][NRRD_SMALL_STRLEN];  
                                 /* short info string for each axis */

  /* 
  ** Information of dubious standing- descriptive of whole array, but
  ** not necessary (meaningful only for some uses of a nrrd), but basic
  ** enough to be part of the basic nrrd type
  */
  char content[NRRD_MED_STRLEN]; /* briefly, just what the hell is this data */
  int blockSize;                 /* for nrrdTypeBlock array, block byte size */
  double min, max;               /* if non-NaN, extremal values for array */
  FILE *dataFile;                /* if non-NULL, where the data is to be 
				    read from.  If NULL, data will be read
				    from current file */
  int dataSkip;                  /* if dataFile non-NULL, the number of lines
				    in dataFile that should be skipped over
				    (so as to bypass another form of ASCII
				    header preceeding raw data) */

  /* 
  ** Comments.  Dynamically allocated.  Read from and written to header
  */
  int numComments;
  char **comment;
  
} Nrrd;

/*
******** typedefs for binary data reading and writing functions
*/
typedef int (*NrrdReadDataType)(FILE *file, Nrrd *nrrd);
typedef int (*NrrdWriteDataType)(FILE *file, Nrrd *nrrd);

/*
******** nrrdMagic enum
**
** the different "magic numbers" that nrrd knows about.  Not as useful
** as you might want- the readers (io.c) can (currently) only deal
** with a magic which is on a line on its own, with a carraige return.
** This is the case for the nrrd magic, and for the PNMs written by xv
** (HOWEVER: PNMs don't require a carriage return after the magic) 
*/
typedef enum {
  nrrdMagicUnknown,      /* 0: nobody knows! */
  nrrdMagicNrrd0001,     /* 1: currently, the only "native" nrrd header */
  nrrdMagicP1,           /* 2: ascii PBM */
  nrrdMagicP2,           /* 3: ascii PGM */
  nrrdMagicP3,           /* 4: ascii PPM */
  nrrdMagicP4,           /* 5: binary PBM */
  nrrdMagicP5,           /* 6: binary PGM */
  nrrdMagicP6,           /* 7: binary PPM */
  nrrdMagicLast          /* 8: after the last valid magic */
} nrrdMagic;

/*
******** nrrdMagic2Str[][]
**
** the actual strings for each magic
*/
extern char nrrdMagic2Str[][NRRD_SMALL_STRLEN];

/*
******** nrrdType enum
**
** all the different types, identified by integer
*/
typedef enum {
  nrrdTypeUnknown,       /*  0: nobody knows! */
  nrrdTypeChar,          /*  1:   signed 1-byte integral */
  nrrdTypeUChar,         /*  2: unsigned 1-byte integral */
  nrrdTypeShort,         /*  3:   signed 2-byte integral */
  nrrdTypeUShort,        /*  4: unsigned 2-byte integral */
  nrrdTypeInt,           /*  5:   signed 4-byte integral */
  nrrdTypeUInt,          /*  6: unsigned 4-byte integral */
  nrrdTypeLLong,         /*  7:   signed 8-byte integral */
  nrrdTypeULLong,        /*  8: unsigned 8-byte integral */
  nrrdTypeFloat,         /*  9:          4-byte floating point */
  nrrdTypeDouble,        /* 10:          8-byte floating point */
  nrrdTypeLDouble,       /* 11:         16-byte floating point */
  nrrdTypeBlock,         /* 12: size is user defined at run time */
  nrrdTypeLast           /* 13: after the last valid type */
} nrrdType;

/*
******** nrrdType2Str[][]
**
** strings for each type, except the non-type nrrdTypeLast.
** These are written in the NRRD header; it is only for the sake
** of clarity and simplicity that they also happen to be valid C types.
** There is more flexibility in interpreting the type read from the NRRD
** header, see nrrdStr2Type.
** (Besides the enum above, the actual C types used internally 
** are in nrrdDefines.h)
*/
extern char nrrdType2Str[][NRRD_SMALL_STRLEN];

/*
******** nrrdType2Conv[][]
**
** conversion sequence for each type, ie "%d" for nrrdTypeInt
*/
extern char nrrdType2Conv[][NRRD_SMALL_STRLEN];

/*
******** nrrdTypeSize[]
**
** expected sizes of the types, except for the non-type nrrdTypeLast
*/
extern int nrrdTypeSize[];

/*
******** nrrdEncoding enum
**
** how data might be encoded into a bytestream
*/
typedef enum {
  nrrdEncodingUnknown,       /* 0: nobody knows */
  nrrdEncodingRaw,           /* 1: file directly reflects memory */
  nrrdEncodingZlib,          /* 2: using the zlib compression library */
  nrrdEncodingAscii,         /* 3: decimal values are spelled out in ascii */
  nrrdEncodingHex,           /* 4: hexadecimal encoding */
  nrrdEncodingBase85,        /* 5: using base-85 (as in Postscript Level 2) */
  nrrdEncodingUser,          /* 6: something the user choses */
  nrrdEncodingLast           /* 7: after last valid one */
} nrrdEncoding;

/*
******** nrrdEncoding2Str[][]
**
** strings for each encoding type, except the non-type nrrdEncodeLast
*/
extern char nrrdEncoding2Str[][NRRD_SMALL_STRLEN];

extern nrrdReadDataRaw(FILE *, Nrrd *);
extern nrrdReadDataZlib(FILE *, Nrrd *);
extern nrrdReadDataAscii(FILE *, Nrrd *);
extern nrrdReadDataHex(FILE *, Nrrd *);
extern nrrdReadDataBase85(FILE *, Nrrd *);
extern nrrdWriteDataRaw(FILE *, Nrrd *);
extern nrrdWriteDataZlib(FILE *, Nrrd *);
extern nrrdWriteDataAscii(FILE *, Nrrd *);
extern nrrdWriteDataHex(FILE *, Nrrd *);
extern nrrdWriteDataBase85(FILE *, Nrrd *);

/*
******** nrrdReadDataFptr[]
**
** different functions for reading data into nrrd, indexed by a value from
** the nrrdEncode enum.  User can set the last one.
*/
extern NrrdReadDataType nrrdReadDataFptr[];

/*
******** nrrdDataWriteFptr[]
**
** different functions for writing data into nrrd, indexed by a value from
** the nrrdEncode enum.  User can set the last one.
*/
extern NrrdWriteDataType nrrdWriteDataFptr[];

/*
******** nrrdMeasr enum
**
** ways to "measure" some portion of the array
*/
typedef enum {
  nrrdMeasrUnknown,          /* 0: nobody knows */
  nrrdMeasrMin,              /* 1: smallest value */
  nrrdMeasrMax,              /* 2: biggest value */
  nrrdMeasrSum,              /* 3: sum of all values */
  nrrdMeasrMean,             /* 4: average of values */
  nrrdMeasrMedian,           /* 5: value at 50th percentile */
  nrrdMeasrMode,             /* 6: most common value */
  /* 
  ** these nrrdMeasrHisto* measures interpret the array as a histogram
  ** of some implied value distribution, and the measure uses the index
  ** space of the array for its range
  */
  nrrdMeasrHistoMin,         /* 7 */
  nrrdMeasrHistoMax,         /* 8 */
  nrrdMeasrHistoSum,         /* 9 */
  nrrdMeasrHistoMean,        /* 10 */
  nrrdMeasrHistoMedian,      /* 11 */
  nrrdMeasrHistoMode,        /* 12 */
  nrrdMeasrLast              /* 13: after last valid one */
} nrrdMeasr;

/******** type related */
/* (tested in typestest.c) */
extern int nrrdStr2Type(char *str);

/******** nan related silliness */
/* (tested in nantest.c) */
extern int nrrdIsNand(double);
extern int nrrdIsNanf(float);
extern double nrrdNand(void);
extern float nrrdNanf(void);

/******** error reporting */
/* (err.c) */
extern int nrrdSetErr(char *str);
extern void nrrdClearErr();
extern int nrrdAddErr(char *str);
extern int nrrdGetErr(char *str);
extern char *nrrdStrdupErr(void);

/******** making and destroying nrrds and basic info within */
/* (methods.c) */
extern Nrrd *nrrdNew(void);
extern void nrrdInit(Nrrd *nrrd);
extern Nrrd *nrrdNix(Nrrd *nrrd);
extern void nrrdEmpty(Nrrd *nrrd);
extern Nrrd *nrrdNuke(Nrrd *nrrd);
extern void nrrdWrap(Nrrd *nrrd, void *data, 
		     NRRD_BIG_INT num, int type, int dim);
extern Nrrd *nrrdNewWrap(void *data, NRRD_BIG_INT num, int type, int dim);
extern void nrrdSetInfo(Nrrd *nrrd, NRRD_BIG_INT num, int type, int dim);
extern Nrrd *nrrdNewSetInfo(NRRD_BIG_INT num, int type, int dim);
extern int nrrdAlloc(Nrrd *nrrd, NRRD_BIG_INT num, int type, int dim);
extern Nrrd *nrrdNewAlloc(NRRD_BIG_INT num, int type, int dim);
extern void nrrdAddComment(Nrrd *nrrd, char *cmt);
extern void nrrdClearComments(Nrrd *nrrd);
extern void nrrdDescribe(FILE *file, Nrrd *nrrd);
extern int nrrdCheck(Nrrd *nrrd);
extern void nrrdRange(Nrrd *nrrd);

/******** getting information to and from files */
/* io.c */
extern int nrrdRead(FILE *file, Nrrd *nrrd);
extern Nrrd *nrrdNewRead(FILE *file);
extern int nrrdReadHeader(FILE *file, Nrrd *nrrd); 
extern Nrrd *nrrdNewReadHeader(FILE *file); 
extern int nrrdReadData(FILE *file, Nrrd *nrrd);
extern int nrrdReadDataRaw(FILE *file, Nrrd *nrrd);
extern int nrrdReadDataZlib(FILE *file, Nrrd *nrrd);
extern int nrrdReadDataAscii(FILE *file, Nrrd *nrrd);
extern int nrrdReadDataHex(FILE *file, Nrrd *nrrd);
extern int nrrdReadDataBase85(FILE *file, Nrrd *nrrd);
extern int nrrdWrite(FILE *file, Nrrd *nrrd);
extern int nrrdWriteHeader(FILE *file, Nrrd *nrrd);
extern int nrrdWriteData(FILE *file, Nrrd *nrrd);
extern int nrrdWriteDataRaw(FILE *file, Nrrd *nrrd);
extern int nrrdWriteDataZlib(FILE *file, Nrrd *nrrd);
extern int nrrdWriteDataAscii(FILE *file, Nrrd *nrrd);
extern int nrrdWriteDataHex(FILE *file, Nrrd *nrrd);
extern int nrrdWriteDataBase85(FILE *file, Nrrd *nrrd);
extern int nrrdReadPNMHeader(FILE *file, Nrrd *nrrd, nrrdMagic magic);
extern int nrrdWritePNM(FILE *file, Nrrd *nrrd);

/******** slicing, cropping, sampling, and permuting */
/* subset.c */
extern int nrrdSlice(Nrrd *nrrdIn, Nrrd *nrrdOut, int axis, int pos);
extern Nrrd *nrrdNewSlice(Nrrd *nrrdIn, int axis, int pos);
extern int nrrdCrop(Nrrd *nrrdIn, Nrrd *nrrdOut, int *minIdx, int *maxIdx);
extern Nrrd *nrrdNewCrop(Nrrd *nrrdIn, int *minIdx, int *maxIdx);
extern int nrrdPermuteAxes(Nrrd *nrrdIn, Nrrd *nrrdOut, int *axes);
extern Nrrd *nrrdNewPermuteAxes(Nrrd *nrrdIn, int *axes);
extern int nrrdSample(Nrrd *nrrdIn, int *coord, void *val);
extern int (*nrrdIValue[13])(void *v);
extern float (*nrrdFValue[13])(void *v);
extern double (*nrrdDValue[13])(void *v);
extern int (*nrrdILookup[13])(void *v, int i);
extern float (*nrrdFLookup[13])(void *v, int i);
extern double (*nrrdDLookup[13])(void *v, int i);
extern void nrrdInitValue();

/********* HISTOGRAMS!!! */
/* histogram.c */
extern int nrrdHisto(Nrrd *nin, Nrrd *nout, int bins);
extern Nrrd *nrrdNewHisto(Nrrd *nin, int bins);
extern int nrrdDrawHisto(Nrrd *nin, Nrrd *nout, int sy);
extern Nrrd *nrrdNewDrawHisto(Nrrd *nin, int sy);

#endif /* NRRD_HAS_BEEN_INCLUDED */



