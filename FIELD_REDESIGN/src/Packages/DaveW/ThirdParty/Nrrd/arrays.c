#include "include/nrrd.h"

/* for all the arrays that everyone needs.  See nrrd.h for documentation */

char 
nrrdMagic2Str[][NRRD_SMALL_STRLEN] = {
  "(unknown magic)",
  NRRD_HEADER,
  "P1",
  "P2",
  "P3",
  "P4",
  "P5",
  "P6"
};

char 
nrrdType2Str[][NRRD_SMALL_STRLEN] = {
  "(unknown type)",
  "char",
  "unsigned char",
  "short",
  "unsigned short",
  "int",
  "unsigned int",
  "long long",
  "unsigned long long",
  "float",
  "double",
  "long double",
  "block"
};

char
nrrdType2Conv[][NRRD_SMALL_STRLEN] = {
  "%*d",  /* what else? sscanf: skip, printf: use "minimum precision" */
  "%d",
  "%u",
  "%hd",
  "%hu",
  "%d",
  "%u",
  "%lld",
  "%llu",
  "%f",
  "%lf",
  "%Lf",
  "%*d"  /* what else? */
};

int 
nrrdTypeSize[] = {
  -1, /* unknown */
  1,  /* char */
  1,  /* unsigned char */
  2,  /* short */
  2,  /* unsigned short */
  4,  /* int */
  4,  /* unsigned int */
  8,  /* long long */
  8,  /* unsigned long long */
  4,  /* float */
  8,  /* double */
  16, /* long double */
  -1  /* effectively unknown; user has to set explicitly */
};

char 
nrrdEncoding2Str[][NRRD_SMALL_STRLEN] = {
  "(unknown encoding)",
  "raw",
  "zlib",
  "ascii",
  "hex",
  "base85",
  "user"
  "",
};

NrrdReadDataType 
nrrdReadDataFptr[] = {
  NULL,
  nrrdReadDataRaw,
  nrrdReadDataZlib,
  nrrdReadDataAscii,
  nrrdReadDataHex,
  nrrdReadDataBase85,
  NULL
};

NrrdWriteDataType 
nrrdWriteDataFptr[] = {
  NULL,
  nrrdWriteDataRaw,
  nrrdWriteDataZlib,
  nrrdWriteDataAscii,
  nrrdWriteDataHex,
  nrrdWriteDataBase85,
  NULL
};

