#ifndef NRRD_DEFINES_HAS_BEEN_INCLUDED
#define NRRD_DEFINES_HAS_BEEN_INCLUDED

#define NRRD_MAX_DIM 12         /* The maximum dimension which we can handle */

/* the C types which we will assume are present.  The #defines must give
   types which can be given literally as a variable type in a source file.
   Thus, if there is not a type available on the architecture which matches
   the description in enum nrrdType in nrrd.h, define it to something which
   does exist, and the library will detect the deficiency.  
   /usr/include/inttypes.h may be useful, if it exists.
   >>---> THIS MUST BE IN SYNC WITH THE ENUMS IN nrrd.h <---<<
*/
#define NRRD_CHAR signed char
#define NRRD_UCHAR unsigned char
#define NRRD_SHORT signed short int
#define NRRD_USHORT unsigned short int
#define NRRD_INT signed int
#define NRRD_UINT unsigned int
#define NRRD_LLONG signed long long int
#define NRRD_ULLONG unsigned long long int
#define NRRD_FLOAT float
#define NRRD_DOUBLE double
#define NRRD_LDOUBLE long double

#define NRRD_BIG_INT long long  /* biggest integral type allowed on system; 
				   used to hold number of items in array 
				   (MUST be a signed type, even though as a 
				   store for # of things, could be unsigned)
				   using "int" will give a maximum array size
				   of 1024x1024x1024 
				   (really 1024x1024x2048-1) */
#define NRRD_BIG_INT_PRINTF "%lld"

#define NRRD_SMALL_STRLEN  129  /* single words, types and labels */
#define NRRD_MED_STRLEN    257  /* phrases, single line of error message */
#define NRRD_BIG_STRLEN    513  /* lines, ascii header lines from file */
#define NRRD_HUGE_STRLEN  1025  /* sentences */
#define NRRD_ERR_STRLEN   2049  /* error message accumulation */

/* -------------------------------------------------------- */
/* ----------- END of user-alterable defines -------------- */
/* -------------------------------------------------------- */

#define NRRD_HEADER "NRRD00.01"
#define NRRD_COMMENT_CHAR '#'

#endif /* NRRD_DEFINES_HAS_BEEN_INCLUDED */
