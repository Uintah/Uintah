#ifndef SIRE_CONSTH
#define SIRE_CONSTH

/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
sire_const.h
----------------------------------------------------------------------------
PURPOSE:   This file specifies various constants used in sire
HISTORY:   9 Mar 97, J. Roberts, MIRL, Dept. Radiology, School of Med, U of U
              SLC, Utah, roberts@doug.med.utah.edu
---------------------------------------------------------------------------- */

#define SIRE_USERINPFILE "sire.inp"  /* Default user input file name */

#define SIRE_MAXCHARLEN 200      /* Max character array length */

/* Signa data file parameters */
#define SIRE_SIGNA_MAXIMGSIZE 512
#define RDB_HEADER_SIZE_BYTES 39940   /* Size of the raw PFile header */
#define IMG_HEADER_SIZE_BYTES 7904    /* Size of the image header */
#define SIRE_MAXRCVRMIN1 3            /* Default max number of receivers */
#define SIRE_MAXSLABMIN1 63           /* Default max number of slabs */
#define SIRE_SIGNAXIAL 2              /* Default flags for orientation */
#define SIRE_SIGNASAG 4
#define SIRE_SIGNACOR 8
#define SIRE_MAXPATNAMELEN 24         /* Max lengths of patient strings */
#define SIRE_MAXPATIDLEN 12
#define SIRE_SIGNASHORT 2             /* Size in bytes of Signa types */
#define SIRE_SIGNAINT 4

/* Acquisition parameters */
#define SIRE_FASTCARDSTR "fstcd"     /* String identifying fast card */
#define SIRE_3DFSEBBSTR "3dfse"      /* String to identify 3dfse black blood*/
#define SIRE_NORMALACQ   0
#define SIRE_FASTCARDACQ 1
#define SIRE_3DFSEBBACQ    2

#define SIRE_PI 3.141592654

#define SIRE_EXTRAPHASE  1   /* Number of extra samples in the phase direction
                                This number will be discarded when reading the
                                Signa raw k-space data files */

#define SIRE_MAXSLICEPERPFILE 512

#define SIRE_MAXFILTYPEDIM 2 /* Maximum dimension of filter type specification 
                                array */

/* Boolean flags */
#define SIRE_YES 1
#define SIRE_NO 0

/* Fourier transform parameters */
#define SIRE_XAXIS     1
#define SIRE_YAXIS     2
#define SIRE_ZAXIS     3
#define SIRE_XYAXIS    4
#define SIRE_XZAXIS    5
#define SIRE_YZAXIS    6
#define SIRE_XYZAXIS   7
#define SIRE_READAXIS  SIRE_XAXIS
#define SIRE_PHASEAXIS SIRE_YAXIS
#define SIRE_SLICEAXIS SIRE_ZAXIS
#define SIRE_READPHASEAXIS SIRE_XYAXIS
#define SIRE_READSLICEAXIS SIRE_XZAXIS
#define SIRE_PHASESLICEAXIS SIRE_YZAXIS
#define SIRE_ALLAXES SIRE_XYZAXIS
#define SIRE_FORWARD   -1
#define SIRE_BACKWARD -(SIRE_FORWARD)

/* Filter types */
#define SIRE_NOFILTER          0    /* No filter will be applied */
#define SIRE_USERFILTER        1    /* Read from a file */
#define SIRE_HANFILTER         2    /* Hanning filter */
#define SIRE_HAMFILTER         3    /* Hamming filter */
#define SIRE_HANALPHA          0.5  /* Alpha for hanning routine */
#define SIRE_HAMALPHA          0.54 /* Alpha for hanning routine */
#define SIRE_ASYMSTEPFILTER    4    /* Step filter for asymmetric data */
#define SIRE_ASYMSTEPFILTERW   0.05 /* Width of transition region in filter */
#define SIRE_ASYMRAMPFILTER    5    /* Ramp filter for asymmetric data */

/* Default user filter rootnames */
#define SIRE_USERFILTEREAD "sire_readfilter"
#define SIRE_USERFILTERPHASE "sire_phasefilter"
#define SIRE_USERFILTERSLICE "sire_slicefilter"

/* Reconstruction methods */
#define SIRE_DEFAULTRECON  0
#define SIRE_FILTERECON    1
#define SIRE_HOMODYNERECON 2
#define SIRE_CUPPENRECON   3

/* Controls for the 3Dfftdft function */
#define SIRE_RETFFTMAGPHA 0
#define SIRE_RETFFTCOMPLEX 1
#define SIRE_RETFFTMAGONLY 2
#define SIRE_RETFFTCOSSIN 3

/* Default rotation values from Signa raw file */
#define SIRE_SIGNA_0     0
#define SIRE_SIGNA_90    1
#define SIRE_SIGNA_180   2
#define SIRE_SIGNA_270   3

/* Coordinates for Signa images */
#define SIRE_NATIVE_COORD 0
#define SIRE_SIGNA_COORD 1

/* ZFI factor for recentering black blood profile */
#define SIRE_BBRCNTRZFI 16

/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif
