#ifndef SIRE_STRUCTH
#define SIRE_STRUCTH
/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ 
sire_struct.h
---------------------------------------------------------------------------- 
PURPOSE:   This file specifies the non-Signa structures used in sire
HISTORY:   9 Mar 97, J. Roberts, MIRL, Dept. Radiology, School of Med, U of U
              SLC, Utah, roberts@doug.med.utah.edu
---------------------------------------------------------------------------- */
#include "fftw.h"

#define SIRE_COMPLEX FFTW_COMPLEX

typedef struct
{
   int Read;            /* Control parameter for read direction */
   int Phase;           /* Control parameter for phase direction */
   int Slice;           /* Control parameter for slice direction */
} SIRE_DIRINFO;

typedef struct 
{
   int ReconType;       /* Index specifying the type of recon to perform */
   int RDBRawCollect;   /* Flag from the signa raw file indicating whether the
                           data was reconstructed (at least in the slice 
                           direction) or not */
   int AcqType;         /* Flag indicating the type of acquisition.  So far
                           can be SIRE_NORMALACQ, SIRE_FASTCARDACQ, or 
                           SIRE_3DFSEACQ */
   int HotsaData;       /* Special flag for HOTSA data.  */
   int NCuppenIter;     /* If cuppen-pocs, then how many iterations */
   int AutoDetectAsym;  /* Allows users to turn off the autodetection of 
                           asymmetric echo (and hence turn off auto-Homodyne 
                           reconstruction in such cases */
   int AutoScalePhase;  /* User specified phase step (0) or calculated (1) */
   int AutoScaleSlice;  /* User specified slice step (0) or calculated (1) */
   SIRE_DIRINFO AsymmetricEcho;  /* Whether asymmetric(1) or not (0) in 
                                    respective directions */
   int NkzAcquired;     /* Number of kz acquired in cases of asymmetric kz
                           3dfse bb data sets */
   SIRE_DIRINFO Trunc;  /* Truncation flags controlling asymmetric data; a 
                           value of 1 truncates the data based on the 
                           position of the echo center; 0 produces no 
                           truncation */
   int FindEveryCntr;   /* Flag for forcing the center of k-space to be 
                           determined for every receiver (1) or one time 
                           only (0) during execution.  Default (0) to save
                           time */
   int ReconRcvrSlabs;  /* Recon receiver slabs (1) or not (0) */
   int InitReconkz;     /* Turn on/off the initial recon in the z direction */
   int FlattenBB;       /* Flatten the black blood profile (1) or not (0) */
   int SaveScale;       /* Save the scaling factor data */
   int SaveFilter;      /* Save the filter data */
   int StartRcvr;       /* 0-based index of first receiver to be recon. */
   int StartSlab;       /* 0-based index of first slab of receiver to recon */
   int EndRcvr;         /* 0-based index of last receiver in recon */
   int EndSlab;         /* 0-based index of last slab of last receiver */
   int OverlapRcvrSlabs;/* Overlap slabs (1) for each receiver or not (0) */
   int InPlaceOverlap;  /* Overlap by using the disk (0) or memory (1) */
   int AutoCor;         /* Perform (1) autocorrelation of overlapping slabs 
                           or not (0) */
   int SaveRcvrSlabs;   /* Save (1) or delete intermediate receiver slabs */
   int MakeZFImg;       /* Overlap (1) receiver data to form 3d ZFI or no (0)*/
   int SaveRcvrs;       /* Save (1) or delete separate receiver 3D data sets */
   int MakeNonZFImg;    /* Create (1) or not (0) non-ZFI image data file */
   int MakeSignaImg;    /* Create (1) or not (0) Signa format image files */
   int SaveZFImg;       /* Save (1) or delete final 3D ZFI image data file */
} SIRE_FLAGS;

typedef struct
{
   float RawImgDC;      /* Sets the level below which all data will be set 
                           to zero (ie. dc is subtracted) prior to scaling
                           to MaxImgValue.  Permitted values from 0 to 1.0 */
   int MaxImgValue;     /* Output maximum image data value */
   int RealPartOut;     /* Save real (1) or imaginary (0) image data */
   int UseUserScale;    /* Flag indicating whether user entered a scale */
   float UserScale;     /* User entered scaling factor */
   
} SIRE_IMGINFO;

typedef struct
{
   SIRE_DIRINFO Type;   /* Filter type structure */
   SIRE_DIRINFO Width;  /* Filter width data */
} SIRE_FILTERINFO;

typedef struct
{
   SIRE_DIRINFO ImgAxis;/* Axis assignments data space -> image space */
   int Rotation;        /* Flag from the Signa P file indicating the rotation
                           necessary to line the reconstructed image in the
                           "nose-up" direction */
   int Transpose;       /* Flag from the P file indicating any transposition
                           after rotation */
   int CoordType;       /* Flag specifying native (0) or Signa (1) coordinate
                           specification of window corners and dimensions */
   int ImgRead0;        /* Image window corner in read direction */
   int ImgPhase0;       /* Image window corner in phase direction */
   int NImgRead;        /* Dimension of output image in read direction */
   int NImgPhase;       /* Dimension of output image in phase direction */
   int HdrExamNum;      /* Exam number as extracted from the raw data header */
   int ForceNewExam;    /* Flag allowing user to force the new exam number */
   int NewExamNum;      /* Exam number assigned to new images */
   int NewSeriesNum;    /* Series number assigned to new images */
   float XPixelSize;    /* Output x pixel dimension */
   float YPixelSize;    /* Output y pixel dimension */
   float SliceThick;    /* Acquired slice thickness of data */
   float ScanSpacing;   /* Spacing wrt slice thickness, negative for overlap */
   
} SIRE_SIGNA_IMGINFO;

/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif
