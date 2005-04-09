/*
 *  SiRe.h: The SiRe classes - derived from VoidStar
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_Modules_MRA_SiRe_h
#define SCI_Modules_MRA_SiRe_h 1

#include <Datatypes/VoidStar.h>

#include <Modules/MRA/fftw.h>
#include <Modules/MRA/sire_const.h>
#include <Modules/MRA/sire_struct.h>
#include <Modules/MRA/sire_version.h>

// SiRe class definition

typedef struct _SiReDataS {
   /* User inputs */
   char                   FirstPFile[SIRE_MAXCHARLEN];
   SIRE_FLAGS             Flag;
   SIRE_DIRINFO           Step;
   SIRE_FILTERINFO        FilterInfo[SIRE_MAXFILTYPEDIM];
   SIRE_IMGINFO           SireImgInfo;
   SIRE_SIGNA_IMGINFO     SignaImgInfo;

   /* Time variables */
   time_t                 TimeStamp;
   double                 RunTime;
   char                   TimeStr[SIRE_MAXCHARLEN];

    /* Store these for each piece of the reconstruction */
   int 			  FirstSlab;
   int			  LastSlab;
   int 			  IRcvr;
   int			  ISlab;   

   /* Nonuser inputs */
   int                    NRead, NPhase, NRawRead, NRawPhase, NRawSlice;
   int                    NSlabPerPFile, NPFile, NSlBlank, Rcvr0, RcvrN;
   int                    NFinalImgSlice, PointSize;

   /* Nonuser control parameters */
   char                   **PFiles, ***RcvrSlabImgFiles, 
                             **Rcvr3DImgFiles;
   char                   *FinalImgFile;
   int                    *SlabIndices, NSlicePerRcvr, NRcvrPerSlab;
   int                    NSlab, NRcnRead, NRcnPhase, NRcnSlicePerRcvr;
   int                    NRcnOverlap, NRcnFinalImgSlice;

   /* Reconstruction arrays */
   SIRE_COMPLEX           *RcvrRaw, *ShiftRcvrRaw;
   short                  *ZFIRcvrImg, **RcvrDCOffInd, 
                          *OverlapImg;
   float                  ***Filter;
   int NPasses;
   int PassIdx;
   int ShrinkFactor;
} SiReDataS;

void Pio(Piostream&, SiReDataS&);

class SiReData : public VoidStar {
public:
    SiReDataS s;
    Semaphore lockstepSem;
public:
    SiReData();
    SiReData(const SiReData& copy);
    virtual ~SiReData();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


#endif
