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

#ifndef SCI_Packages_DaveW_Datatypes_SiRe_h
#define SCI_Packages_DaveW_Datatypes_SiRe_h 1

#include <Core/Datatypes/VoidStar.h>
#include <Packages/DaveW/ThirdParty/SiRe/include/fftw.h>
#include <Packages/DaveW/ThirdParty/SiRe/include/sire_const.h>
#include <Packages/DaveW/ThirdParty/SiRe/include/sire_struct.h>
#include <Packages/DaveW/ThirdParty/SiRe/include/sire_version.h>
#include <Core/Thread/Semaphore.h>

// SiRe class definition

namespace DaveW {
using namespace SCIRun;

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

} // End namespace DaveW
void Pio(Piostream&, SiReDataS&);


#endif
