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

#ifndef SCI_DaveW_Datatypes_SiRe_h
#define SCI_DaveW_Datatypes_SiRe_h 1

#include <SCICore/Datatypes/VoidStar.h>
#include <SCICore/Multitask/ITC.h>
#include <DaveW/ThirdParty/SiRe/include/fftw.h>
#include <DaveW/ThirdParty/SiRe/include/sire_const.h>
#include <DaveW/ThirdParty/SiRe/include/sire_struct.h>
#include <DaveW/ThirdParty/SiRe/include/sire_version.h>

// SiRe class definition

namespace DaveW {
namespace Datatypes {

using SCICore::Datatypes::VoidStar;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

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
    SCICore::Multitask::Semaphore lockstepSem;
public:
    SiReData();
    SiReData(const SiReData& copy);
    virtual ~SiReData();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

void Pio(Piostream&, SiReDataS&);
} // End namespace Datatypes
} // End namespace DaveW


//
// $Log$
// Revision 1.2  1999/08/25 03:47:35  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/23 02:53:02  dmw
// Dave's Datatypes
//
// Revision 1.2  1999/05/03 04:52:06  dmw
// Added and updated DaveW Datatypes/Modules
//
//
#endif
