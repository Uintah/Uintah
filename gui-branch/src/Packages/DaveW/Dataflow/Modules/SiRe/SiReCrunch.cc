
/*
 *  SiReCrunch.cc:  Crunch the SiRe data -- from k-space to image space
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/SiRe/SiRe.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/VoidStarPort.h>
#include <Core/Datatypes/VoidStar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Tester/RigorousTest.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

#define DEBUG 0

/* Prototypes */
extern "C" {
void sire_3Dfftdft(int, int,int,SIRE_COMPLEX *, int, int, float **, int);
void sire_calcfilter(int,int,int,int,SIRE_FILTERINFO *,SIRE_DIRINFO,
                     float ****);
void sire_calcfilterw(int, int, int, SIRE_FLAGS, SIRE_DIRINFO, SIRE_DIRINFO *,
                      SIRE_FILTERINFO *);
void sire_calcscale(int,int,int,SIRE_FLAGS, SIRE_IMGINFO,SIRE_COMPLEX *, int,
                   float **,float **, float **);
void sire_cuppen3Drcn(int, int, int, SIRE_DIRINFO, SIRE_DIRINFO, SIRE_DIRINFO,
                      float ***, SIRE_COMPLEX *, int);
void sire_deletefiles(int, char**);
void sire_findcntr(int, int, int, SIRE_COMPLEX *, SIRE_DIRINFO *);
void sire_halfshiftcmplx3D (SIRE_COMPLEX *, int, int, int, int);
void sire_homodyne3Drcn(int, int, int, SIRE_DIRINFO, SIRE_DIRINFO,
                        SIRE_DIRINFO, float ***, SIRE_COMPLEX *);
void sire_insertrcvr(int, int,int,int,int,int,SIRE_DIRINFO,SIRE_IMGINFO,int,
                     float *, float *, SIRE_COMPLEX *, int, int, int,
                     short int *);
int ispowerof2(int);
void sire_overlaprcvrimg(char *, int, char **, int, int, int);
void sire_rcntrfprof(int,int,int,int,SIRE_DIRINFO,SIRE_COMPLEX *,int,double *);
void sire_rdscale(int, int, float *, float *, float *);
void sire_revcmplx3D(SIRE_COMPLEX *, int, int, int, int);
void sire_revconjkz (int,int,int,int,int,int,SIRE_DIRINFO *,SIRE_COMPLEX *);
void sire_shiftcmplx3D (SIRE_COMPLEX *, int, int, int, int, int, int);
void sire_shiftrcvr(int, int, int, SIRE_DIRINFO, SIRE_DIRINFO, int, int, int,
                    SIRE_COMPLEX *, SIRE_COMPLEX *, int, short **);
void sire_truncate(int, int, int,SIRE_DIRINFO, SIRE_DIRINFO, SIRE_COMPLEX *);
void sire_wrtfilter(int, int, int, SIRE_FILTERINFO *, float ***);
void sire_wrtscale(int, SIRE_IMGINFO, int, float *, float *, float *);
void sire_wrtsireinp(char [], SIRE_FLAGS, char [], SIRE_DIRINFO,
                     SIRE_FILTERINFO [], SIRE_IMGINFO, SIRE_SIGNA_IMGINFO,
                     double, char [], char []);
}

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class SiReCrunch : public Module {
    VoidStarIPort* iport;
    VoidStarOPort* oport;
    float *ScalePt, *RawImgDC, *ScaleFactor;
public:
    SiReCrunch(const clString& id);
    virtual ~SiReCrunch();
    virtual void execute();
};

extern "C" Module* make_SiReCrunch(const clString& id)
{
    return new SiReCrunch(id);
}

SiReCrunch::SiReCrunch(const clString& id)
: Module("SiReCrunch", id, Filter), ScalePt(0), RawImgDC(0), ScaleFactor(0)
{
    iport = scinew VoidStarIPort(this, "SiReData", VoidStarIPort::Atomic);
    add_iport(iport);
    oport = scinew VoidStarOPort(this, "SiReData", VoidStarIPort::Atomic);
    add_oport(oport);
}

SiReCrunch::~SiReCrunch()
{
}

void SiReCrunch::execute()
{
   /* Misc. variables */
   int                    IStepRead, IStepPhase, IStepSlice;
   int                    AllocFilter;
   int                    CalcScale, AllocScale;
   int                    FindCntr, CalcRcvrDCOffInd, CalcBBShift;


   double                 BBRFShift;
   SIRE_DIRINFO           EchoCntr, EchoSymWidth;


   VoidStarHandle vsh;
   if (!iport->get(vsh)) return;
   SiReData *s;
   if (!vsh.get_rep() || !(s=dynamic_cast<SiReData* >(vsh.get_rep()))) return;

   cerr << "\n\nSIRECRUNCH\n\n\n";
   s->lockstepSem.up();
   if (s->s.IRcvr == s->s.Flag.StartRcvr && s->s.ISlab == s->s.Flag.StartSlab){
       AllocFilter = SIRE_YES;   /* Allocate Filter, SymRcvrRaw arrays */
       FindCntr = SIRE_YES;      /* Find the center at least the first time */
       CalcBBShift = SIRE_YES;   /* Calculate the shift the first time */
       CalcScale = SIRE_YES;     /* Calculate the scaling factors */
       AllocScale = SIRE_YES;    /* Allocate scale factor arrays */
   } else {
       AllocFilter = SIRE_NO;    /* Allocated from the first time */
       FindCntr = SIRE_NO;       /* Find the center at least the first time */
       CalcBBShift = SIRE_NO;    /* Calculate the shift the first time */
       CalcScale = SIRE_NO;      /* Calculate the scaling factors */
       AllocScale = SIRE_NO;     /* Allocate scale factor arrays */
   }       


   /* Calculate scaling factors for each slab in 3dFSE bb */
   if (s->s.Flag.FlattenBB == SIRE_YES)
       CalcScale = SIRE_YES;

   /* Transform the z-direction back to kz */
   if (s->s.Flag.InitReconkz == SIRE_YES)
       {
#if DEBUG
	   printf("      Re-inverting slice direction\n");
	   cerr << "2. nslice: " << s->s.NSlicePerRcvr << '\n';
	   cerr << "2. rawread: " << s->s.NRawRead << '\n';
	   cerr << "2. phase: " << s->s.NRawPhase << '\n';
	   cerr << "2. total: " << s->s.NSlicePerRcvr*s->s.NRawRead*s->s.NRawPhase << '\n';

	    printf("Before shift (nsliceperrcvr=%d)...\n",s->s.NSlicePerRcvr);
	    for (int i=0; i<s->s.NSlicePerRcvr*s->s.NRawPhase*s->s.NRawRead; i+=3207) printf("(%d) - %lf %lf  ", i/3207, s->s.RcvrRaw[i].re, s->s.RcvrRaw[i].im);
	   printf("\n");
#endif
	   sire_3Dfftdft(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr, 
			 s->s.RcvrRaw, SIRE_ZAXIS, SIRE_FORWARD, NULL, 
			 SIRE_RETFFTCOMPLEX);
#if DEBUG
	   cerr << "After shift...\n";
	   for (int qq=0; qq<s->s.NSlicePerRcvr; qq++)
	       fprintf(stderr, "(%d) - %lf %lf  ", qq, (double)s->s.RcvrRaw[qq*s->s.NRawPhase*s->s.NRawRead].re, (double)s->s.RcvrRaw[qq*s->s.NRawPhase*s->s.NRawRead].im);
	   fprintf(stderr, "\n");
#endif
       }

   /* Prepare k-space so that the image is shifted to the center */
   if (s->s.Flag.AcqType != SIRE_FASTCARDACQ)
       sire_halfshiftcmplx3D(s->s.RcvrRaw, s->s.NSlicePerRcvr, s->s.NPhase, 
			     s->s.NRead, SIRE_READPHASEAXIS);
   else
       sire_halfshiftcmplx3D(s->s.RcvrRaw, s->s.NSlicePerRcvr, s->s.NPhase, 
			     s->s.NRead, SIRE_READSLICEAXIS);

   /* Find the echo center */
   if ((s->s.Flag.FindEveryCntr == SIRE_YES) || (FindCntr == SIRE_YES))
       {
	   printf("      Determining echo center\n");
	   if (s->s.Flag.AsymmetricEcho.Slice == SIRE_NO)
               sire_findcntr(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr, 
			     s->s.RcvrRaw, &EchoCntr);
	   else
               sire_findcntr(s->s.NRead, s->s.NPhase, s->s.Flag.NkzAcquired,
			     s->s.RcvrRaw,&EchoCntr);
	   CalcRcvrDCOffInd = SIRE_YES;
       }

   /* For black blood data, reverse and conjugate k-space */
   if ((s->s.Flag.AcqType == SIRE_3DFSEBBACQ) &&
       (s->s.Flag.AsymmetricEcho.Slice == SIRE_YES))
       {
	   printf("      Reversing/conjugating asymmetric kz\n");
	   sire_revconjkz(s->s.NRead, s->s.NRawRead, s->s.NPhase, 
			  s->s.NRawPhase, s->s.NSlicePerRcvr, 
			  s->s.Flag.NkzAcquired, &EchoCntr, s->s.RcvrRaw);
       }

   /* Truncate as directed */
   if ((s->s.Flag.Trunc.Read == SIRE_NO) &&
       (s->s.Flag.AsymmetricEcho.Read == SIRE_YES) &&
       (ispowerof2(s->s.NRawRead) == SIRE_YES))
       {
	   printf("      WARNING!  Asymmetric data in readout direction not");
	   printf("being truncated.\n");
       }
   if ((s->s.Flag.Trunc.Read == SIRE_YES) || 
       (s->s.Flag.Trunc.Phase == SIRE_YES) ||
       (s->s.Flag.Trunc.Slice == SIRE_YES))
       {
	   printf("      Truncating asymmetric data\n");
	   sire_truncate(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr, 
			 EchoCntr, s->s.Flag.Trunc, s->s.RcvrRaw);
       }

   /* For black blood data, recenter the profile */
   if (s->s.Flag.AcqType == SIRE_3DFSEBBACQ)
       {
	   printf("      Centering black blood rf profile\n");
	   sire_rcntrfprof(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr, 
			   s->s.Step.Slice, EchoCntr, s->s.RcvrRaw, 
			   CalcBBShift, &BBRFShift);
       } 

   /* Update filter widths and calculate filter values */
   if ((s->s.Flag.ReconType != SIRE_DEFAULTRECON) && 
       (AllocFilter == SIRE_YES))
       {
	   printf("      Checking/calculating filter widths\n");
	   sire_calcfilterw(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr, 
			    s->s.Flag, EchoCntr, &EchoSymWidth, 
			    s->s.FilterInfo);
	   printf("      Calculating/loading filter values\n");
	   sire_calcfilter(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr, 
			   AllocFilter, s->s.FilterInfo, EchoCntr, 
			   &s->s.Filter);
	   if (s->s.Flag.SaveFilter == SIRE_YES)
	       {
		   printf("      Saving filter data to file.\n");
		   sire_wrtfilter(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr,
				  s->s.FilterInfo,s->s.Filter);
	       }
       }
   /* Save run time parameters */
   if (AllocFilter == SIRE_YES)
       {
	   printf("      Writing second sire.inp to sire.runtime.2\n");
	   sire_wrtsireinp("sire.runtime.2", s->s.Flag, s->s.FirstPFile, 
			   s->s.Step,s->s.FilterInfo, s->s.SireImgInfo, 
			   s->s.SignaImgInfo, NULL,NULL,SIREVERSION);
       }
   
   /* ------------------------------------------------------------- */
   /* Perform zero filled interpolation */
   /* ------------------------------------------------------------- */
   cerr << "s->s.Step.{Read,Phase,Slice} = ("<<s->s.Step.Read<<","<<s->s.Step.Phase<<","<<s->s.Step.Slice<<")\n";
   for (IStepRead = 0; IStepRead < s->s.Step.Read; IStepRead++)
       for (IStepPhase = 0; IStepPhase < s->s.Step.Phase; IStepPhase++)
	   for (IStepSlice = 0; IStepSlice < s->s.Step.Slice; IStepSlice++) {
	       printf("      ZFI: read %d of %d, ", IStepRead+1, 
		      s->s.Step.Read);
	       printf("phase %d of %d, ", IStepPhase+1, s->s.Step.Phase);
	       printf("slice %d of %d ", IStepSlice+1, s->s.Step.Slice);
	       printf("(%d of %d)\n", IStepRead * s->s.Step.Phase * 
		      s->s.Step.Slice + IStepPhase * s->s.Step.Slice + 
		      IStepSlice + 1, s->s.Step.Read * s->s.Step.Phase * 
		      s->s.Step.Slice);
	       
	       /* Shift the slice for the ZFI */
	       printf("         Shifting k-space\n");
	       if ((s->s.Flag.AcqType == SIRE_3DFSEBBACQ) &&
		   (s->s.Flag.AsymmetricEcho.Slice == SIRE_YES))
		   sire_shiftrcvr(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr, 
				  s->s.Step, EchoCntr, -IStepRead, -IStepPhase,
				  IStepSlice, s->s.RcvrRaw, s->s.ShiftRcvrRaw, 
				  CalcRcvrDCOffInd, s->s.RcvrDCOffInd);
	       else
		   sire_shiftrcvr(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr, 
				  s->s.Step, EchoCntr,IStepRead, IStepPhase, 
				  IStepSlice, s->s.RcvrRaw, s->s.ShiftRcvrRaw, 
				  CalcRcvrDCOffInd, s->s.RcvrDCOffInd);
		   
	       if ((IStepRead !=0) || (IStepPhase !=0) || (IStepSlice !=0))
		   CalcRcvrDCOffInd = SIRE_NO;

	       /* Reconstruct */
	       switch(s->s.Flag.ReconType) {
	       case SIRE_HOMODYNERECON: { /* Homodyne reconstruction */
		   printf("         Homodyne reconstruction\n");
		   sire_homodyne3Drcn(s->s.NRead, s->s.NPhase, 
				      s->s.NSlicePerRcvr,
				      s->s.Flag.AsymmetricEcho, EchoCntr,
				      EchoSymWidth, s->s.Filter, 
				      s->s.ShiftRcvrRaw);
		   break;
	       }
	       case SIRE_CUPPENRECON: { /* Cuppen pocs algorithm */
		   printf("         Cuppen reconstruction\n");
		   sire_cuppen3Drcn(s->s.NRead, s->s.NPhase, 
				    s->s.NSlicePerRcvr,
				    s->s.Flag.AsymmetricEcho, EchoCntr,
				    EchoSymWidth, s->s.Filter, 
				    s->s.ShiftRcvrRaw,
				    s->s.Flag.NCuppenIter);
		   break;
	       }
	       case SIRE_FILTERECON: { /* Filtered reconstruction */
		   printf("         Filtered reconstruction\n");
		   sire_3Dfftdft(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr,
				 s->s.ShiftRcvrRaw, SIRE_ALLAXES,SIRE_BACKWARD,
				 s->s.Filter[0], SIRE_RETFFTMAGPHA);
		   break;
	       }
	       default: {  /* With FilterType set to none */
		   printf("         Default reconstruction\n");
		   if (s->s.SireImgInfo.RealPartOut == SIRE_NO) {
		       sire_shiftcmplx3D(s->s.ShiftRcvrRaw, s->s.NSlicePerRcvr,
					 s->s.NPhase, s->s.NRead, 
					 -EchoCntr.Slice, -EchoCntr.Phase, 
					 -EchoCntr.Read);
		       sire_3Dfftdft(s->s.NRead, s->s.NPhase, 
				     s->s.NSlicePerRcvr,
				     s->s.ShiftRcvrRaw, SIRE_ALLAXES,
				     SIRE_BACKWARD, NULL,
				     SIRE_RETFFTMAGPHA);
		   } else {
		       cerr << "STARTING FFT!\n";

#if DEBUG
		       for (int qq=0; qq<s->s.NSlicePerRcvr; qq++)
			   fprintf(stderr, "(%d) - %lf %lf  ", qq, (double)s->s.ShiftRcvrRaw[qq*s->s.NPhase*s->s.NRead].re, (double)s->s.ShiftRcvrRaw[qq*s->s.NPhase*s->s.NRead].im);
		       fprintf(stderr, "\n");
#endif
		       sire_3Dfftdft(s->s.NRead, s->s.NPhase, 
				     s->s.NSlicePerRcvr,
				     s->s.ShiftRcvrRaw, SIRE_ALLAXES,
				     SIRE_BACKWARD, NULL,
				     SIRE_RETFFTMAGONLY);
		       cerr << "DONE WITH FFT!\n";
		   }
		   break;
	       }
	       }

	       /* Reverse directions when necessary */
	       if ((s->s.Flag.AcqType == SIRE_3DFSEBBACQ) &&
		   (s->s.Flag.AsymmetricEcho.Slice == SIRE_YES)) {
		   printf("         Reversing x and y directions.\n");
		   sire_revcmplx3D(s->s.ShiftRcvrRaw, s->s.NSlicePerRcvr,
				   s->s.NPhase,s->s.NRead, SIRE_READPHASEAXIS);
	       } 
                  
	       printf("         Scaling image and ");
	       printf("inserting into ZFI image\n");
	       /* Determine the scaling factors */
	       if (CalcScale == SIRE_YES) {
		   if ((s->s.Flag.FlattenBB == SIRE_YES) && (s->s.IRcvr > 0))
		       sire_rdscale(s->s.ISlab, s->s.NSlicePerRcvr, ScalePt, 
				    RawImgDC, ScaleFactor);
		   else
		       sire_calcscale(s->s.NRead, s->s.NPhase, 
				      s->s.NSlicePerRcvr, s->s.Flag,
				      s->s.SireImgInfo, s->s.ShiftRcvrRaw,
				      AllocScale, &ScalePt, &RawImgDC, 
				      &ScaleFactor);
		   if (s->s.Flag.SaveScale == SIRE_YES)
		       sire_wrtscale(s->s.NSlicePerRcvr, s->s.SireImgInfo,
				     AllocScale, ScalePt, RawImgDC,
				     ScaleFactor);
	       }

	       /* Insert into larger ZFI receiver, blanking boundaries */
	       sire_insertrcvr(s->s.NRead, s->s.NPhase, s->s.NSlicePerRcvr, 
			       IStepRead, IStepPhase, IStepSlice, s->s.Step, 
			       s->s.SireImgInfo, s->s.NSlBlank, RawImgDC,
			       ScaleFactor, s->s.ShiftRcvrRaw, s->s.NRcnRead, 
			       s->s.NRcnPhase, s->s.NRcnSlicePerRcvr,
			       s->s.ZFIRcvrImg);
	   } /* end stepping through ZFI */
   oport->send(vsh);
   s->lockstepSem.down();
} // End namespace DaveW
}

