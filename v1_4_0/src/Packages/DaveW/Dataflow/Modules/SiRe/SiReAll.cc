
/*
 *  SiReAll.cc:  Put all of John Roberts' SiRe code in a Dataflow module
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
#include <Core/Datatypes/VoidStar.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Tester/RigorousTest.h>

#include <fstream>
#include <iostream>
using std::cerr;
#include <stdio.h>

/* Prototypes */
extern "C" {
void sire_3Dfftdft(int, int,int,SIRE_COMPLEX *, int, int, float **, int);
void sire_allocwork (int,int,int,int,int,int,int,SIRE_COMPLEX **,short ***,
                     short **);
void sire_calcfilter(int,int,int,int,SIRE_FILTERINFO *,SIRE_DIRINFO,
                     float ****);
void sire_calcfilterw(int, int, int, SIRE_FLAGS, SIRE_DIRINFO, SIRE_DIRINFO *,
                      SIRE_FILTERINFO *);
void sire_calconparam (char [], int, int, int, int, int, int, int, int,
                       int, int,SIRE_FLAGS *,SIRE_DIRINFO *, char ***, int **,
                       char ****, char ***, char **, int *, int *, int *,
                       int *,int *, int *, int *, int *, SIRE_SIGNA_IMGINFO *);
void sire_calcscale(int,int,int,SIRE_FLAGS, SIRE_IMGINFO,SIRE_COMPLEX *, int,
                   float **,float **, float **);
void sire_cuppen3Drcn(int, int, int, SIRE_DIRINFO, SIRE_DIRINFO, SIRE_DIRINFO,
                      float ***, SIRE_COMPLEX *, int);
void sire_deletefiles(int, char**);
void sire_findcntr(int, int, int, SIRE_COMPLEX *, SIRE_DIRINFO *);
void sire_getnonuserinp(char [], int *, int *, int *, int *, int *, int *,
                        int *, int *, int *, int *, int *, int *,
                        SIRE_FLAGS *, SIRE_SIGNA_IMGINFO *);
void sire_getime(int, time_t *, double *, char []);
void sire_halfshiftcmplx3D (SIRE_COMPLEX *, int, int, int, int);
void sire_homodyne3Drcn(int, int, int, SIRE_DIRINFO, SIRE_DIRINFO,
                        SIRE_DIRINFO, float ***, SIRE_COMPLEX *);
void sire_init(void);
void sire_inplace_overlapslabimg(char *,int, int, int, int, int, int,
                                 short *, int, short **);
void sire_insertrcvr(int, int,int,int,int,int,SIRE_DIRINFO,SIRE_IMGINFO,int,
                     float *, float *, SIRE_COMPLEX *, int, int, int,
                     short int *);
int ispowerof2(int);
void sire_makesigna(char [], int, int, int, SIRE_FLAGS, SIRE_DIRINFO,
                    SIRE_SIGNA_IMGINFO *);
void sire_meshslabimg(char [], char **, int, int, int, int);
void sire_overlaprcvrimg(char *, int, char **, int, int, int);
void sire_overlapslabimg(char *,char **,int,int,int, int, int, int, short *);
void sire_rcntrfprof(int,int,int,int,SIRE_DIRINFO,SIRE_COMPLEX *,int,double *);
void sire_rdscale(int, int, float *, float *, float *);
void sire_rdsipfilercvr (char [], int, SIRE_FLAGS, int, int, int, int, int,
                         int, int, int, int, int, SIRE_COMPLEX **);
void sire_rdsireinp(char [], SIRE_FLAGS *, char [], SIRE_DIRINFO *,
                    SIRE_FILTERINFO *, SIRE_IMGINFO *, SIRE_SIGNA_IMGINFO *);
void sire_revcmplx3D(SIRE_COMPLEX *, int, int, int, int);
void sire_revconjkz (int,int,int,int,int,int,SIRE_DIRINFO *,SIRE_COMPLEX *);
void sire_shiftcmplx3D (SIRE_COMPLEX *, int, int, int, int, int, int);
void sire_shiftrcvr(int, int, int, SIRE_DIRINFO, SIRE_DIRINFO, int, int, int,
                    SIRE_COMPLEX *, SIRE_COMPLEX *, int, short **);
void sire_truncate(int, int, int,SIRE_DIRINFO, SIRE_DIRINFO, SIRE_COMPLEX *);
void sire_wrtfilter(int, int, int, SIRE_FILTERINFO *, float ***);
void sire_wrtrcvrimg(char *, int, int, int, short int *);
void sire_wrtscale(int, SIRE_IMGINFO, int, float *, float *, float *);
void sire_wrtsireinp(char [], SIRE_FLAGS, char [], SIRE_DIRINFO,
                     SIRE_FILTERINFO [], SIRE_IMGINFO, SIRE_SIGNA_IMGINFO,
                     double, char [], char []);
}

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class SiReAll : public Module {
public:
    SiReAll(const clString& id);
    virtual ~SiReAll();
    virtual void execute();
};

extern "C" Module* make_SiReAll(const clString& id)
{
    return new SiReAll(id);
}

SiReAll::SiReAll(const clString& id)
: Module("SiReAll", id, Filter)
{
}

SiReAll::~SiReAll()
{
}

void SiReAll::execute()
{
   SiReData* ss = new SiReData;
   VoidStarHandle vsh(ss);

   int                    IStepRead, IStepPhase, IStepSlice;
   int                    AllocRcvr, AllocFilter, AllocWork;
   int                    CalcScale, AllocScale;
   int                    FindCntr, CalcRcvrDCOffInd, CalcBBShift;

   float                  *ScalePt=NULL, *RawImgDC=NULL, *ScaleFactor=NULL;
   double                 BBRFShift;
   SIRE_DIRINFO           EchoCntr, EchoSymWidth;
//   int                    TempIndex;

   /* ---------------------------------------------------------------------- */
   /* Get inputs and control parameters */
   /* ---------------------------------------------------------------------- */

   /* Initial message */
   printf("\n\n\nSiRe Reconstruction Program\n");
   printf("\nGetting inputs:\n");

   /* Initialization */
//   sire_init();
   sire_getime (SIRE_YES, &ss->s.TimeStamp, &ss->s.RunTime, ss->s.TimeStr);

   /* User inputs */
   printf("   Retrieving user inputs\n");
   sire_rdsireinp(SIRE_USERINPFILE, &ss->s.Flag, ss->s.FirstPFile, &ss->s.Step, ss->s.FilterInfo,
                   &ss->s.SireImgInfo, &ss->s.SignaImgInfo);

   /* Get nonuser inputs */
   printf("   Getting non-user inputs\n");
   sire_getnonuserinp(ss->s.FirstPFile, &ss->s.PointSize, &ss->s.NRead, &ss->s.NPhase, &ss->s.NRawRead,
                      &ss->s.NRawPhase, &ss->s.NRawSlice, &ss->s.NSlabPerPFile, &ss->s.NPFile,
                      &ss->s.NSlBlank, &ss->s.Rcvr0, &ss->s.RcvrN, &ss->s.NFinalImgSlice, &ss->s.Flag,
                      &ss->s.SignaImgInfo);

   /* Calculate remaining control parameters */
   printf("   Calculating run-time parameters\n");
   sire_calconparam (ss->s.FirstPFile, ss->s.NRead, ss->s.NPhase, ss->s.NRawSlice, ss->s.NSlabPerPFile,
                     ss->s.NPFile, ss->s.NSlBlank, ss->s.Rcvr0, ss->s.RcvrN, ss->s.NFinalImgSlice,
                     SIRE_YES, &ss->s.Flag, &ss->s.Step, &ss->s.PFiles, &ss->s.SlabIndices,
                     &ss->s.RcvrSlabImgFiles, &ss->s.Rcvr3DImgFiles, &ss->s.FinalImgFile,
                     &ss->s.NSlicePerRcvr, &ss->s.NRcvrPerSlab, &ss->s.NSlab, &ss->s.NRcnRead,
                     &ss->s.NRcnPhase, &ss->s.NRcnSlicePerRcvr, &ss->s.NRcnOverlap,
                     &ss->s.NRcnFinalImgSlice, &ss->s.SignaImgInfo);

   /* Save run time parameters */
   printf("   Writing first sire.inp to sire.RunTime.1\n");
   sire_wrtsireinp("sire.RunTime.1", ss->s.Flag, ss->s.FirstPFile, ss->s.Step, ss->s.FilterInfo,
                   ss->s.SireImgInfo, ss->s.SignaImgInfo, NULL, NULL, SIREVERSION);

   /* -------------------------------------------------------------------- */
   /* Allocate working arrays */
   /* -------------------------------------------------------------------- */
   AllocWork = SIRE_YES;        /* Allocate working arrays */
   if ((ss->s.Flag.ReconRcvrSlabs == SIRE_YES) || (ss->s.Flag.OverlapRcvrSlabs==SIRE_YES))
   {
      printf("\nAllocating working arrays\n");
      sire_allocwork(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.NRcnRead, ss->s.NRcnPhase,
                     ss->s.NRcnSlicePerRcvr, AllocWork, &ss->s.ShiftRcvrRaw,&ss->s.RcvrDCOffInd,
                     &ss->s.ZFIRcvrImg);
   }

   /* -------------------------------------------------------------------- */
   /* Loop through receiver data sets */
   /* -------------------------------------------------------------------- */

   /* Initialize additional control parameters */
   AllocRcvr = SIRE_YES;        /* Causes allocation of the RcvrRaw, filter */
   AllocFilter = SIRE_YES;      /*    and SymRcvrRaw arrays the first pass */
   FindCntr = SIRE_YES;         /* Find the center at least the first time */
   CalcBBShift = SIRE_YES;      /* Calculate the shift the first time */
   CalcScale = SIRE_YES;        /* Calculate the scaling factors */
   AllocScale = SIRE_YES;       /* Allocate scale factor arrays */

   if (ss->s.Flag.ReconRcvrSlabs == SIRE_YES)
      printf("\nLooping through receivers\n");
   for (ss->s.IRcvr=ss->s.Flag.StartRcvr; ss->s.IRcvr<=ss->s.Flag.EndRcvr; ss->s.IRcvr++)
   {

      if ((ss->s.Flag.ReconRcvrSlabs==SIRE_YES) || (ss->s.Flag.OverlapRcvrSlabs==SIRE_YES))
      {
         printf("\n   Receiver %d of %d ",ss->s.IRcvr+1,ss->s.NRcvrPerSlab);
         printf("(Reconstruction %d of %d)\n", ss->s.IRcvr - ss->s.Flag.StartRcvr + 1,
                ss->s.Flag.EndRcvr - ss->s.Flag.StartRcvr + 1);
      }

      /* Set initial indices based on user inputs */
      ss->s.FirstSlab = (ss->s.IRcvr == ss->s.Flag.StartRcvr) ? ss->s.Flag.StartSlab : 0;
      ss->s.LastSlab = (ss->s.IRcvr == ss->s.Flag.EndRcvr) ? ss->s.Flag.EndSlab : (ss->s.NSlab-1);

      for (ss->s.ISlab=ss->s.FirstSlab; (ss->s.ISlab<=ss->s.LastSlab) &&
           (ss->s.Flag.ReconRcvrSlabs == SIRE_YES); ss->s.ISlab++)
      {

	  char fname[100];
	  sprintf(fname, "sire.rcvr%d.slab%d.pre.vs", ss->s.IRcvr,ss->s.ISlab);
	  TextPiostream stream(fname, Piostream::Write);
	  Pio(stream, vsh);

         /* Calculate scaling factors for each slab in 3dFSE bb */
         if (ss->s.Flag.FlattenBB == SIRE_YES)
            CalcScale = SIRE_YES;

         if (ss->s.ISlab > ss->s.FirstSlab)
         {
            printf("\n   Continuing receiver %d of %d ",ss->s.IRcvr+1,ss->s.NRcvrPerSlab);
            printf("(Reconstruction %d of %d)\n", ss->s.IRcvr - ss->s.Flag.StartRcvr + 1,
                   ss->s.Flag.EndRcvr - ss->s.Flag.StartRcvr + 1);
         }

         /* Read in a slab for a receiver */
         printf("      Reading raw data from %s (Slab %d of %d)\n",
                ss->s.PFiles[ss->s.ISlab], ss->s.ISlab+1, ss->s.NSlab);
         sire_rdsipfilercvr(ss->s.PFiles[ss->s.ISlab], ss->s.PointSize, ss->s.Flag, ss->s.NRead,
                            ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.NRawRead, ss->s.NRawPhase,
                            ss->s.NRcvrPerSlab, ss->s.NSlabPerPFile, AllocRcvr,
                            ss->s.SlabIndices[ss->s.ISlab], ss->s.IRcvr, &ss->s.RcvrRaw);

         AllocRcvr = SIRE_NO;   /* Turn off further allocation or RcvrRaw */

    cerr << "nslice: " << ss->s.NSlicePerRcvr << '\n';
    cerr << "rawread: " << ss->s.NRawRead << '\n';
    cerr << "phase: " << ss->s.NRawPhase << '\n';
    cerr << "read: " << ss->s.NRead << '\n';
    cerr << "phase: " << ss->s.NPhase << '\n';
    cerr << "total: " << ss->s.NSlicePerRcvr*ss->s.NRawRead*ss->s.NRawPhase << '\n';
         if (ss->s.Flag.InitReconkz == SIRE_YES)
         {
            printf("      Re-inverting slice direction\n");

	    printf("Before shift (nsliceperrcvr=%d)...\n",ss->s.NSlicePerRcvr);
	    for (int i=0; i<ss->s.NSlicePerRcvr*ss->s.NRawPhase*ss->s.NRawRead; i+=3207) printf("(%d) - %lf %lf  ", i/3207, ss->s.RcvrRaw[i].re, ss->s.RcvrRaw[i].im);

            sire_3Dfftdft(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, 
			  ss->s.RcvrRaw, SIRE_ZAXIS, SIRE_FORWARD, NULL, 
			  SIRE_RETFFTCOMPLEX);

	    printf(" After shift...\n");
	    for (int qq=0; qq<ss->s.NSlicePerRcvr; qq++)
		printf("(%d) - %lf %lf  ", qq, (double)ss->s.RcvrRaw[qq*ss->s.NRawPhase*ss->s.NRawRead].re, (double)ss->s.RcvrRaw[qq*ss->s.NRawPhase*ss->s.NRawRead].im);
	    printf("\n");
         }

         /* Prepare k-space so that the image is shifted to the center */
         if (ss->s.Flag.AcqType != SIRE_FASTCARDACQ)
            sire_halfshiftcmplx3D(ss->s.RcvrRaw, ss->s.NSlicePerRcvr, ss->s.NPhase, ss->s.NRead,
                                  SIRE_READPHASEAXIS);
         else
            sire_halfshiftcmplx3D(ss->s.RcvrRaw, ss->s.NSlicePerRcvr, ss->s.NPhase, ss->s.NRead,
                                  SIRE_READSLICEAXIS);

         /* Find the echo center */
         if ((ss->s.Flag.FindEveryCntr == SIRE_YES) || (FindCntr == SIRE_YES))
         {
            printf("      Determining echo center\n");
            if (ss->s.Flag.AsymmetricEcho.Slice == SIRE_NO)
               sire_findcntr(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.RcvrRaw, &EchoCntr);
            else
               sire_findcntr(ss->s.NRead,ss->s.NPhase,ss->s.Flag.NkzAcquired,ss->s.RcvrRaw,&EchoCntr);
            FindCntr = SIRE_NO;
            CalcRcvrDCOffInd = SIRE_YES;
         }

         /* For black blood data, reverse and conjugate k-space */
         if ((ss->s.Flag.AcqType == SIRE_3DFSEBBACQ) &&
             (ss->s.Flag.AsymmetricEcho.Slice == SIRE_YES))
         {
            printf("      Reversing/conjugating asymmetric kz\n");
            sire_revconjkz(ss->s.NRead, ss->s.NRawRead, ss->s.NPhase, ss->s.NRawPhase, ss->s.NSlicePerRcvr, 
                           ss->s.Flag.NkzAcquired, &EchoCntr, ss->s.RcvrRaw);
         }

         /* Truncate as directed */
         if ((ss->s.Flag.Trunc.Read == SIRE_NO) &&
             (ss->s.Flag.AsymmetricEcho.Read == SIRE_YES) &&
             (ispowerof2(ss->s.NRawRead) == SIRE_YES))
         {
            printf("      WARNING!  Asymmetric data in readout direction not");
            printf("being truncated.\n");
         }
         if ((ss->s.Flag.Trunc.Read == SIRE_YES) || (ss->s.Flag.Trunc.Phase == SIRE_YES) ||
             (ss->s.Flag.Trunc.Slice == SIRE_YES))
         {
            printf("      Truncating asymmetric data\n");
            sire_truncate(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, EchoCntr, ss->s.Flag.Trunc,
                          ss->s.RcvrRaw);
         }

         /* For black blood data, recenter the profile */
         if (ss->s.Flag.AcqType == SIRE_3DFSEBBACQ)
         {
            printf("      Centering black blood rf profile\n");
            sire_rcntrfprof(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.Step.Slice,
                            EchoCntr, ss->s.RcvrRaw, CalcBBShift, &BBRFShift);
            CalcBBShift = SIRE_NO;
         } 


         /* Update filter widths and calculate filter values */
         if ((ss->s.Flag.ReconType != SIRE_DEFAULTRECON) && 
             (AllocFilter == SIRE_YES))
         {
            printf("      Checking/calculating filter widths\n");
            sire_calcfilterw(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.Flag,
                             EchoCntr, &EchoSymWidth, ss->s.FilterInfo);
            printf("      Calculating/loading filter values\n");
            sire_calcfilter(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, AllocFilter,
                            ss->s.FilterInfo, EchoCntr, &ss->s.Filter);
            if (ss->s.Flag.SaveFilter == SIRE_YES)
            {
               printf("      Saving filter data to file.\n");
               sire_wrtfilter(ss->s.NRead,ss->s.NPhase,ss->s.NSlicePerRcvr,ss->s.FilterInfo,ss->s.Filter);
            }
         }
         /* Save run time parameters */
         if (AllocFilter == SIRE_YES)
         {
            printf("      Writing second sire.inp to sire.RunTime.2\n");
            sire_wrtsireinp("sire.RunTime.2",ss->s.Flag,ss->s.FirstPFile, ss->s.Step,ss->s.FilterInfo,
                            ss->s.SireImgInfo, ss->s.SignaImgInfo, NULL,NULL,SIREVERSION);
         }
         AllocFilter = SIRE_NO;

         /* ------------------------------------------------------------- */
         /* Perform zero filled interpolation */
         /* ------------------------------------------------------------- */
         for (IStepRead = 0; IStepRead < ss->s.Step.Read; IStepRead++)
            for (IStepPhase = 0; IStepPhase < ss->s.Step.Phase; IStepPhase++)
               for (IStepSlice = 0; IStepSlice < ss->s.Step.Slice; IStepSlice++)
               {

                  printf("      ZFI: read %d of %d, ",IStepRead+1,ss->s.Step.Read);
                  printf("phase %d of %d, ", IStepPhase+1, ss->s.Step.Phase);
                  printf("slice %d of %d ", IStepSlice+1, ss->s.Step.Slice);
                  printf("(%d of %d)\n", IStepRead * ss->s.Step.Phase * ss->s.Step.Slice +
                         IStepPhase * ss->s.Step.Slice + IStepSlice + 1,
                         ss->s.Step.Read * ss->s.Step.Phase * ss->s.Step.Slice);

                  /* Shift the slice for the ZFI */
                  printf("         Shifting k-space\n");
                  if ((ss->s.Flag.AcqType == SIRE_3DFSEBBACQ) &&
                      (ss->s.Flag.AsymmetricEcho.Slice == SIRE_YES))
                     sire_shiftrcvr(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.Step,
                                    EchoCntr,-IStepRead,-IStepPhase,IStepSlice,
                                    ss->s.RcvrRaw, ss->s.ShiftRcvrRaw, CalcRcvrDCOffInd,
                                    ss->s.RcvrDCOffInd);
                  else
                     sire_shiftrcvr(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.Step,
                                    EchoCntr,IStepRead, IStepPhase, IStepSlice,
                                    ss->s.RcvrRaw, ss->s.ShiftRcvrRaw, CalcRcvrDCOffInd,
                                    ss->s.RcvrDCOffInd);

                  if ((IStepRead !=0) || (IStepPhase !=0) || (IStepSlice !=0))
                     CalcRcvrDCOffInd = SIRE_NO;

                  /* Reconstruct */
                  switch(ss->s.Flag.ReconType)
                  {
                     case SIRE_HOMODYNERECON: /* Homodyne reconstruction */
                     {
                        printf("         Homodyne reconstruction\n");
                        sire_homodyne3Drcn(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr,
                                           ss->s.Flag.AsymmetricEcho, EchoCntr,
                                           EchoSymWidth, ss->s.Filter, ss->s.ShiftRcvrRaw);
                        break;
                     }

                     case SIRE_CUPPENRECON: /* Cuppen pocs algorithm */
                     {
                        printf("         Cuppen reconstruction\n");
                        sire_cuppen3Drcn(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr,
                                         ss->s.Flag.AsymmetricEcho, EchoCntr,
                                         EchoSymWidth, ss->s.Filter, ss->s.ShiftRcvrRaw,
                                         ss->s.Flag.NCuppenIter);
                        break;
                     }

                     case SIRE_FILTERECON: /* Filtered reconstruction */
                     {
                        printf("         Filtered reconstruction\n");
                        sire_3Dfftdft(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr,
                                      ss->s.ShiftRcvrRaw, SIRE_ALLAXES,SIRE_BACKWARD,
                                      ss->s.Filter[0], SIRE_RETFFTMAGPHA);
                        break;
                     }

                     default:   /* With FilterType set to none */
                     {
                        printf("         Default reconstruction\n");
                        if (ss->s.SireImgInfo.RealPartOut == SIRE_NO)
                        {
                           sire_shiftcmplx3D(ss->s.ShiftRcvrRaw, ss->s.NSlicePerRcvr,
                                             ss->s.NPhase, ss->s.NRead, -EchoCntr.Slice,
                                             -EchoCntr.Phase, -EchoCntr.Read);
                           sire_3Dfftdft(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr,
                                         ss->s.ShiftRcvrRaw, SIRE_ALLAXES,
                                         SIRE_BACKWARD, NULL,
                                         SIRE_RETFFTMAGPHA);
                        }
                        else {

			    for (int qq=0; qq<ss->s.NSlicePerRcvr; qq++)
				fprintf(stderr, "(%d) - %lf %lf  ", qq, (double)ss->s.ShiftRcvrRaw[qq*ss->s.NPhase*ss->s.NRead].re, (double)ss->s.ShiftRcvrRaw[qq*ss->s.NPhase*ss->s.NRead].im);
			    fprintf(stderr, "\n");

                           sire_3Dfftdft(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr,
                                         ss->s.ShiftRcvrRaw, SIRE_ALLAXES,
                                         SIRE_BACKWARD, NULL,
                                         SIRE_RETFFTMAGONLY);
}
                        break;
                     }

                  }

                  /* Reverse directions when necessary */
                  if ((ss->s.Flag.AcqType == SIRE_3DFSEBBACQ) &&
                      (ss->s.Flag.AsymmetricEcho.Slice == SIRE_YES))
                  {
                     printf("         Reversing x and y directionss->s.\n");
                     sire_revcmplx3D(ss->s.ShiftRcvrRaw,ss->s.NSlicePerRcvr,ss->s.NPhase,ss->s.NRead,
                                     SIRE_READPHASEAXIS);
                  } 
                  
                  printf("         Scaling image and ");
                  printf("inserting into ZFI image\n");
                  /* Determine the scaling factors */
                  if (CalcScale == SIRE_YES)
                  {
                     if ((ss->s.Flag.FlattenBB == SIRE_YES) && (ss->s.IRcvr > 0))
                        sire_rdscale(ss->s.ISlab, ss->s.NSlicePerRcvr, ScalePt, RawImgDC,
                                     ScaleFactor);
                     else
                        sire_calcscale(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.Flag,
                                       ss->s.SireImgInfo,ss->s.ShiftRcvrRaw,AllocScale,
                                       &ScalePt, &RawImgDC, &ScaleFactor);
                     if (ss->s.Flag.SaveScale == SIRE_YES)
                        sire_wrtscale(ss->s.NSlicePerRcvr, ss->s.SireImgInfo,
                                      AllocScale, ScalePt, RawImgDC,
                                      ScaleFactor);
                     CalcScale = AllocScale = SIRE_NO;
                  }

                  /* Insert into larger ZFI receiver, blanking boundaries */
                  sire_insertrcvr(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, IStepRead,
                                  IStepPhase, IStepSlice, ss->s.Step, ss->s.SireImgInfo,
                                  ss->s.NSlBlank,RawImgDC,ScaleFactor,ss->s.ShiftRcvrRaw,
                                  ss->s.NRcnRead, ss->s.NRcnPhase, ss->s.NRcnSlicePerRcvr,
                                  ss->s.ZFIRcvrImg);

               } /* end stepping through ZFI */

         /* ------------------------------------------------------------ */
         /* Store receiver to disk, blanking boundary slices */
         /* ------------------------------------------------------------ */

	 int firstSlice = 0;
	 if (ss->s.ISlab != 0) 
	     firstSlice = (ss->s.ISlab * (ss->s.NRcnSlicePerRcvr - ss->s.NRcnOverlap));
	 int lastSlice = firstSlice + ss->s.NRcnSlicePerRcvr;
	 cerr << "NNRcnSlicePerRcvr="<<ss->s.NRcnSlicePerRcvr<<"  ss->s.NRcnOverlap="<<ss->s.NRcnOverlap<<"  firstSlice="<<firstSlice<<"  lastSlice="<<lastSlice<<"\n";
	 for (int qq=0; qq<ss->s.NRcnSlicePerRcvr; qq++)
	     fprintf(stderr, "(%d)-%d ", qq, (int)ss->s.ZFIRcvrImg[qq*ss->s.NRcnPhase*ss->s.NRcnRead+(ss->s.NRcnPhase/2)*ss->s.NRcnRead+ss->s.NRcnRead/2]);
	 fprintf(stderr, "\n");
         if (ss->s.Flag.InPlaceOverlap == SIRE_NO)
         {
            printf("      Saving receiver ZFI slab data to %s\n",
                   ss->s.RcvrSlabImgFiles[ss->s.IRcvr][ss->s.ISlab]);
            sire_wrtrcvrimg(ss->s.RcvrSlabImgFiles[ss->s.IRcvr][ss->s.ISlab], ss->s.NRcnRead,
                            ss->s.NRcnPhase, ss->s.NRcnSlicePerRcvr, ss->s.ZFIRcvrImg);
         }
         else
         {
            printf("      Updating ZFI receiver %s\n", ss->s.Rcvr3DImgFiles[ss->s.IRcvr]);
            sire_inplace_overlapslabimg(ss->s.Rcvr3DImgFiles[ss->s.IRcvr],
                                        ss->s.NSlab, ss->s.NRcnRead, ss->s.NRcnPhase,
                                        ss->s.NRcnSlicePerRcvr, ss->s.NRcnOverlap,
                                        ss->s.Flag.AutoCor, ss->s.ZFIRcvrImg,
                                        ss->s.ISlab, &ss->s.OverlapImg);
         }

	 sprintf(fname, "sire.rcvr%d.slab%d.post.vs", ss->s.IRcvr,ss->s.ISlab);
	 TextPiostream stream2(fname, Piostream::Write);
	 Pio(stream2, vsh);

      } /* end loop over slabs and ss->s.PFiles */

      /* Overlap receiver slabs */
      if ((ss->s.Flag.OverlapRcvrSlabs == SIRE_YES) &&
          (ss->s.Flag.InPlaceOverlap == SIRE_NO))
      {
         printf("      Overlapping slabs and saving in %s\n",
                ss->s.Rcvr3DImgFiles[ss->s.IRcvr]);
         if (ss->s.Flag.AcqType != SIRE_FASTCARDACQ)
            sire_overlapslabimg(ss->s.Rcvr3DImgFiles[ss->s.IRcvr], ss->s.RcvrSlabImgFiles[ss->s.IRcvr],
                                ss->s.NSlab, ss->s.NRcnRead, ss->s.NRcnPhase, ss->s.NRcnSlicePerRcvr,
                                ss->s.NRcnOverlap, ss->s.Flag.AutoCor, ss->s.ZFIRcvrImg);
         else
            sire_meshslabimg(ss->s.Rcvr3DImgFiles[ss->s.IRcvr], ss->s.RcvrSlabImgFiles[ss->s.IRcvr],
                             ss->s.NSlab, ss->s.NRcnRead, ss->s.NRcnPhase, ss->s.NRcnSlicePerRcvr);
      }

      /* Delete receiver slab files if not needed */
      if ((ss->s.Flag.SaveRcvrSlabs == SIRE_NO) && (ss->s.Flag.InPlaceOverlap==SIRE_NO) &&
          (ss->s.Flag.ReconRcvrSlabs == SIRE_YES))
      {
         printf("      Deleteing temporary slab files\n");
         sire_deletefiles(ss->s.NSlab,ss->s.RcvrSlabImgFiles[ss->s.IRcvr]);
      }

   } /* end receiver loop */

   /* Delete receiver slab files not hit by previous delete */
   if ((ss->s.Flag.SaveRcvrSlabs == SIRE_NO) && (ss->s.Flag.InPlaceOverlap==SIRE_NO) &&
       (ss->s.Flag.ReconRcvrSlabs == SIRE_YES))
   {
      printf("      Deleteing temporary slab files\n");
      for (ss->s.IRcvr = ss->s.Rcvr0; ss->s.IRcvr < ss->s.Flag.StartRcvr; ss->s.IRcvr++)
         sire_deletefiles(ss->s.NSlab,ss->s.RcvrSlabImgFiles[ss->s.IRcvr]);
      for (ss->s.IRcvr = ss->s.Flag.EndRcvr+1; ss->s.IRcvr <= ss->s.RcvrN; ss->s.IRcvr++)
         sire_deletefiles(ss->s.NSlab,ss->s.RcvrSlabImgFiles[ss->s.IRcvr]);
   }

#if 0
   /* -------------------------------------------------------------------- */
   /* Free up arrays used for reconstruction */
   /* -------------------------------------------------------------------- */
   if ((ss->s.ShiftRcvrRaw != NULL) || (ss->s.RcvrRaw != NULL) || (ss->s.Filter != NULL) ||
       (ScalePt != NULL))
   {
      printf("\nFreeing working arrays\n");
      if (ss->s.RcvrRaw != NULL)
      {
         AllocRcvr = -SIRE_YES;
         sire_rdsipfilercvr(ss->s.PFiles[ss->s.ISlab], ss->s.PointSize, ss->s.Flag, ss->s.NRead,
                            ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.NRawRead, ss->s.NRawPhase,
                            ss->s.NRcvrPerSlab, ss->s.NSlabPerPFile, AllocRcvr,
                            ss->s.SlabIndices[ss->s.ISlab], ss->s.IRcvr, &ss->s.RcvrRaw);
      }
      if (ss->s.ShiftRcvrRaw != NULL)
      {
         AllocWork = -SIRE_YES;
         sire_allocwork(ss->s.NRead,ss->s.NPhase,ss->s.NSlicePerRcvr, ss->s.NRcnRead, ss->s.NRcnPhase,
                        ss->s.NRcnSlicePerRcvr,AllocWork,&ss->s.ShiftRcvrRaw,
                        &ss->s.RcvrDCOffInd, &ss->s.ZFIRcvrImg);
      }
      if (ss->s.Filter != NULL)
      {
         AllocFilter = -SIRE_YES;
         sire_calcfilter(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, AllocFilter,
                         ss->s.FilterInfo, EchoCntr, &ss->s.Filter);
      }
      if (ScalePt != NULL)
      {
         AllocScale = -SIRE_YES;
         sire_calcscale(ss->s.NRead, ss->s.NPhase, ss->s.NSlicePerRcvr, ss->s.Flag, ss->s.SireImgInfo,
                       ss->s.ShiftRcvrRaw, AllocScale, &ScalePt, &RawImgDC,
                       &ScaleFactor);
      }
   }
#endif

   /* -------------------------------------------------------------------- */
   /* Compile the receivers into a single file */
   /* -------------------------------------------------------------------- */
   if (ss->s.Flag.MakeZFImg == SIRE_YES)
   {
      printf("\nCreating single 3D compilation of receivers: %s\n",
             ss->s.FinalImgFile);
      sire_overlaprcvrimg(ss->s.FinalImgFile, ss->s.NRcvrPerSlab, ss->s.Rcvr3DImgFiles, ss->s.NRcnRead,
                          ss->s.NRcnPhase, ss->s.NRcnFinalImgSlice);

      /* Delete receiver files if not needed */
      if (ss->s.Flag.SaveRcvrs == SIRE_NO)
      {
         printf("\nDeleting temporary receiver data files\n");
         sire_deletefiles(ss->s.NRcvrPerSlab, ss->s.Rcvr3DImgFiles);
      }

   }

   /* -------------------------------------------------------------------- */
   /* Create signa images if directed */
   /* -------------------------------------------------------------------- */
   if (ss->s.Flag.MakeSignaImg == SIRE_YES)
   {
      printf("\nCreating Signa images\n\n");
      sire_makesigna(ss->s.FinalImgFile, ss->s.NRcnRead, ss->s.NRcnPhase, ss->s.NRcnFinalImgSlice,
                     ss->s.Flag, ss->s.Step, &ss->s.SignaImgInfo);

      /* Delete ZFI file if not needed */
      if (ss->s.Flag.SaveZFImg == SIRE_NO)
      {
         printf("\nDeleteing 3D compilation data file\n");
         sire_deletefiles(1, &ss->s.FinalImgFile);
      }

   }

   /* -------------------------------------------------------------------- */
   /* Cleanup what is left */
   /* -------------------------------------------------------------------- */
#if 0
   /* Free up filenames and indices */
   sire_calconparam (ss->s.FirstPFile, ss->s.NRead, ss->s.NPhase, ss->s.NRawSlice, ss->s.NSlabPerPFile,
                     ss->s.NPFile,ss->s.NSlBlank, ss->s.Rcvr0, ss->s.RcvrN, ss->s.NFinalImgSlice,
                     -SIRE_YES, &ss->s.Flag, &ss->s.Step, &ss->s.PFiles, &ss->s.SlabIndices,
                     &ss->s.RcvrSlabImgFiles, &ss->s.Rcvr3DImgFiles, &ss->s.FinalImgFile,
                     &ss->s.NSlicePerRcvr, &ss->s.NRcvrPerSlab, &ss->s.NSlab, &ss->s.NRcnRead,
                     &ss->s.NRcnPhase, &ss->s.NRcnSlicePerRcvr, &ss->s.NRcnOverlap,
                     &ss->s.NRcnFinalImgSlice, &ss->s.SignaImgInfo);
#endif
   /* Complete the time stamp */
   sire_getime(SIRE_NO, &ss->s.TimeStamp, &ss->s.RunTime, ss->s.TimeStr);

   /* Create the output sire.inp file */
   printf("   Creating final sire.inp.date file\n");
   sire_wrtsireinp(SIRE_USERINPFILE, ss->s.Flag, ss->s.FirstPFile, ss->s.Step, ss->s.FilterInfo,
                   ss->s.SireImgInfo, ss->s.SignaImgInfo, ss->s.RunTime, ss->s.TimeStr, SIREVERSION);

   printf("\nSiRe Reconstruction program finished.\n");
} // End namespace DaveW
}


