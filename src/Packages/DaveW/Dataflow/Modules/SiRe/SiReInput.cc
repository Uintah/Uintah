
/*
 *  SiReInput.cc:  Read in the SiRe input data
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
#include <Core/Persistent/Pstreams.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Tester/RigorousTest.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

/* Prototypes */
extern "C" {
void sire_allocwork (int,int,int,int,int,int,int,SIRE_COMPLEX **,short ***,
                     short **);
void sire_calconparam (char [], int, int, int, int, int, int, int, int,
                       int, int,SIRE_FLAGS *,SIRE_DIRINFO *, char ***, int **,
                       char ****, char ***, char **, int *, int *, int *,
                       int *,int *, int *, int *, int *, SIRE_SIGNA_IMGINFO *);
void sire_getnonuserinp(char [], int *, int *, int *, int *, int *, int *,
                        int *, int *, int *, int *, int *, int *,
                        SIRE_FLAGS *, SIRE_SIGNA_IMGINFO *);
void sire_getime(int, time_t *, double *, char []);
void sire_rdsipfilercvr (char [], int, SIRE_FLAGS, int, int, int, int, int,
                         int, int, int, int, int, SIRE_COMPLEX **);
void sire_rdsireinp(char [], SIRE_FLAGS *, char [], SIRE_DIRINFO *,
                    SIRE_FILTERINFO *, SIRE_IMGINFO *, SIRE_SIGNA_IMGINFO *);
void sire_wrtsireinp(char [], SIRE_FLAGS, char [], SIRE_DIRINFO,
                     SIRE_FILTERINFO [], SIRE_IMGINFO, SIRE_SIGNA_IMGINFO,
                     double, char [], char []);
}

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class SiReInput : public Module {
    VoidStarOPort* oport;
    SiReData s;
    int rcvrIter;
    int slabIter;
    VoidStarHandle vsh2;
    GuiInt ShrinkFactor;
    GuiInt NPasses;
    GuiString PFileStr;
public:
    SiReInput(const clString& id);
    virtual ~SiReInput();
    virtual void execute();
};

extern "C" Module* make_SiReInput(const clString& id)
{
    return new SiReInput(id);
}

SiReInput::SiReInput(const clString& id)
: Module("SiReInput", id, Filter), rcvrIter(-1), slabIter(-1),
  PFileStr("PFileStr", id, this), ShrinkFactor("ShrinkFactor", id, this),
  NPasses("NPasses", id, this)
{
    oport = scinew VoidStarOPort(this, "SiReData", VoidStarIPort::Atomic);
    add_oport(oport);
    s.s.PassIdx=0;
}

SiReInput::~SiReInput()
{
}

void SiReInput::execute()
{
    cerr << "\n\nSIREINPUT\n\n\n";
    if (rcvrIter == -1) {

	/* ------------------------------------------------------------ */
	/* Get inputs and control parameters */
	/* ------------------------------------------------------------ */

	/* Initial message */
	printf("\n\n\nSiRe Reconstruction Program\n");
	printf("\nGetting inputs:\n");

	sire_getime (SIRE_YES, &s.s.TimeStamp, &s.s.RunTime, s.s.TimeStr);

	/* User inputs */
	printf("   Retrieving user inputs\n");
	sire_rdsireinp(SIRE_USERINPFILE, &s.s.Flag, s.s.FirstPFile, &s.s.Step,
		       s.s.FilterInfo, &s.s.SireImgInfo, &s.s.SignaImgInfo);

	sprintf(s.s.FirstPFile, "%s", (PFileStr.get())());
	s.s.ShrinkFactor = (int) pow(2., ShrinkFactor.get());
	s.s.NPasses = NPasses.get();
	cerr << "s.s.FirstPFile = "<<s.s.FirstPFile<<"\n";
	cerr << "s.s.ShrinkFactor = "<<s.s.ShrinkFactor<<"\n";
	cerr << "s.s.NPasses = "<<s.s.NPasses<<"\n";
	
	/* Get nonuser inputs */
	printf("   Getting non-user inputs\n");
	sire_getnonuserinp(s.s.FirstPFile, &s.s.PointSize, &s.s.NRead, 
			   &s.s.NPhase, &s.s.NRawRead, &s.s.NRawPhase, 
			   &s.s.NRawSlice, &s.s.NSlabPerPFile, &s.s.NPFile, 
			   &s.s.NSlBlank, &s.s.Rcvr0, &s.s.RcvrN, 
			   &s.s.NFinalImgSlice, &s.s.Flag, &s.s.SignaImgInfo);

	/* Calculate remaining control parameters */
	printf("   Calculating run-time parameters\n");
	sire_calconparam (s.s.FirstPFile, s.s.NRead, s.s.NPhase, s.s.NRawSlice,
			  s.s.NSlabPerPFile, s.s.NPFile, s.s.NSlBlank, 
			  s.s.Rcvr0, s.s.RcvrN, s.s.NFinalImgSlice, SIRE_YES, 
			  &s.s.Flag, &s.s.Step, &s.s.PFiles, &s.s.SlabIndices,
			  &s.s.RcvrSlabImgFiles, &s.s.Rcvr3DImgFiles, 
			  &s.s.FinalImgFile, &s.s.NSlicePerRcvr, 
			  &s.s.NRcvrPerSlab, &s.s.NSlab, &s.s.NRcnRead, 
			  &s.s.NRcnPhase, &s.s.NRcnSlicePerRcvr, 
			  &s.s.NRcnOverlap, &s.s.NRcnFinalImgSlice, 
			  &s.s.SignaImgInfo);

	/* Save run time parameters */
	printf("   Writing first sire.inp to sire.runtime.1\n");
	sire_wrtsireinp("sire.runtime.1", s.s.Flag, s.s.FirstPFile, s.s.Step, 
			s.s.FilterInfo, s.s.SireImgInfo, s.s.SignaImgInfo, 
			NULL, NULL, SIREVERSION);

	/* ------------------------------------------------------------- */
	/* Allocate working arrays */
	/* ------------------------------------------------------------- */
	if ((s.s.Flag.ReconRcvrSlabs == SIRE_YES) || 
	    (s.s.Flag.OverlapRcvrSlabs==SIRE_YES))
	    {
		printf("\nAllocating working arrays\n");
		sire_allocwork(s.s.NRead, s.s.NPhase, s.s.NSlicePerRcvr, 
			       s.s.NRcnRead, s.s.NRcnPhase, 
			       s.s.NRcnSlicePerRcvr, SIRE_YES, 
			       &s.s.ShiftRcvrRaw,&s.s.RcvrDCOffInd, 
			       &s.s.ZFIRcvrImg);
	    }
	
	/* ------------------------------------------------------------- */
	/* Loop through receiver data sets */
	/* ------------------------------------------------------------- */

	if (s.s.Flag.ReconRcvrSlabs == SIRE_YES)
	    printf("\nLooping through receivers\n");
	rcvrIter = s.s.Flag.StartRcvr;
	s.s.IRcvr = s.s.Flag.StartRcvr;
    }
    
    if (slabIter == -1) {
	if ((s.s.Flag.ReconRcvrSlabs==SIRE_YES) || 
	    (s.s.Flag.OverlapRcvrSlabs==SIRE_YES)) {
	    printf("\n   Receiver %d of %d ",s.s.IRcvr+1,s.s.NRcvrPerSlab);
	    printf("(Reconstruction %d of %d)\n", s.s.IRcvr-
		   s.s.Flag.StartRcvr+1, s.s.Flag.EndRcvr - 
		   s.s.Flag.StartRcvr + 1);
	}
	
	/* Set initial indices based on user inputs */
	s.s.FirstSlab=(s.s.IRcvr == s.s.Flag.StartRcvr) ? s.s.Flag.StartSlab:0;
	s.s.LastSlab=(s.s.IRcvr == s.s.Flag.EndRcvr) ? s.s.Flag.EndSlab : 
	    (s.s.NSlab-1);
    
	s.s.ISlab = s.s.FirstSlab;
	slabIter = s.s.FirstSlab;
    }

    char fname[100];
#if 0
    vsh2 = &s;
    sprintf(fname, "sire.rcvr%d.slab%d.pre.vs1", s.s.IRcvr, s.s.ISlab);
    TextPiostream stream(fname, Piostream::Write);
    Pio(stream, vsh2);
#endif    

    if (s.s.ISlab > s.s.FirstSlab)
	{
            printf("\n   Continuing receiver %d of %d ",s.s.IRcvr+1,
		   s.s.NRcvrPerSlab);
            printf("(Reconstruction %d of %d)\n", 
		   s.s.IRcvr - s.s.Flag.StartRcvr + 1,
                   s.s.Flag.EndRcvr - s.s.Flag.StartRcvr + 1);
	}
    
    /* Read in a slab for a receiver */
    printf("      Reading raw data from %s (Slab %d of %d)\n",
	   s.s.PFiles[s.s.ISlab], s.s.ISlab+1, s.s.NSlab);
//    if (s.s.ISlab > 0) while(1);
    sire_rdsipfilercvr(s.s.PFiles[s.s.ISlab], s.s.PointSize, s.s.Flag, 
		       s.s.NRead, s.s.NPhase, s.s.NSlicePerRcvr, 
		       s.s.NRawRead, s.s.NRawPhase, s.s.NRcvrPerSlab, 
		       s.s.NSlabPerPFile, SIRE_YES,
		       s.s.SlabIndices[s.s.ISlab], s.s.IRcvr, 
		       &s.s.RcvrRaw);
    
    //  we're gonna leave allocation on, since we want to crunch in parallel...
    // 	  AllocRcvr = SIRE_NO;   /* Turn off further allocation or RcvrRaw */

    // now send it out to the next module...

    SiReData *c = new SiReData(s);

    cerr << "c->s.NPasses="<<c->s.NPasses<<"\n";
    cerr << "c->s.PassIdx="<<c->s.PassIdx<<"\n";
    int p=((int)pow(2.,c->s.NPasses-c->s.PassIdx-1))*c->s.ShrinkFactor;
    cerr << "p="<<p<<"\n";
    c->s.NRead /= p;
    c->s.NPhase /= p;
    c->s.NRawRead /= p;
    c->s.NRawPhase /= p;
    c->s.NRcnPhase /= p;
    c->s.NRcnRead /= p;
    
    cerr << "nslice: " << c->s.NSlicePerRcvr << '\n';
    cerr << "rawread: " << c->s.NRawRead << '\n';
    cerr << "phase: " << c->s.NRawPhase << '\n';
    cerr << "read: " << c->s.NRead << '\n';
    cerr << "phase: " << c->s.NPhase << '\n';
    cerr << "total: " << c->s.NSlicePerRcvr*c->s.NRawRead*c->s.NRawPhase << '\n';
    SIRE_COMPLEX *sc = new 
	SIRE_COMPLEX[c->s.NSlicePerRcvr*c->s.NRead*c->s.NPhase];
    
    SIRE_COMPLEX *pp = sc;
    SIRE_COMPLEX *orig = c->s.RcvrRaw;
    for (int aa=0; aa<c->s.NSlicePerRcvr; aa++)
	for (int bb=0; bb<c->s.NRawPhase*p; bb++)
	    for (int cc=0; cc<c->s.NRawRead*p; cc++, orig++)
//		if (bb<c->s.NRawPhase*3/2 && cc<c->s.NRawRead*3/2 &&
//		    bb>=c->s.NRawPhase/2 && cc>=c->s.NRawRead/2) {
//		if ((bb & 1) == 0 && (cc & 1) == 0) {
		if (bb<c->s.NRawPhase && cc<c->s.NRawRead) {
		    *pp = *orig;
		    pp++;
		}
    c->s.RcvrRaw = sc;

		 
    VoidStarHandle vsh(c);

    sprintf(fname, "sire.rcvr%d.slab%d.pre.vs2", s.s.IRcvr, s.s.ISlab);
    TextPiostream stream2(fname, Piostream::Write);
    Pio(stream2, vsh);

    oport->send(vsh);

    if (s.s.ISlab < s.s.LastSlab && s.s.Flag.ReconRcvrSlabs == SIRE_YES) {
	s.s.ISlab++;
	slabIter++;
    } else {
	slabIter = -1;
	rcvrIter++;
	s.s.IRcvr++;
    }

    if (s.s.IRcvr <= s.s.Flag.EndRcvr) {
	cerr << "waiting to execute...\n";
	c->lockstepSem.down();
	cerr << "ok to execute!\n";
	want_to_execute();
    } else {
	rcvrIter = -1;
	if (s.s.PassIdx<(s.s.NPasses-1)) {
	    s.s.PassIdx++;
	    cerr << "waiting to execute...\n";
	    c->lockstepSem.down();
	    cerr << "ok to execute!\n";
	    want_to_execute();
	} else s.s.PassIdx=0;
    }
} // End namespace DaveW
}

