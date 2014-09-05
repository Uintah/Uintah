//static char *id="@(#) $Id$";

/*
 *  SiReOutput.cc:  Output the SiRe image-space data
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <DaveW/Datatypes/SiRe/SiRe.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VoidStarPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Datatypes/VoidStar.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Tester/RigorousTest.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

/* Prototypes */
extern "C" {
void sire_calconparam (char [], int, int, int, int, int, int, int, int,
                       int, int,SIRE_FLAGS *,SIRE_DIRINFO *, char ***, int **,
                       char ****, char ***, char **, int *, int *, int *,
                       int *,int *, int *, int *, int *, SIRE_SIGNA_IMGINFO *);
void sire_deletefiles(int, char**);
void sire_getime(int, time_t *, double *, char []);
void sire_inplace_overlapslabimg(char *,int, int, int, int, int, int,
                                 short *, int, short **);
void sire_makesigna(char [], int, int, int, SIRE_FLAGS, SIRE_DIRINFO,
                    SIRE_SIGNA_IMGINFO *);
void sire_meshslabimg(char [], char **, int, int, int, int);
void sire_overlapslabimg(char *,char **,int,int,int, int, int, int, short *);
void sire_overlaprcvrimg(char *, int, char **, int, int, int);
void sire_wrtsireinp(char [], SIRE_FLAGS, char [], SIRE_DIRINFO,
                     SIRE_FILTERINFO [], SIRE_IMGINFO, SIRE_SIGNA_IMGINFO,
                     double, char [], char []);
void sire_wrtrcvrimg(char *, int, int, int, short int *);
}

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using namespace SCICore::Math;

class SiReOutput : public Module {
    VoidStarIPort* iport;
    ScalarFieldOPort* oport;
    ScalarFieldRGuchar *sf;
//    ScalarFieldRG *sf;
    ScalarFieldHandle sfh;
public:
    SiReOutput(const clString& id);
    virtual ~SiReOutput();
    virtual void execute();
};

extern "C" Module* make_SiReOutput(const clString& id)
{
    return new SiReOutput(id);
}

SiReOutput::SiReOutput(const clString& id)
: Module("SiReOutput", id, Filter)
{
    iport = scinew VoidStarIPort(this, "SiReData", VoidStarIPort::Atomic);
    add_iport(iport);
    oport = scinew ScalarFieldOPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_oport(oport);
}

SiReOutput::~SiReOutput()
{
}

int nextPowerOfTwo(int a) {
    a-=1;
    int i=1;
    while (a>0) {a = a>>1; i*=2;}
    return i;
}

uchar cap(short i) {
    if (i<0) i=0;
    if (i>255) i=255;
    return (uchar)i;
}

void SiReOutput::execute()
{
   VoidStarHandle vsh;
   if (!iport->get(vsh)) return;
   SiReData *s;
   if (!vsh.get_rep() || !(s=dynamic_cast<SiReData* >(vsh.get_rep()))) return;
    
   cerr << "\n\nSIREOUTPUT\n\n\n";

   s->lockstepSem.up();
   
   // add this image into the SCIRun volume
   //   if this is the first one, allocate the volume
   if (s->s.ISlab == s->s.Flag.StartSlab && s->s.IRcvr == s->s.Flag.StartRcvr){
       int x1 = s->s.NRcnFinalImgSlice;
       int y1 = s->s.NRcnPhase;
       int z1 = s->s.NRcnRead;
       int x2 = nextPowerOfTwo(x1);
       int y2 = nextPowerOfTwo(y1);
       int z2 = nextPowerOfTwo(z1);

       cerr << x1 << " -> "<<x2<<"\n";
       cerr << y1 << " -> "<<y2<<"\n";
       cerr << z1 << " -> "<<z2<<"\n";

       // set the tcl variable, so Pete knows to build the texture
       // set the tcl variables for x1, y1, z1

       TCL::execute("set vol-mode 0");
       clString s1(clString("set vol-nx ")+to_string(z1));
       TCL::execute(s1);
       s1 = clString("set vol-ny ")+to_string(y1);
       TCL::execute(s1);
       s1 = clString("set vol-nz ")+to_string(x1);
       TCL::execute(s1);

       int scale=((int)pow(2., s->s.NPasses-s->s.PassIdx-1))*s->s.ShrinkFactor;

       ScalarFieldRGuchar* sft = new ScalarFieldRGuchar();

       sft->set_bounds(Point(0,0,0), Point(x1*1.5,y1*scale,z1*scale));

       sft->resize(x2, y2, z2);

       sft->grid.initialize(0);
       if (s->s.PassIdx != 0) {
	   short *ptr = &(s->s.ZFIRcvrImg[0]);
	   for (int ii=0; ii<s->s.NRcnFinalImgSlice; ii++)
	       for (int jj=0; jj<s->s.NRcnPhase; jj++)
		   for (int kk=0; kk<s->s.NRcnRead; kk++, ptr++)
		       sft->grid(ii,jj,kk) = sf->grid(ii,jj/2,kk/2);
       }
       sfh=sf=sft;
   } else {
       TCL::execute("set vol-mode 1");
       // set the tcl variable, so Pete knows to reuse the memory from before
   }

   //   add in contribution from this slab
   short *ptr;
   int startIdx = 0;

   ptr = &(s->s.ZFIRcvrImg[startIdx]);
   int firstSlice = 0;
   if (s->s.ISlab != 0) 
       firstSlice = (s->s.ISlab * (s->s.NRcnSlicePerRcvr - s->s.NRcnOverlap));
   int lastSlice = firstSlice + s->s.NRcnSlicePerRcvr;
   cerr << "NNRcnSlicePerRcvr="<<s->s.NRcnSlicePerRcvr<<"  s->s.NRcnOverlap="<<s->s.NRcnOverlap<<"  firstSlice="<<firstSlice<<"  lastSlice="<<lastSlice<<"\n";
   int clear=0;
   for (int ii=firstSlice; ii<lastSlice; ii++) {
       if (s->s.IRcvr == 0 && 
	   (s->s.ISlab == 0 || ii>=s->s.ISlab*s->s.NRcnSlicePerRcvr)) clear=1;
       for (int	jj=0; jj<s->s.NRcnPhase; jj++)
	   for (int kk=0; kk<s->s.NRcnRead; kk++, ptr++)
	       if (clear)
		   sf->grid(ii,jj,kk) = cap(*ptr);
	       else	
		   sf->grid(ii,jj,kk) = Max(sf->grid(ii,jj,kk), cap(*ptr));
       cerr << "("<<ii<<")-"<<(int)sf->grid(ii,s->s.NRcnPhase/2,s->s.NRcnRead/2)<<" ";
   }
   //   send through output port

   cerr << "\nABOUT TO SEND FROM OUTPUT!\n";
   oport->send(sfh);
   cerr << "DONE SENDING FROM OUTPUT!\n";

#if 0
   /* ------------------------------------------------------------ */
   /* Store receiver to disk, blanking boundary slices */
   /* ------------------------------------------------------------ */
   if (s->s.Flag.InPlaceOverlap == SIRE_NO) {
       printf("      Saving receiver ZFI slab data to %s\n",
	      s->s.RcvrSlabImgFiles[s->s.IRcvr][s->s.ISlab]);
       sire_wrtrcvrimg(s->s.RcvrSlabImgFiles[s->s.IRcvr][s->s.ISlab], 
		       s->s.NRcnRead, s->s.NRcnPhase, s->s.NRcnSlicePerRcvr, 
		       s->s.ZFIRcvrImg);
   } else {
       printf("      Updating ZFI receiver %s\n", 
	      s->s.Rcvr3DImgFiles[s->s.IRcvr]);
       sire_inplace_overlapslabimg(s->s.Rcvr3DImgFiles[s->s.IRcvr],
				   s->s.NSlab, s->s.NRcnRead, s->s.NRcnPhase,
				   s->s.NRcnSlicePerRcvr, s->s.NRcnOverlap,
				   s->s.Flag.AutoCor, s->s.ZFIRcvrImg,
				   s->s.ISlab, &s->s.OverlapImg);
   }

   // if this is the last slab of this receiver

   
   char fname[100];
   sprintf(fname, "sire.rcvr%d.slab%d.post.vs1", s->s.IRcvr, s->s.ISlab);
   TextPiostream stream(fname, Piostream::Write);
   Pio(stream, vsh);

   if (s->s.ISlab == s->s.LastSlab) {

       /* Overlap receiver slabs */
       if ((s->s.Flag.OverlapRcvrSlabs == SIRE_YES) &&
	   (s->s.Flag.InPlaceOverlap == SIRE_NO)) {
	   printf("      Overlapping slabs and saving in %s\n",
		  s->s.Rcvr3DImgFiles[s->s.IRcvr]);
	   if (s->s.Flag.AcqType != SIRE_FASTCARDACQ)
	       sire_overlapslabimg(s->s.Rcvr3DImgFiles[s->s.IRcvr], 
				   s->s.RcvrSlabImgFiles[s->s.IRcvr],
				   s->s.NSlab, s->s.NRcnRead, s->s.NRcnPhase, 
				   s->s.NRcnSlicePerRcvr, s->s.NRcnOverlap, 
				   s->s.Flag.AutoCor, s->s.ZFIRcvrImg);
	   else
	       sire_meshslabimg(s->s.Rcvr3DImgFiles[s->s.IRcvr], 
				s->s.RcvrSlabImgFiles[s->s.IRcvr],
				s->s.NSlab, s->s.NRcnRead, s->s.NRcnPhase, 
				s->s.NRcnSlicePerRcvr);
       }

       /* Delete receiver slab files if not needed */
       if ((s->s.Flag.SaveRcvrSlabs == SIRE_NO) && 
	   (s->s.Flag.InPlaceOverlap==SIRE_NO) &&
	   (s->s.Flag.ReconRcvrSlabs == SIRE_YES)) {
	   printf("      Deleteing temporary slab files\n");
	   sire_deletefiles(s->s.NSlab, s->s.RcvrSlabImgFiles[s->s.IRcvr]);
       }
       
       // if this is that last slab of the last receiver...

       if (s->s.IRcvr == s->s.Flag.EndRcvr) {

	   /* Delete receiver slab files not hit by previous delete */
	   if ((s->s.Flag.SaveRcvrSlabs == SIRE_NO) && 
	       (s->s.Flag.InPlaceOverlap==SIRE_NO) &&
	       (s->s.Flag.ReconRcvrSlabs == SIRE_YES)) {
	       printf("      Deleteing temporary slab files\n");
	       sire_deletefiles(s->s.NSlab,s->s.RcvrSlabImgFiles[s->s.IRcvr]);
	   }

	   /* ------------------------------------------------------------ */
	   /* Compile the receivers into a single file */
	   /* ------------------------------------------------------------ */
	   if (s->s.Flag.MakeZFImg == SIRE_YES) {
	       printf("\nCreating single 3D compilation of receivers: %s\n",
		  s->s.FinalImgFile);
	       sire_overlaprcvrimg(s->s.FinalImgFile, s->s.NRcvrPerSlab, 
				   s->s.Rcvr3DImgFiles, s->s.NRcnRead,
				   s->s.NRcnPhase, s->s.NRcnFinalImgSlice);
	       
	       /* Delete receiver files if not needed */
	       if (s->s.Flag.SaveRcvrs == SIRE_NO) {
		   printf("\nDeleting temporary receiver data files\n");
		   sire_deletefiles(s->s.NRcvrPerSlab, s->s.Rcvr3DImgFiles);
	       }
	   }

	   /* ------------------------------------------------------------ */
	   /* Create signa images if directed */
	   /* ------------------------------------------------------------ */
	   if (s->s.Flag.MakeSignaImg == SIRE_YES) {
	       printf("\nCreating Signa images\n\n");
	       sire_makesigna(s->s.FinalImgFile, s->s.NRcnRead, s->s.NRcnPhase,
			      s->s.NRcnFinalImgSlice,
			      s->s.Flag, s->s.Step, &s->s.SignaImgInfo);

	       /* Delete ZFI file if not needed */
	       if (s->s.Flag.SaveZFImg == SIRE_NO) {
		   printf("\nDeleteing 3D compilation data file\n");
		   sire_deletefiles(1, &s->s.FinalImgFile);
	       }
	   }

	   /* ------------------------------------------------------------ */
	   /* Cleanup what is left */
	   /* ------------------------------------------------------------ */

	   /* Free up filenames and indices */
	   sire_calconparam (s->s.FirstPFile, s->s.NRead, s->s.NPhase, 
			     s->s.NRawSlice, s->s.NSlabPerPFile, s->s.NPFile,
			     s->s.NSlBlank, s->s.Rcvr0, s->s.RcvrN, 
			     s->s.NFinalImgSlice, -SIRE_YES, &s->s.Flag, 
			     &s->s.Step, &s->s.PFiles, &s->s.SlabIndices,
			     &s->s.RcvrSlabImgFiles, &s->s.Rcvr3DImgFiles, 
			     &s->s.FinalImgFile, &s->s.NSlicePerRcvr, 
			     &s->s.NRcvrPerSlab, &s->s.NSlab, &s->s.NRcnRead, 
			     &s->s.NRcnPhase, &s->s.NRcnSlicePerRcvr, 
			     &s->s.NRcnOverlap, &s->s.NRcnFinalImgSlice, 
			     &s->s.SignaImgInfo);

	   /* Complete the time stamp */
	   sire_getime(SIRE_NO, &s->s.TimeStamp, &s->s.RunTime, s->s.TimeStr);
       
	   /* Create the output sire.inp file */
	   printf("   Creating final sire.inp.date file\n");
	   sire_wrtsireinp(SIRE_USERINPFILE, s->s.Flag, s->s.FirstPFile, 
			   s->s.Step, s->s.FilterInfo, s->s.SireImgInfo, 
			   s->s.SignaImgInfo, s->s.RunTime, s->s.TimeStr, 
			   SIREVERSION);
	   
	   printf("\nSiRe Reconstruction program finished.\n");
       }
   }
#endif
}
} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.6  2000/03/17 09:26:01  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  1999/10/07 02:06:40  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/09/08 02:26:31  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/30 20:19:21  sparker
// Updates to compile with -LANG:std on SGI
// Other linux/irix porting oscillations
//
// Revision 1.2  1999/08/25 03:47:42  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/24 06:23:06  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:17  dmw
// Added and updated DaveW Datatypes/Modules
//
// Revision 1.1.1.1  1999/04/24 23:12:16  dav
// Import sources
//
//
