/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/

/*
************************************************************
marker.c

This file contains most of the marker information.

************************************************************
*/

/*LABEL marker.c */

#include "globals.h"
#include "marker.h"

/*PUBLIC*/

extern void ByteAlign();
extern void WriteVEHeader();
extern void WriteVSHeader();
extern int ReadVSHeader();
extern void WriteGOPHeader();
extern void ReadGOPHeader();
extern void WritePictureHeader();
extern void ReadPictureHeader();
extern void WriteMBSHeader();
extern void ReadMBSHeader();
extern void ReadHeaderTrailer();
extern int ReadHeaderHeader();
extern int ClearToHeader();
extern void WriteMBHeader();
extern int ReadMBHeader();

static void CodeMV();
static int DecodeMV();

/*PRIVATE*/

extern int MPEGIntraQ[];
extern int MPEGNonIntraQ[];

extern int XING;

extern int HorizontalSize;
extern int VerticalSize;
extern int Aprat;
extern int Prate;
extern int Brate;
extern int Rate;
extern int Bsize;
extern int BufferSize;
extern int ConstrainedParameterFlag;
extern int LoadIntraQuantizerMatrix;
extern int LoadNonIntraQuantizerMatrix;

extern int TimeCode;
extern int ClosedGOP;
extern int BrokenLink;

extern int TemporalReference;
extern int PType;
extern int BufferFullness;
extern int FullPelForward;
extern int ForwardIndex;
extern int FullPelBackward;
extern int BackwardIndex;
extern int PictureExtra;
extern int PictureExtraInfo;

extern int SQuant;
extern int SliceExtra;
extern int SliceExtraInfo;
extern int MBSRead;

extern int MType;
extern int MQuant;
extern int SVP;
extern int MVD1H;
extern int MVD1V;
extern int MVD2H;
extern int MVD2V;
extern int CBP;
extern int MBAIncrement;
extern int LastMBA;

extern int *QuantPMType[];
extern int *MFPMType[];
extern int *MBPMType[];
extern int *CBPPMType[];
extern int *IPMType[];

extern int LastMVD1V;
extern int LastMVD1H;
extern int LastMVD2V;
extern int LastMVD2H;

extern FSTORE *CFS;

extern int bit_set_mask[];
extern int extend_mask[];

extern DHUFF *MBADHuff;
extern DHUFF *MVDDHuff;
extern DHUFF *CBPDHuff;
extern DHUFF *IntraDHuff;
extern DHUFF *PredictedDHuff;
extern DHUFF *InterpolatedDHuff;
extern DHUFF **ModuloDHuff;

extern EHUFF *MBAEHuff;
extern EHUFF *MVDEHuff;
extern EHUFF *CBPEHuff;
extern EHUFF *MTypeEHuff;
extern EHUFF *IntraEHuff;
extern EHUFF *PredictedEHuff;
extern EHUFF *InterpolatedEHuff;
extern EHUFF **ModuloEHuff;

extern int NumberBitsCoded;

int TrailerValue=0;
int MacroAttributeBits=0;
int MotionVectorBits=0;

#define Zigzag(i) izigzag_index[i]
static int izigzag_index[] =
{0,  1,  8, 16,  9,  2,  3, 10,
17, 24, 32, 25, 18, 11,  4,  5,
12, 19, 26, 33, 40, 48, 41, 34,
27, 20, 13,  6,  7, 14, 21, 28,
35, 42, 49, 56, 57, 50, 43, 36,
29, 22, 15, 23, 30, 37, 44, 51,
58, 59, 52, 45, 38, 31, 39, 46,
53, 60, 61, 54, 47, 55, 62, 63};

/*START*/

/*BFUNC

ByteAlign() aligns the current stream to a byte-flush boundary. This
is used in the standard, with the assumption that input device is
byte-buffered.

EFUNC*/

void ByteAlign()
{
  BEGIN("ByteAlign");

  zeroflush();
}

/*BFUNC

WriteVEHeader() writes out a video sequence end marker.

EFUNC*/

void WriteVEHeader()
{
  BEGIN("WriteVEHeader");

  ByteAlign();
  mputv(MBSC_LENGTH,MBSC);
  mputv(VSEC_LENGTH,VSEC);
}


/*BFUNC

WriteVSHeader() writes out a video sequence start marker.  Note that
Brate and Bsize are defined automatically by this routine. 

EFUNC*/

void WriteVSHeader()
{
  BEGIN("WriteVSHeader");
  int i;

  ByteAlign();
  mputv(MBSC_LENGTH,MBSC);
  mputv(VSSC_LENGTH,VSSC);
  mputv(12,HorizontalSize);
  mputv(12,VerticalSize);
  mputv(4,Aprat);
  if (XING) mputv(4,0x09);
  else mputv(4,Prate);
  if (Rate) Brate = (Rate+399)/400;         /* Round upward */
  if (XING)
    mputv(18,0x3ffff);
  else
    mputv(18,((Brate!=0) ? Brate : 0x3ffff));  /* Var-bit rate if Brate=0 */
  mputb(1);             /* Reserved bit */
  Bsize=BufferSize/(16*1024);
  if (XING) mputv(10,16);
  else mputv(10,Bsize);

  mputb(ConstrainedParameterFlag);

  mputb(LoadIntraQuantizerMatrix);
  if(LoadIntraQuantizerMatrix)
    {
      for(i=0;i<64;i++)
	mputv(8,MPEGIntraQ[Zigzag(i)]);
    }

  mputb(LoadNonIntraQuantizerMatrix);
  if(LoadNonIntraQuantizerMatrix)
    {
      for(i=0;i<64;i++)
	mputv(8,MPEGNonIntraQ[Zigzag(i)]);
    }

  if (XING)
    {
      ByteAlign();
      mputv(32, 0x000001b2);
      mputv(8, 0x00);      /* number of frames? */
      mputv(8, 0x00);
      mputv(8, 0x02);      /* Other values 0x00e8 */
      mputv(8, 0xd0);
    }
}

/*BFUNC

ReadVSHeader() reads in the body of the video sequence start marker.

EFUNC*/

int ReadVSHeader()
{
  BEGIN("ReadVSHeader");
  int i;

  HorizontalSize = mgetv(12);
  VerticalSize = mgetv(12);
  Aprat = mgetv(4);
  if ((!Aprat)||(Aprat==0xf))
    {
      WHEREAMI();
      printf("Aspect ratio ill defined: %d.\n",Aprat);
    }
  Prate = mgetv(4);
  if ((!Prate) || (Prate > 8))
    {
      WHEREAMI();
      printf("Bad picture rate definition: %d\n",Prate);
      Prate = 6;
    }
  Brate = mgetv(18);
  if (!Brate)
    {
      WHEREAMI();
      printf("Illegal bit rate: %d.\n",Brate);
    }
  if (Brate == 0x3ffff)
    Rate=0;
  else
    Rate = Brate*400;
      
  (void) mgetb();             /* Reserved bit */

  Bsize = mgetv(10);
  BufferSize = Bsize*16*1024;

  ConstrainedParameterFlag = mgetb();

  LoadIntraQuantizerMatrix = mgetb();
  if(LoadIntraQuantizerMatrix)
    {
      for(i=0;i<64;i++)
	MPEGIntraQ[Zigzag(i)]= mgetv(8);
    }
  LoadNonIntraQuantizerMatrix = mgetb();
  if(LoadNonIntraQuantizerMatrix)
    {
      for(i=0;i<64;i++)
	MPEGNonIntraQ[Zigzag(i)] = mgetv(8);
    }
  return(0);
}

/*BFUNC

WriteGOPHeader() write a group of pictures header. Note that
the TimeCode variable needs to be defined correctly.

EFUNC*/

void WriteGOPHeader()
{
  BEGIN("WriteGOPHeader");

  ByteAlign();
  mputv(MBSC_LENGTH,MBSC);
  mputv(GOP_LENGTH,GOPSC);
  mputv(25,TimeCode);
  mputb(ClosedGOP);
  mputb(BrokenLink);
}

/*BFUNC

ReadGOPHeader() reads the body of the group of pictures marker.

EFUNC*/

void ReadGOPHeader()
{
  BEGIN("ReadGOPHeader");

  TimeCode = mgetv(25);
  ClosedGOP = mgetb();
  BrokenLink = mgetb();
}

/*BFUNC

WritePictureHeader() writes the header of picture out to the stream.
One of these is necessary before every frame is transmitted.

EFUNC*/

void WritePictureHeader()
{
  BEGIN("WritePictureHeader");
  static int frame=1;

  ByteAlign();
  mputv(MBSC_LENGTH,MBSC);
  mputv(PSC_LENGTH,PSC);
  if (XING) mputv(10,frame++);
  else mputv(10,TemporalReference);
  mputv(3,PType);

  if (XING) mputv(16,0xffff);
  else
    {
      if (BufferFullness<0)
	{
	  WHEREAMI();
	  printf("Virtual decoder buffer fullness less than zero.\n");
	  mputv(16,0);
	}
      else if (BufferFullness > 65535)
	{
	  WHEREAMI();
	  printf("Virtual decoder buffer fullness > 65536/90000 second.\n");
	  mputv(16,0xffff);
	}
      else
	mputv(16,BufferFullness);
    }


  if ((PType == P_PREDICTED) || (PType == P_INTERPOLATED))
    {
      mputb(FullPelForward);
      mputv(3,ForwardIndex);
    }
  if (PType == P_INTERPOLATED)
    {
      mputb(FullPelBackward);
      mputv(3,BackwardIndex);
    }
  if (XING)
    {
      mputb(1);
      mputv(8,0xff);
      mputb(1);
      mputv(8,0xfe);
      ByteAlign();           /* The bytealign seems to work well here */
      mputv(32, 0x000001b2);
      mputv(8, 0xff);
      mputv(8, 0xff);
    }
  else
    {
      mputb(PictureExtra);
      if (PictureExtra)
	{
	  mputv(8,PictureExtraInfo);
	  mputb(0);
	}
    }
}

/*BFUNC

ReadPictureHeader() reads the header off of the stream. It assumes
that the first PSC has already been read in. (Necessary to tell the
difference between a new picture and another GOP.)

EFUNC*/

void ReadPictureHeader()
{
  BEGIN("ReadPictureHeader");

  TemporalReference = mgetv(10);
  PType = mgetv(3);
  if (!PType)
    {
      WHEREAMI();
      printf("Illegal PType received.\n");
    }
  BufferFullness = mgetv(16);

  if ((PType == P_PREDICTED) || (PType == P_INTERPOLATED))
    {
      FullPelForward = mgetb();
      ForwardIndex = mgetv(3);
    }
  if (PType == P_INTERPOLATED)
    {
      FullPelBackward = mgetb();
      BackwardIndex = mgetv(3);
    }
  PictureExtra=0;
  while(mgetb())
    {
      PictureExtraInfo = mgetv(8);
      PictureExtra = 1;
    }
}

/*BFUNC

WriteMBSHeader() writes a macroblock slice header out to the stream.

EFUNC*/

void WriteMBSHeader()
{
  BEGIN("WriteMBSHeader");

  ByteAlign();
  mputv(MBSC_LENGTH,MBSC);
  /* printf("Wrote: MBS-> SVP: %d\n",SVP);*/
  mputv(8,SVP);
  mputv(5,SQuant);
  if (SliceExtra)
    {
      mputb(1);
      mputv(8,SliceExtraInfo);
    }
  mputb(0);
}

/*BFUNC

ReadMBSHeader() reads the slice information off of the stream. We
assume that the first bits have been read in by ReadHeaderHeader... or
some such routine.

EFUNC*/

void ReadMBSHeader()
{
  BEGIN("ReadMBSHeader");

  SQuant = mgetv(5);
  for(SliceExtra=0;mgetb();)
    {
      SliceExtraInfo = mgetv(8);
      SliceExtra = 1;
    }
}

/*BFUNC

ReadHeaderTrailer(GOP,)
     ) reads the trailer of the GOP; reads the trailer of the GOP, PSC or MBSC code. It
is used to determine whether it is just a new group of frames, new
picture, or new slice.

EFUNC*/

void ReadHeaderTrailer()
{
  BEGIN("ReadHeaderTrailer");

  while(1)
    {
      TrailerValue = mgetv(8);
      if (!TrailerValue)              /* Start of picture */
	{
	  MBSRead = -1;
	  break;
	}
      else if (TrailerValue==GOPSC)   /* Start of group of picture */
	{
	  MBSRead = -2;
	  break;
	}
      else if (TrailerValue==VSEC)   /* Start of group of picture */
	{
	  MBSRead = -3;
	  break;
	}
      else if (TrailerValue==VSSC)
	{
	  MBSRead = -4;
	  break;
	}
      else if ((TrailerValue > 0) && (TrailerValue < 0xb0)) /* Slice vp */
	{
	  MBSRead = TrailerValue-1;
	  SVP = TrailerValue;
	  /* printf("Read SVP %d\n",SVP);*/
	  break;
	}
      else if (TrailerValue == UDSC)
	{
	  printf("User data code found.\n");
	  ClearToHeader();
	}
      else if (TrailerValue == EXSC)
	{
	  printf("Extension data code found.\n");
	  ClearToHeader();
	}
      else
	break;
    }
}

/*BFUNC

ReadHeaderHeader() reads the common structure header series of bytes
(and alignment) off of the stream.  This is a precursor to the GOP
read or the PSC read. It returns -1 on error.

EFUNC*/

int ReadHeaderHeader()
{
  BEGIN("ReadHeaderHeader");
  int input;

  readalign();                /* Might want to check if all 0's */
  if ((input = mgetv(MBSC_LENGTH)) != MBSC)
    {
      while(!input)
	{
	  if ((input = mgetv(8)) == MBSC)  /* Get next byte */
	    return(0);
	  else if (input && (seof()))
	    {
	      WHEREAMI();
	      printf("End of file.\n");
	    }
	}
      WHEREAMI();
      printf("Bad input read: %d\n",input);
      return(-1);
    }
  return(0);
}

/*BFUNC

ClearToHeader() reads the header header off of the stream. This is
a precursor to the GOP read or the PSC read. It returns -1 on error.

EFUNC*/

int ClearToHeader()
{
  BEGIN("ReadHeaderHeader");
  int input;

  readalign();                /* Might want to check if all 0's */
  if ((input = mgetv(MBSC_LENGTH)) != MBSC)
    {
      do
	{
	  if (seof())
	    {
	      WHEREAMI();
	      printf("Illegal termination.\n");
	      exit(-1);
	    }
	  input = input & 0xffff;               /* Shift off by 8 */
	  input = (input << 8) | mgetv(8);
	}
      while (input != MBSC);  /* Get marker */
    }
  return(0);
}

/*BFUNC

WriteStuff() writes a MB stuff code. 

EFUNC*/

WriteStuff()
{
  BEGIN("WriteStuff");
  mputv(11,0xf);
}

/*BFUNC

WriteMBHeader() writes the macroblock header information out to the stream.

EFUNC*/

void WriteMBHeader()
{
  BEGIN("WriteMBHeader");
  int i,TempH,TempV,Start,retval;

  /* printf("[MBAInc: %d]",MBAIncrement); */
  /* printf("[Writing: %d]\n",MType); */
#ifdef MV_DEBUG
#endif
  Start=swtell();
  if (MBAIncrement > 33)
    {
      for(i=0;i<((MBAIncrement-1)/33);i++)
	{
	  if (!Encode(35,MBAEHuff)) /* Escape code */
	    {
	      WHEREAMI();
	      printf("Attempting to write an empty Huffman code (35).\n");
	      exit(ERROR_HUFFMAN_ENCODE);
	    }
	}
      if (!Encode(((MBAIncrement-1)%33)+1,MBAEHuff))
	{
	  WHEREAMI();
	  printf("Attempting to write an empty Huffman code (%d).\n",
		 (MBAIncrement-1)%33);
	  exit(ERROR_HUFFMAN_ENCODE);
	}
    }
  else
    {
      if (!Encode(MBAIncrement,MBAEHuff))
	{
	  WHEREAMI();
	  printf("Attempting to write an empty Huffman code (%d).\n",
		 MBAIncrement);
	  exit(ERROR_HUFFMAN_ENCODE);
	}
    }
  switch(PType)
    {
    case P_INTRA:
      retval=Encode(MType,IntraEHuff);
      break;
    case P_PREDICTED:
      retval=Encode(MType,PredictedEHuff);
      break;
    case P_INTERPOLATED:
      retval=Encode(MType,InterpolatedEHuff);
      break;
    case P_DCINTRA:
      mputb(1);           /* only one type for DC Intra */
      retval=1;
      break;
    default:
      WHEREAMI();
      printf("Bad picture type: %d\n",PType);
      break;
    }
  if (!retval)
    {
      WHEREAMI();
      printf("Attempting to write an empty Huffman code.\n");
      exit(ERROR_HUFFMAN_ENCODE);
    }
  NumberBitsCoded=0;
  if (QuantPMType[PType][MType]) mputv(5,MQuant);
  if (MFPMType[PType][MType])
    {
      TempH = MVD1H - LastMVD1H;
      TempV = MVD1V - LastMVD1V;

#ifdef MV_DEBUG
      printf("1st FI: %d  Type: %d  Actual: H %d  V %d  Coding: H %d  V %d\n",
	     ForwardIndex,MType,MVD1H,MVD1V,TempH,TempV);
#endif
      if (FullPelForward)
	{
	  CodeMV(TempH/2,ForwardIndex);
	  CodeMV(TempV/2,ForwardIndex);
	}
      else
	{
	  CodeMV(TempH,ForwardIndex);
	  CodeMV(TempV,ForwardIndex);
	}
      LastMVD1V = MVD1V;
      LastMVD1H = MVD1H;
    }
  if (MBPMType[PType][MType])
    {
      TempH = MVD2H - LastMVD2H;
      TempV = MVD2V - LastMVD2V;

#ifdef MV_DEBUG
      printf("2nd BI: %d  Type: %d  Actual: H %d  V %d  Coding: H %d  V %d\n",
             BackwardIndex,MType,MVD2H,MVD2V,TempH,TempV);
#endif
      if (FullPelBackward)
	{
	  CodeMV(TempH/2,BackwardIndex);
	  CodeMV(TempV/2,BackwardIndex);
	}
      else
	{
	  CodeMV(TempH,BackwardIndex);
	  CodeMV(TempV,BackwardIndex);
	}
      LastMVD2V = MVD2V;
      LastMVD2H = MVD2H;
    }
  MotionVectorBits+=NumberBitsCoded;

  if (CBPPMType[PType][MType])
    {
      /* printf("CBP: %d\n",CBP); */
      if (!Encode(CBP,CBPEHuff))
	{
	  WHEREAMI();
	  printf("CBP write error: PType: %d  MType: %d CBP: %d.\n",
		 PType,MType,CBP);
	  exit(-1);
	}
    }
  MacroAttributeBits+=(swtell()-Start);
}

static void CodeMV(d,fd)
     int d;
     int fd;
{
  BEGIN("CodeMV");
  int v,limit;
  int fcode, motion_r;

  if (!d)
    {
      /* printf("Encoding zero\n"); */
      if (!Encode(0,MVDEHuff))
	{
	  WHEREAMI();
	  printf("Cannot encode motion vectors.\n");
	}
      return;
    }
  limit = 1 << (fd+3); /* find limit 16*(1<<(fd-1)) */
  /* Calculate modulo factors */
  if (d < -limit) /* Do clipping */
    {
      d += (limit << 1);
      if (d <= 0)
	{
	  WHEREAMI();
	  printf("Motion vector out of bounds: (residual) %d\n",d);
	}
    }
  else if (d>=limit)
    {
      d -= (limit << 1);
      if (d >= 0)
	{
	  WHEREAMI();
	  printf("Motion vector out of bounds: (residual) %d\n",d);
	}
    }
  fd--;
  if (fd)
    {
      if (!d)
	{
	  if (!Encode(d,MVDEHuff))
	    {
	      WHEREAMI();
	      printf("Cannot encode zero motion vector.\n");
	    }
	  return;
	}
      if (d > 0)
	{
	  d--;                           /* Dead band at zero */
	  fcode = (d>>fd)+1;             /* Quantize, move up */
	  motion_r = d&((1<<fd)-1);      /* use lowest fd bits */
	}                                /* May not need mask for mputv */
      else
	{
	  fcode = d>>fd;                 /* negative, dead band auto. */
	  motion_r = (-1^d)&((1<<fd)-1); /* lowest fd bits of (abs(d)-1)*/
	}                                /* May not need mask for mputv */
      if (fcode < 0) v = 33+fcode;       /* We have positive array index */
      else v = fcode;
      if (!Encode(v,MVDEHuff))
	{
	  WHEREAMI();
	  printf("Cannot encode motion vectors.\n");
	}
      mputv(fd,motion_r);
    }
  else
    {
      if (d < 0) v = 33+d;         /* Compensate for positive array index */
      else v = d;
      if (!Encode(v,MVDEHuff))
	{
	  WHEREAMI();
	  printf("Cannot encode motion vectors.\n");
	}
    }
}

static int DecodeMV(fd,oldvect)
     int fd;
     int oldvect;
{
  BEGIN("DecodeMV");
  int r,v,limit;
  
  /* fd is the frame displacement */
  /* limit is the maximum displacement range we can have = 16*(1<<(fd-1)) */
  limit = 1 << (fd+3); 
  v = Decode(MVDDHuff);
  if (v)
    {
      if (v > 16) {v = v-33;}       /* our codes are positive, negative vals */
                                    /* are greater than 16. */
      fd--;                         /*Find out number of modulo bits=fd */
      if (fd)
	{
	  r = mgetv(fd);            /* modulo lower bits */
	  if (v > 0) 
	    v = (((v-1)<<fd)|r)+1;  /* just "or" and add 1 for dead band. */
	  else
	    v = ((v+1)<<fd)+(-1^r); /* Needs mask to do an "or".*/
	}
      if (v==limit)
	{
	  WHEREAMI();
	  printf("Warning: motion vector at positive limit.\n");
	}
    }
  v += oldvect;                      /* DPCM */
  if (v < -limit)
    v += (limit << 1);
  else if (v >= limit)
    v -= (limit << 1);
  if (v==limit)
    {
      WHEREAMI();
      printf("Apparently illegal reference: (MV %d) (LastMV %d).\n",v,oldvect);
    }
  return(v);
}

/*BFUNC

ReadMBHeader() reads the macroblock header information from the stream.

EFUNC*/

int ReadMBHeader()
{
  BEGIN("ReadMBHeader");
  int Readin;

  for(MBAIncrement=0;;)
    {
      do
	{
	  Readin = Decode(MBADHuff);
	}
      while(Readin == 34);  /* Get rid of stuff bits */
      if (Readin <34)
	{
	  MBAIncrement += Readin;
	  break;
	}
      else if (Readin == 36)
	{
	  while(!mgetb());
	  return(-1); /* Start of Picture Headers */
	}
      else if (Readin == 35)
	MBAIncrement += 33;
      else
	{
	  WHEREAMI();
	  printf("Bad MBA Read: %d \n",Readin);
	  break;
	}
    }
  /* printf("[MBAInc: %d]\n",MBAIncrement); */
  switch(PType)
    {
    case P_INTRA:
      MType = Decode(IntraDHuff);
      break;
    case P_PREDICTED:
      if (MBAIncrement > 1) MVD1H=MVD1V=0; /* Erase MV for skipped mbs */
      MType = Decode(PredictedDHuff);
      break;
    case P_INTERPOLATED:
      MType = Decode(InterpolatedDHuff);
      break;
    case P_DCINTRA:
      if (!mgetb())
	{
	  WHEREAMI();
	  printf("Expected one bit for DC Intra, 0 read.\n");
	}
      break;
    default:
      WHEREAMI();
      printf("Bad picture type.\n");
      break;
    }
#ifdef MV_DEBUG
  printf("[Reading: %d]",MType);
#endif
  if (QuantPMType[PType][MType]) MQuant=mgetv(5);
  if (MFPMType[PType][MType])
    {
      if (FullPelForward)
	{
	  MVD1H = DecodeMV(ForwardIndex,MVD1H/2)<<1;
	  MVD1V = DecodeMV(ForwardIndex,MVD1V/2)<<1;
	}
      else
	{
	  MVD1H = DecodeMV(ForwardIndex,MVD1H);
	  MVD1V = DecodeMV(ForwardIndex,MVD1V);
	}
#ifdef MV_DEBUG
      printf("1st FI: %d  Type: %d  Actual: H %d  V %d\n",
	     ForwardIndex,MType,MVD1H,MVD1V);
#endif
    }
  else if (PType==P_PREDICTED)
    MVD1H=MVD1V=0;

  if (MBPMType[PType][MType])
    {
      if (FullPelBackward)
	{
	  MVD2H = DecodeMV(BackwardIndex,MVD2H/2)<<1;
	  MVD2V = DecodeMV(BackwardIndex,MVD2V/2)<<1;
	}
      else
	{
	  MVD2H = DecodeMV(BackwardIndex,MVD2H);
	  MVD2V = DecodeMV(BackwardIndex,MVD2V);
	}
#ifdef MV_DEBUG
      printf("2nd BI: %d  Type: %d   Actual: H %d  V %d.\n",
	     BackwardIndex,MType,MVD2H,MVD2V);
#endif
    }
  if (CBPPMType[PType][MType]) CBP = Decode(CBPDHuff);
  else if (IPMType[PType][MType]) CBP=0x3f;
  else CBP=0;

  /* printf("CBP: %d\n",CBP);*/
  return(0);
}

/*END*/

