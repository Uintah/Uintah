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
prototypes.h

This file contains the functional prototypes for type checking, if
required.

************************************************************
*/

/* mpeg.c */


int main();
extern void MpegEncodeSequence();
extern void MpegDecodeSequence();
extern void MpegEncodeIPBDFrame();
extern void MpegDecodeIPBDFrame();
extern void PrintImage();
extern void PrintFrame();
extern void MakeImage();
extern void MakeFrame();
extern void MakeFGroup();
extern void LoadFGroup();
extern void SCILoadFGroup(unsigned char* imageY,
			  unsigned char* imageU,
			  unsigned char* imageV,
			  unsigned char* oldY);
extern void MakeFStore();
extern void MakeStat();
extern void SetCCITT();
extern void CreateFrameSizes();
extern void Help();
extern void MakeFileNames();
extern void VerifyFiles();
extern int Integer2TimeCode();
extern int TimeCode2Integer();

/* codec.c */

extern void EncodeAC();
extern void CBPEncodeAC();
extern void DecodeAC();
extern void CBPDecodeAC();
extern int DecodeDC();
extern void EncodeDC();

/* huffman.c */

extern void inithuff();
extern int Encode();
extern int Decode();
extern void PrintDhuff();
extern void PrintEhuff();
extern void PrintTable();

/* io.c */

extern void MakeIob();
extern void SuperSubCompensate();
extern void SubCompensate();
extern void AddCompensate();
extern void Sub2Compensate();
extern void Add2Compensate();
extern void MakeMask();
extern void ClearFS();
extern void InitFS();
extern void ReadFS();
extern void SCIReadFS(unsigned char* imageY,
		      unsigned char* imageU,
		      unsigned char* imageV);
extern void InstallIob();
extern void InstallFSIob();
extern void WriteFS();
extern void MoveTo();
extern int Bpos();
extern void ReadBlock();
extern void WriteBlock();
extern void PrintIob();

/* chendct.c */

extern void ChenDct();
extern void ChenIDct();

/* lexer.c */

extern void initparser();
extern void parser();

/* marker.c */

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

/* me.c */

extern void initme();
extern void HPFastBME();
extern void BruteMotionEstimation();
extern void InterpolativeBME();

/* mem.c */


extern void CopyMem();
extern ClearMem();
extern SetMem();
extern MEM *MakeMem();
extern void FreeMem();
extern MEM *LoadMem();
extern MEM *LoadPartialMem();
extern MEM *SaveMem();
extern MEM *SavePartialMem();

/* stat.c */

extern void Statistics();

/* stream.c */

extern void readalign();
extern void mropen();
extern void mrclose();
extern void mwopen();
extern void mwclose();
extern void zeroflush();
extern int mgetb();
extern void mputv();
extern int mgetv();
extern long mwtell();
extern long mrtell();
extern void mwseek();
extern void mrseek();
extern int seof();

/* transform.c */

extern void ReferenceDct();
extern void ReferenceIDct();
extern void TransposeMatrix();
extern void MPEGIntraQuantize();
extern void MPEGIntraIQuantize();
extern void MPEGNonIntraQuantize();
extern void MPEGNonIntraIQuantize();
extern void BoundIntegerMatrix();
extern void BoundQuantizeMatrix();
extern void BoundIQuantizeMatrix();
extern void ZigzagMatrix();
extern void IZigzagMatrix();
extern void PrintMatrix();
extern void ClearMatrix();

