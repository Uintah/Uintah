
/*
 *  SiRe.cc: The SiRe classes - derived from VoidStar
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/SiRe/SiRe.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>

#include <iostream>
using std::cerr;

// Here's the code for the SiReData

namespace DaveW {
using namespace SCIRun;

static Persistent* make_SiReData()
{
    return scinew SiReData;
}
PersistentTypeID SiReData::type_id("SiReData", "VoidStar", make_SiReData);

SiReData::SiReData()
: VoidStar(), lockstepSem("SiReData lockstep semaphore", 0)
{
    s.PFiles=0;
    s.RcvrSlabImgFiles=0;
    s.Rcvr3DImgFiles=0;
    s.FinalImgFile=0;
    s.SlabIndices=0;
    s.ShiftRcvrRaw=0;
    s.ZFIRcvrImg=0;
    s.RcvrDCOffInd=0;
    s.OverlapImg=0;
    s.Filter=0;
    s.TimeStamp=0;
    s.RunTime=0;
    s.FirstSlab=0;
    s.LastSlab=0;
    s.IRcvr=0;
    s.ISlab=0;
    s.NPasses=0;
    s.PassIdx=0;
    s.ShrinkFactor=0;
}

SiReData::SiReData(const SiReData& copy)
: VoidStar(copy), lockstepSem("SiReData lockstep semaphore", 0)
{
    s=copy.s;
    int i,j;

    /* copy pointer data */
    if (s.PFiles) {
	s.PFiles = (char **) malloc(s.NPFile * sizeof(char *));
	for (i=0; i<s.NPFile; i++) {
	    s.PFiles[i] = (char *) malloc(SIRE_MAXCHARLEN * sizeof(char));
	    for (j=0; j<SIRE_MAXCHARLEN; j++)
		s.PFiles[i][j] = copy.s.PFiles[i][j];
	}
    }
    if (s.FinalImgFile) {
	s.FinalImgFile = (char *) malloc(SIRE_MAXCHARLEN * sizeof(char));
	for (i=0; i<SIRE_MAXCHARLEN; i++) {
	    s.FinalImgFile[i] = copy.s.FinalImgFile[i];
	}
    }
    if (s.SlabIndices) {
	s.SlabIndices = (int *) malloc(s.NSlab * sizeof(int));
	for (i=0; i<s.NSlab; i++) 
	    s.SlabIndices[i]=copy.s.SlabIndices[i];
    }
    s.NPasses=copy.s.NPasses;
    s.PassIdx=copy.s.PassIdx;
    s.ShrinkFactor=copy.s.ShrinkFactor;
    /* don't bother copying these -- if they exist, just copy the pointer */

    if (s.RcvrRaw) cerr<< "copying pointer to SiReData.RcvrRaw...\n";
    if (s.ShiftRcvrRaw) cerr<< "copying pointer to SiReData.ShiftRcvrRaw...\n";
    if (s.ZFIRcvrImg) cerr<< "copying pointer to SiReData.ZFIRcvrImg...\n";
    if (s.RcvrDCOffInd) cerr<< "copying pointer to SiReData.RcvrDCOffInd...\n";
    if (s.OverlapImg) cerr<< "copying pointer to SiReData.OverlapImg...\n";
    if (s.Filter) cerr<< "copying pointer to SiReData.Filter...\n";
}

SiReData::~SiReData() {
}

VoidStar* SiReData::clone()
{
    return scinew SiReData(*this);
}

#define SiReData_VERSION 1
void SiReData::io(Piostream& stream) {
    using DaveW::Pio;

    /* int version=*/stream.begin_class("SiReData", SiReData_VERSION);
    VoidStar::io(stream);
    Pio(stream, s);
    stream.end_class();
}

void Pio(Piostream& stream, SIRE_DIRINFO& d) {

    stream.begin_cheap_delim();
    Pio(stream, d.Read);	
    Pio(stream, d.Phase);
    Pio(stream, d.Slice);
    stream.end_cheap_delim();
}

void Pio(Piostream& stream, SIRE_FLAGS& f) {
    using DaveW::Pio;

    stream.begin_cheap_delim();
    Pio(stream, f.ReconType);
    Pio(stream, f.RDBRawCollect);
    Pio(stream, f.AcqType);
    Pio(stream, f.HotsaData);
    Pio(stream, f.NCuppenIter);
    Pio(stream, f.AutoDetectAsym);
    Pio(stream, f.AutoScalePhase);
    Pio(stream, f.AutoScaleSlice);
    Pio(stream, f.AsymmetricEcho);
    Pio(stream, f.NkzAcquired);
    Pio(stream, f.Trunc);
    Pio(stream, f.FindEveryCntr);
    Pio(stream, f.ReconRcvrSlabs);
    Pio(stream, f.InitReconkz);
    Pio(stream, f.FlattenBB);
    Pio(stream, f.SaveScale);
    Pio(stream, f.SaveFilter);
    Pio(stream, f.StartRcvr);
    Pio(stream, f.StartSlab);
    Pio(stream, f.EndRcvr);
    Pio(stream, f.EndSlab);
    Pio(stream, f.OverlapRcvrSlabs);
    Pio(stream, f.InPlaceOverlap);
    Pio(stream, f.AutoCor);
    Pio(stream, f.SaveRcvrSlabs);
    Pio(stream, f.MakeZFImg);
    Pio(stream, f.SaveRcvrs);
    Pio(stream, f.MakeNonZFImg);
    Pio(stream, f.MakeSignaImg);
    Pio(stream, f.SaveZFImg);
    stream.end_cheap_delim();
}
    
void Pio(Piostream& stream, SIRE_IMGINFO& i) {

    stream.begin_cheap_delim();
    Pio(stream, i.RawImgDC);
    Pio(stream, i.MaxImgValue);
    Pio(stream, i.RealPartOut);
    Pio(stream, i.UseUserScale);
    Pio(stream, i.UserScale);
    stream.end_cheap_delim();
}

void Pio(Piostream& stream, SIRE_FILTERINFO& f) {
    using DaveW::Pio;

    stream.begin_cheap_delim();
    Pio(stream, f.Type);
    Pio(stream, f.Width);
    stream.end_cheap_delim();
}

void Pio(Piostream& stream, SIRE_SIGNA_IMGINFO& s) {
    using DaveW::Pio;

    stream.begin_cheap_delim();
    Pio(stream, s.ImgAxis);
    Pio(stream, s.Rotation);
    Pio(stream, s.Transpose);
    Pio(stream, s.CoordType);
    Pio(stream, s.ImgRead0);
    Pio(stream, s.ImgPhase0);
    Pio(stream, s.NImgRead);
    Pio(stream, s.NImgPhase);
    Pio(stream, s.HdrExamNum);
    Pio(stream, s.ForceNewExam);
    Pio(stream, s.NewExamNum);
    Pio(stream, s.NewSeriesNum);
    Pio(stream, s.XPixelSize);
    Pio(stream, s.YPixelSize);
    Pio(stream, s.SliceThick);
    Pio(stream, s.ScanSpacing);
    stream.end_cheap_delim();
}

void Pio(Piostream& stream, DaveW::SiReDataS& s)
{
    using DaveW::Pio;

    stream.begin_cheap_delim();
    int i;
//    for (i=0; i<SIRE_MAXCHARLEN; i++)
//	Pio(stream, s.FirstPFile[i]);
    Pio(stream, s.Flag);
    Pio(stream, s.Step);
    for (i=0; i<SIRE_MAXFILTYPEDIM; i++)
	Pio(stream, s.FilterInfo[i]);
    Pio(stream, s.SireImgInfo);
    Pio(stream, s.SignaImgInfo);
    Pio(stream, s.TimeStamp);
    Pio(stream, s.RunTime);
//    for (i=0; i<SIRE_MAXCHARLEN; i++)
//	Pio(stream, s.TimeStr[i]);
    Pio(stream, s.FirstSlab);
    Pio(stream, s.LastSlab);
    Pio(stream, s.IRcvr);
    Pio(stream, s.ISlab);
    Pio(stream, s.NRead);
    Pio(stream, s.NPhase);
    Pio(stream, s.NRawRead);
    Pio(stream, s.NRawPhase);
    Pio(stream, s.NRawSlice);
    Pio(stream, s.NSlabPerPFile);
    Pio(stream, s.NPFile);
    Pio(stream, s.NSlBlank);
    Pio(stream, s.Rcvr0);
    Pio(stream, s.RcvrN);
    Pio(stream, s.NFinalImgSlice);
    Pio(stream, s.PointSize);
//    if (stream.writing()) {
//	for (i=0; i<s.NPFile; i++)
//	    for (int j=0; j<SIRE_MAXCHARLEN; j++)
//		Pio(stream, s.PFiles[i][j]);
//    }
//    Pio(stream, s.RcvrSlabImgFiles);
//    Pio(stream, s.Rcvr3DImgFiles);
//    Pio(stream, s.FinalImgFile);
//    Pio(stream, s.SlabIndices);
    Pio(stream, s.NSlicePerRcvr);
    Pio(stream, s.NRcvrPerSlab);
    Pio(stream, s.NSlab);
    Pio(stream, s.NRcnRead);
    Pio(stream, s.NRcnPhase);
    Pio(stream, s.NRcnSlicePerRcvr);
    Pio(stream, s.NRcnOverlap);
    Pio(stream, s.NRcnFinalImgSlice);
//    Pio(stream, s.RcvrRaw);
//    Pio(stream, s.ShiftRcvrRaw);
//    Pio(stream, s.ZFIRcvrImg);
//    Pio(stream, s.RcvrDCOffInd);
//    Pio(stream, s.OverlapImg);
//    Pio(stream, s.Filter);
    stream.end_cheap_delim();
}
} // End namespace DaveW


