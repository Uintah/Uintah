
/*
 *  VoidStar.cc:Just has a rep member -- other trivial classes can inherit
 *		from this, rather than having a full-blown datatype and data-
 *		port for every little thing that comes along...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Datatypes/VoidStar.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <Math/MinMax.h>
#include <iostream.h>

PersistentTypeID VoidStar::type_id("VoidStar", "Datatype", 0);

VoidStar::VoidStar(Representation rep)
: rep(rep)
{
}

VoidStar::VoidStar(const VoidStar& copy)
: rep(copy.rep)
{
    NOT_FINISHED("VoidStar::VoidStar");
}

VoidStar::~VoidStar()
{
}

#define VoidStar_VERSION 1
void VoidStar::io(Piostream& stream) {
    stream.begin_class("VoidStar", VoidStar_VERSION);
    int* repp=(int*)&rep;
    Pio(stream, *repp);
    stream.end_class();
}




// Here's the code for the SiReData

SiReData* VoidStar::getSiReData() {
    if (rep==SiReDataType) {
	return (SiReData*)this;
    } else
	return 0;
}

static Persistent* make_SiReData()
{
    return scinew SiReData;
}
PersistentTypeID SiReData::type_id("SiReData", "VoidStar", make_SiReData);

SiReData::SiReData(Representation r)
: VoidStar(r), lockstepSem(0)
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
: VoidStar(copy), lockstepSem(0)
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
    stream.begin_cheap_delim();
    Pio(stream, f.Type);
    Pio(stream, f.Width);
    stream.end_cheap_delim();
}

void Pio(Piostream& stream, SIRE_SIGNA_IMGINFO& s) {
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

void Pio(Piostream& stream, SiReDataS& s)
{
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



// Here's the code for the Phantoms

Phantoms* VoidStar::getPhantoms() {
    if (rep==PhantomsType) {
	return (Phantoms*)this;
    } else
	return 0;
}

static Persistent* make_Phantoms()
{
    return scinew Phantoms;
}
PersistentTypeID Phantoms::type_id("Phantoms", "VoidStar", make_Phantoms);

Phantoms::Phantoms(Representation r)
: VoidStar(r)
{
}

Phantoms::Phantoms(const Phantoms& copy)
: VoidStar(copy)
{
    NOT_FINISHED("Phantoms::Phantoms");
}

Phantoms::~Phantoms() {
}

VoidStar* Phantoms::clone()
{
    return scinew Phantoms(*this);
}

#define Phantoms_VERSION 1
void Phantoms::io(Piostream& stream) {
    /* int version=*/stream.begin_class("Phantoms", Phantoms_VERSION);
    VoidStar::io(stream);
    Pio(stream, objs);
    stream.end_class();
}
void Pio(Piostream& stream, Phantom& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p.type);
    Pio(stream, p.min);
    Pio(stream, p.max);
    Pio(stream, p.T1);
    Pio(stream, p.T2);
    stream.end_cheap_delim();
}



// Here's the code for the pulses

Pulses* VoidStar::getPulses() {
    if (rep==PulsesType) {
	return (Pulses*)this;
    } else
	return 0;
}

static Persistent* make_Pulses()
{
    return scinew Pulses;
}
PersistentTypeID Pulses::type_id("Pulses", "VoidStar", make_Pulses);

Pulses::Pulses(Representation r)
: VoidStar(r)
{
}

Pulses::Pulses(const Pulses& copy)
: VoidStar(copy)
{
    NOT_FINISHED("Pulses::Pulses");
}

Pulses::~Pulses() {
}

VoidStar* Pulses::clone()
{
    return scinew Pulses(*this);
}

#define Pulses_VERSION 1
void Pulses::io(Piostream& stream) {
    /* int version=*/stream.begin_class("Pulses", Pulses_VERSION);
    VoidStar::io(stream);
    Pio(stream, objs);
    stream.end_class();
}

void Pio(Piostream& stream, Pulse& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p.name);
    Pio(stream, p.start);
    Pio(stream, p.stop);
    Pio(stream, p.amplitude);
    Pio(stream, p.direction);
    Pio(stream, p.samples);
    stream.end_cheap_delim();
}

ImageXYZ* VoidStar::getImageXYZ() {
    if (rep==ImageXYZType) {
	return (ImageXYZ*)this;
    } else
	return 0;
}

static Persistent* make_ImageXYZ()
{
    return scinew ImageXYZ;
}
PersistentTypeID ImageXYZ::type_id("ImageXYZ", "VoidStar", make_ImageXYZ);

ImageXYZ::ImageXYZ(Representation r)
: VoidStar(r)
{
}

ImageXYZ::ImageXYZ(const Array2<double>& XYZ) : VoidStar(ImageXYZType)
{
    xyz=XYZ;
}

ImageXYZ::ImageXYZ(const ImageXYZ& copy)
: xyz(copy.xyz), VoidStar(copy)
{
    NOT_FINISHED("ImageXYZ::ImageXYZ");
}

ImageXYZ::~ImageXYZ() {
}

VoidStar* ImageXYZ::clone()
{
    return scinew ImageXYZ(*this);
}

#define ImageXYZ_VERSION 1
void ImageXYZ::io(Piostream& stream) {
    /* int version=*/stream.begin_class("ImageXYZ", ImageXYZ_VERSION);
    VoidStar::io(stream);
    Pio(stream, xyz);
    stream.end_class();
}

ImageRM* VoidStar::getImageRM() {
    if (rep==ImageRMType) {
	return (ImageRM*)this;
    } else
	return 0;
}

static Persistent* make_ImageRM()
{
    return scinew ImageRM;
}
PersistentTypeID ImageRM::type_id("ImageRM", "VoidStar", make_ImageRM);

ImageRM::ImageRM(Representation r)
: VoidStar(r)
{
}

ImageRM::ImageRM(const Array2<Pixel>& p, const Array1<Spectrum>& ls, 
	    const Array1<clString>& ln, const Array1<Spectrum>& ms,
	    const Array1<clString>& mn, const Array1<double>& d,
	    const Array1<double>& s, const Spectrum& a) : 
    VoidStar(ImageRMType), pix(p), lightSpec(ls), lightName(ln),
    matlSpec(ms), matlName(mn), kd(d), ks(s), ka(a), LS(0), MS(0), KAS(0)
{
}

ImageRM::ImageRM(const ImageRM& copy)
: pix(copy.pix), lightSpec(copy.lightSpec), lightName(copy.lightName),
  matlSpec(copy.matlSpec), matlName(copy.matlName), kd(copy.kd), 
  ks(copy.ks), ka(copy.ka), VoidStar(copy), LS(0), MS(0), KAS(0)
{
    NOT_FINISHED("ImageRM::ImageRM");
}

void ImageRM::getSpectraMinMax(double &min, double &max) {
    cerr << "ImageRM::getSpectraMinMax() NOT IMPLEMENTED!\n";
    min=400;
    max=1000;
}

void ImageRM::bldSpecLightMatrix(int lindex) {
    Array1<double> specvals(num);
    lightSpec[lindex].rediscretize(specvals, min, max);
    for (int j=0; j<num; j++) (*LS)[j][lindex]=specvals[j];
}

void ImageRM::bldSpecMaterialMatrix(int mindex) {
    Array1<double> specvals(num);
    matlSpec[mindex].rediscretize(specvals, min, max);
    for (int j=0; j<num; j++) (*MS)[mindex][j]=specvals[j];
}

void ImageRM::bldSpecAmbientMatrix() {
    Array1<double> specvals(num);
    ka.rediscretize(specvals, min, max);
    for (int j=0; j<num; j++) (*KAS)[j][0]=specvals[j];
}

void ImageRM::bldSpecMatrices() {

    // to build our Spectral matrices, we'll discretize one spectrum at a
    // time and then copy the discretized values into a row of the matrix

    if (MS) free(MS);
    MS = new DenseMatrix(matlSpec.size(), num);
    for (int i=0; i<matlSpec.size(); i++) {
	bldSpecMaterialMatrix(i);
    }

    if (LS) free(LS);
    LS = new DenseMatrix(num, lightSpec.size());
    for (i=0; i<lightSpec.size(); i++) {
	bldSpecLightMatrix(i);
    }

    if (KAS) free(KAS);
    KAS = new DenseMatrix(num, 1);
    bldSpecAmbientMatrix();
}

// R = B(l,1) kd + [B(l,2) kd + B(l,3) kd B(3,2) ks] B(2,1) ks
void ImageRM::bldPixelR() {
    if (pix.dim1() + pix.dim2() == 0) {
	cerr << "NO PIXELS!\n";
	return;
    }

    int nb, nm, nl;
    nb=pix(0,0).D.size();
    nm=matlSpec.size();
    nl=lightSpec.size();
    
    if (kd.size()!=nm || ks.size()!=nm) {
	cerr << "ERROR -- ks and kd matlSpec must be the same size!\n";
	return;
    }
    
    DenseMatrix KD(nm, nm);
    DenseMatrix KS(nm, nm);
    DenseMatrix S0M(nm, nm);
    KS.zero(); KD.zero();
    
    for (int i=0; i<nm; i++) {
//	KS[i][i]=ks[i];
	KS[i][i]=1;
	KD[i][i]=kd[i];
    }


    for (int j=0; j<pix.dim1(); j++) {
	for (i=0; i<pix.dim2(); i++) {
//	    cerr << "Building R for pixel ("<<j<<","<<i<<")\n";
	    if (pix(j,i).E.size() != nl) {
		cerr << "Error in pixel("<<j<<","<<i<<") bad E.\n";
		return;
	    }

	    // gotta check the S and M matrices...
	    for (int k=1; k<nb; k++) {
		if (pix(j,i).S[k-1].nrows() != nm || 
		    pix(j,i).S[k-1].ncols() != nm) {
		    cerr <<"Error in pixel("<<j<<","<<i<<") bad S["<<k-1<<"\n";
		    return;
		}
//		cerr << "Here's S["<<k-1<<"] - ";
//		pix(j,i).S[k-1].print();
		if (pix(j,i).D[k].nrows() != nl || 
		    pix(j,i).D[k].ncols() != nm) {
		    cerr << "Error in pixel("<<j<<","<<i<<") bad D["<<k<<".\n";
		    return;
		}
//		cerr << "Here's D["<<k<<"] - \n";
//		pix(j,i).D[k].print();
	    }

	    // sizes look ok, let's build R...
	    DenseMatrix Rt(nl, nm);
	    DenseMatrix tmp1(nl, nm);
	    DenseMatrix tmp2(nl, nm);

	    // initialize Rt with the last bounce's direct, scaled by kd
	    Mult(Rt, pix(j,i).D[nb-1], KD);
	    cerr << "Pixel("<<j<<","<<i<<")...\n";
	    cerr << "Rt=";Rt.print();
	    // for each bounce, multiply by the specular, add the diffuse,
	    // save in Rt
	    for (k=nb-2; k>=0; k--) {
		Mult(tmp2, Rt, pix(j,i).S[k]);
	    cerr << "tmp2=";tmp2.print();
		Mult(tmp1, tmp2, KS);
	    cerr << "tmp1=";tmp1.print();
		Mult(tmp2, pix(j,i).D[k], KD);
	    cerr << "tmp2=";tmp2.print();
		Add(Rt, tmp1, tmp2);
	    cerr << "Rt=";Rt.print();
	    }

	    // gotta check the S0 vector...
	    if (pix(j,i).S0.size() != nm) {
		cerr << "Error in pixel("<<j<<","<<i<<") bad S0.\n";
		return;
	    }	    

#if 0
	    // S0 is the probability that a ray from the eye sees each matl
	    // Make it into a matrix, so we can multiply by it to get R
	    S0M.zero();
	    for (k=0; k<nm; k++) S0M[k][k]=pix(j,i).S0[k];
	    Mult(tmp1, Rt, S0M);
	    pix(j,i).R=tmp1;
#endif
	    pix(j,i).R=Rt;

//	    cerr << "Here's R: \n";
//	    pix(j,i).R.print();
	}
    }
}

// We have an R matrix for each pixel.  Compute the spectrum one lambda at a
// time for each pixel...
// for each wavelenght: multiply R by the material vector for that
// wavelength, add in the emitted light vector, multiply by the light vector
// for that wavelength, and add in the ambient value for that wavelength

void ImageRM::bldPixelSpectrum() {
    int l,w,x,y;
    int nl=lightSpec.size();
    int nm=matlSpec.size();
    int nw=KAS->nrows();
    int ny=pix.dim1();
    int nx=pix.dim2();
    
    DenseMatrix RM(nl,1);
    DenseMatrix ERM(nl,1);
    DenseMatrix LERM(1,1);
    DenseMatrix ALERM(1,1);
    DenseMatrix E(nl,1);
    DenseMatrix MSlambda(nm,1);
    DenseMatrix LSlambda(1,nl);
    
    for (y=0; y<ny; y++) {
	for (x=0; x<nx; x++) {
	    for (l=0; l<nl; l++) E[l][0]=pix(y,x).E[l];
	    double *sp=new double[nw];
	    for (w=0; w<nw; w++) {			// for each wavelength
		for (int mm=0; mm<nm; mm++) MSlambda[mm][0]=(*MS)[mm][w];
		for (int ll=0; ll<nl; ll++) LSlambda[0][ll]=(*LS)[w][ll];
		Mult(RM, pix(y,x).R, MSlambda);
		Add(ERM, E, RM);
		Mult(LERM, LSlambda, ERM);

		LERM[0][0] /= pix(y,x).nSamples;

		ALERM[0][0]=(*KAS)[w][0]+LERM[0][0];

//		Add(ALERM, *KAS, LERM);

		sp[w]=ALERM[0][0];
	    }
	    pix(y,x).spec=LiteSpectrum(min, max, num, spacing, sp);
//	    cerr << "Pixel ("<<y<<","<<x<<") spec = ";
//	    for (int i=0; i<nw; i++)
//		cerr << sp[i]<<" ";
//	    cerr << "\n";
	}
    }
}

void ImageRM::bldPixelXYZandRGB() {
    Spectrum XSpectra(xAmplitudes, xSpacing, xMinWavelength, xNumEntries);
    Spectrum YSpectra(yAmplitudes, ySpacing, yMinWavelength, yNumEntries);
    Spectrum ZSpectra(zAmplitudes, zSpacing, zMinWavelength, zNumEntries);

    Array1<double> tempXSpectra(num);
    Array1<double> tempYSpectra(num);
    Array1<double> tempZSpectra(num);

    XSpectra.rediscretize(tempXSpectra, min, max);
    YSpectra.rediscretize(tempYSpectra, min, max);
    ZSpectra.rediscretize(tempZSpectra, min, max);
    
    int x,y;
    int nx=pix.dim2();
    int ny=pix.dim1();

    for (y=0; y<ny; y++) {
	for (x=0; x<nx; x++) {
	    Point xyz(pix(y,x).spec.xyz(&(tempXSpectra[0]), 
					&(tempYSpectra[0]), 
					&(tempZSpectra[0])));
//	    cerr << "Pixel ("<<y<<","<<x<<") xyz = "<<xyz<<"\n";
	    Color clr=XYZ_to_RGB(xyz);
	    pix(y,x).xyz=xyz;
	    pix(y,x).c.red=Min(255.,Max(clr.r()*255.,0.));
	    pix(y,x).c.green=Min(255.,Max(clr.g()*255.,0.));
	    pix(y,x).c.blue=Min(255.,Max(clr.b()*255.,0.));
	}
    }
}

ImageRM::~ImageRM() {
}

VoidStar* ImageRM::clone()
{
    return scinew ImageRM(*this);
}

#define ImageRM_VERSION 1
void ImageRM::io(Piostream& stream) {
    /* int version=*/stream.begin_class("ImageRM", ImageRM_VERSION);
    VoidStar::io(stream);
    Pio(stream, pix);
    Pio(stream, lightSpec);
    Pio(stream, lightName);
    Pio(stream, matlSpec);
    Pio(stream, matlName);
    Pio(stream, kd);
    Pio(stream, ks);
    Pio(stream, ka);
    Pio(stream, min);
    Pio(stream, max);
    Pio(stream, num);
    Pio(stream, spacing);
    stream.end_class();
}

// PHANTOM XYZ DEFNS

PhantomXYZ* VoidStar::getPhantomXYZ() {
    if (rep==PhantomXYZType) {
        return (PhantomXYZ*)this;
    } else
        return 0;
}

static Persistent* make_PhantomXYZ()
{
    return scinew PhantomXYZ;
}
PersistentTypeID PhantomXYZ::type_id("PhantomXYZ", "VoidStar", make_PhantomXYZ);

PhantomXYZ::PhantomXYZ(Representation r)
: VoidStar(r), sem(0), Esem(0)
{
   
}

PhantomXYZ::PhantomXYZ(const Vector& XYZ) : VoidStar(PhantomXYZType), 
sem(0), Esem(0)
{
    position=XYZ;
}

PhantomXYZ::PhantomXYZ(const PhantomXYZ& copy)
: position(copy.position), VoidStar(copy), sem(0), Esem(0)
{
    NOT_FINISHED("PhantomXYZ::PhantomXYZ");
}

PhantomXYZ::~PhantomXYZ() {
}

VoidStar* PhantomXYZ::clone()
{
    return scinew PhantomXYZ(*this);
}

#define PhantomXYZ_VERSION 1
void PhantomXYZ::io(Piostream& stream) {
    /* int version=*/stream.begin_class("PhantomXYZ", PhantomXYZ_VERSION);
    VoidStar::io(stream);
    Pio(stream, position);
    stream.end_class();
}

// PHANTOM UVW DEFINITIONS
PhantomUVW* VoidStar::getPhantomUVW() {
    if (rep==PhantomUVWType) {
        return (PhantomUVW*)this;
    } else
        return 0;
}

static Persistent* make_PhantomUVW()
{
    return scinew PhantomUVW;
}
PersistentTypeID PhantomUVW::type_id("PhantomUVW", "VoidStar", make_PhantomUVW);

PhantomUVW::PhantomUVW(Representation r)
: VoidStar(r), sem(0)
{
}

PhantomUVW::PhantomUVW(const Vector& UVW) : VoidStar(PhantomUVWType), sem(0)
{
    force = UVW;
}

PhantomUVW::PhantomUVW(const PhantomUVW& copy)
: force(copy.force), VoidStar(copy), sem(0)
{
    NOT_FINISHED("PhantomUVW::PhantomUVW");
}

PhantomUVW::~PhantomUVW() {
}

VoidStar* PhantomUVW::clone()
{
    return scinew PhantomUVW(*this);
}

#define PhantomUVW_VERSION 1
void PhantomUVW::io(Piostream& stream) {
    /* int version=*/stream.begin_class("PhantomUVW", PhantomUVW_VERSION);
    VoidStar::io(stream);
    Pio(stream, force);
    stream.end_class();
}


#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template class LockingHandle<VoidStar>;

#include <Classlib/Array1.cc>
template class Array1<Phantom>
template class Array1<Pulse>
template void Pio(Piostream&, Array1<Phantom>&);
template void Pio(Piostream&, Array1<Pulse>&);
#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array1.cc>

static void _dummy_(Piostream& p1, Array1<Phantom>& p2)
{
    Pio(p1, p2);
}

static void _dummy_(Piostream& p1, Array1<Pulse>& p2)
{
    Pio(p1, p2);
}

#endif
#endif
