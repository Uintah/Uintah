
/*
 *  VoidStar.h: Just has a rep member -- other trivial classes can inherit
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

#ifndef SCI_Datatypes_VoidStar_h
#define SCI_Datatypes_VoidStar_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/String.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Datatypes/Datatype.h>
#include <Datatypes/Pixel.h>
#include <Datatypes/Spectrum.h>
#include <Multitask/ITC.h>

// these are for the SiRe stuff (in Modules/MRA) which most people won't use...
#include <Datatypes/sire_const.h>
#include <Datatypes/sire_struct.h>
#include <Datatypes/sire_version.h>

class PhantomXYZ;
class PhantomUVW;
class SiReData;
class Phantoms;
class Pulses;
class DRaytracer;
class ImageXYZ;
class ImageRM;

class VoidStar;
typedef LockingHandle<VoidStar> VoidStarHandle;
class VoidStar : public Datatype {
protected:
    enum Representation {
	PhantomsType,
        PulsesType,
	//	DRaytracerType,
	ImageXYZType,
	ImageRMType,
        PhantomXYZType,
        PhantomUVWType,
	SiReDataType,
        Other
    };
    VoidStar(Representation);
private:
    Representation rep;
public:
    VoidStar(const VoidStar& copy);
    virtual ~VoidStar();
    virtual VoidStar* clone()=0;
    Pulses* getPulses();
    Phantoms* getPhantoms();
  //    DRaytracer* getDRaytracer();
    ImageXYZ* getImageXYZ();
    ImageRM* getImageRM();
    PhantomXYZ* getPhantomXYZ();  // i.e. current phantom position in some
                                  // coordinate system (not yet decided)
    PhantomUVW* getPhantomUVW();  // i.e. force broken out into 3 components
    SiReData* getSiReData();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


// SiRe class definition

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

void Pio(Piostream&, SiReDataS&);

class SiReData : public VoidStar {
public:
    SiReDataS s;
    Semaphore lockstepSem;
public:
    SiReData(Representation r=SiReDataType);
    SiReData(const SiReData& copy);
    virtual ~SiReData();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


// Phantoms class definition

typedef struct _Phantom {
    clString type;
    Point min;
    Point max;
    double T1;
    double T2;
} Phantom;
void Pio(Piostream&, Phantom&);

class Phantoms : public VoidStar {
public:
    Array1<Phantom> objs;
public:
    Phantoms(Representation r=PhantomsType);
    Phantoms(const Phantoms& copy);
    virtual ~Phantoms();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};



// Pulses class definition

typedef struct _Pulse {
    clString name;
    double start;
    double stop;
    double amplitude;
    char direction;
    int samples;
} Pulse;
void Pio(Piostream&, Pulse&);

class Pulses : public VoidStar {
public:
    Array1<Pulse> objs;
public:
    Pulses(Representation r=PulsesType);
    Pulses(const Pulses& copy);
    virtual ~Pulses();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class ImageXYZ : public VoidStar {
public:
    Array2<double> xyz;
public:
    ImageXYZ(Representation r=ImageXYZType);
    ImageXYZ(const ImageXYZ& copy);
    ImageXYZ(const Array2<double>& xyz);
    virtual ~ImageXYZ();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};
    
class ImageRM : public VoidStar {
public:
    Array2<Pixel> pix;
    Array1<Spectrum> lightSpec;
    DenseMatrix *LS;
    Array1<clString> lightName;
    Array1<Spectrum> matlSpec;
    DenseMatrix *MS;
    Array1<clString> matlName;
    Array1<double> kd;
    Array1<double> ks;
    Spectrum ka;
    DenseMatrix *KAS;
    int min, max, num;
    double spacing;
public:
    ImageRM(Representation r=ImageRMType);
    ImageRM(const ImageRM& copy);
    ImageRM(const Array2<Pixel>& p, const Array1<Spectrum>& ls, 
	    const Array1<clString>& ln, const Array1<Spectrum>& ms,
	    const Array1<clString>& mn, const Array1<double>& d,
	    const Array1<double>& s, const Spectrum& a);
    void getSpectraMinMax(double &min, double &max);
    void bldSpecLightMatrix(int lindex);
    void bldSpecMaterialMatrix(int mindex);
    void bldSpecAmbientMatrix();
    void bldSpecMatrices();
    void bldPixelR();
    void bldPixelSpectrum();
    void bldPixelXYZandRGB();
    virtual ~ImageRM();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

//#include <Modules/CS684/DRaytracer.h>

// Phantom Position class definition
// based heavily on dweinste's VoidStar.cc in his own work/Datatypes dir
// ImageXYZ

class PhantomXYZ : public VoidStar {
public: 
    Vector position;
    Semaphore sem;
    Semaphore Esem;  // ideally I would have a registry -- anyone who wanted
             // to receive position could sign up. Hack for now: each one
          // gets its own hard-coded semaphore LATER
    CrowdMonitor updateLock; 
public:
    PhantomXYZ(Representation r=PhantomXYZType);
    PhantomXYZ(const PhantomXYZ& copy);
    PhantomXYZ(const Vector& xyz);
    virtual  ~PhantomXYZ();
    virtual  VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

// Phantom Force class definition
// based heavily on dweinste's VoidStar.cc in his own work/Datatypes dir
// ImageXYZ

class PhantomUVW : public VoidStar {
public: 
    Vector force;
    Semaphore sem;
    CrowdMonitor updateLock; 
public:
    PhantomUVW(Representation r=PhantomUVWType);
    PhantomUVW(const PhantomUVW& copy);
    PhantomUVW(const Vector& uvw);
    virtual  ~PhantomUVW();
    virtual  VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
