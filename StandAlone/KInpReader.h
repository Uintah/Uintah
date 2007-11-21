// KInpReader.h
//
// Copyright 2005  Reaction Engineering International
// by Yang
/////////////////////////////////////////////////////

#ifndef KINPREADER_H
#define KINPREADER_H

#pragma warning(disable : 4786)

#include <vector>
#include <string>
#include <map>
#include <set>
#include <cstdio>

//new data type to handle
typedef struct
{
	//card 1
	unsigned int MID;
	double RO;
	double E;
	double PR;
	double SIGY;
	double ETAN;
	double BETA;
	//card2
	double SRC;
	double SRP;
	double FS;
	double VP; //op

}MatPlasticKinematic;

typedef struct
{
	 unsigned int MID;
	 double RO;
	 double E; 
	 double PR;
	 double DA;
	 double DB;
	 double K;
}MatElastic;

typedef struct 
{
	//card 1
	unsigned int MID;
	double RO;
	double E;
	double PR;
	double N;
	double COUPLE;
	double M;
	double ALIASRE;
	//card 2
	double CMO;
	double CON1;
	double CON2;
	//card 3
	double LCOA1;
	double A2;
	double A3;
	double V1;
	double V2;
	double V3;
}MatRigid;

typedef struct {
	//card 1
	unsigned int MID;
	unsigned int LCD;
	unsigned int LCR;
	

}MatSpringNonlinearElastic;


typedef struct{
	unsigned int MID;
	double K;
}MatSpringElastic;


typedef struct {
	//card 1
	unsigned int MID;
	double RO;
	double E;
	double PR;
	double SIGY;
	double ETAN;
	double FAIL;
	double TDEL;
	//card 2
	double C;
	double P;
	unsigned int LCSS;
	unsigned int LCSR;
	double VP;
	//card 3
	double EPS[8];
	//card 4
	double ES[8];
}MatPiecewiseLinearPlasticity;


typedef struct {
	unsigned int MID;
	double RO;
	double G;
	double REF;

}MatBlatzKORubber;


typedef struct {
	//card 1
	unsigned int MID;
	double RO;
	double E;
	double PR;
	double SIGY;
	double VF;
	double MU;
	double BULK;
	//card 2;
	unsigned int LCA;
	unsigned int LCB;
	unsigned int LCC;
	unsigned int LCS;
	unsigned int LCAB;
	unsigned int LCBC;
	unsigned int LCCA;
	unsigned int LCSR;
	//card 3;
	double EAAU;
	double EBBU;
	double ECCU;
	double GABU;
	double GBCU;
	double GCAU;
	double AOPT;
	//card 4;
	double XP;
	double YP;
	double ZP;
	double A1;
	double A2;
	double A3;
	//card 5
	double D1;
	double D2;
	double D3;
	double TSEF;
	double SSEF;

} MatHoneycomb;

typedef struct
{
	unsigned int MID;
	double DC;

} MatDamperViscous;


typedef struct
{
	unsigned int HGID;
	unsigned int IHQ;
	double QM;
	unsigned int IBQ;
	double Q[2];
	double QB;
	double QW;
}Hourglass;



//yang

typedef struct 
{
	unsigned long secid;
	unsigned long elform;
	double shrf;
	double qr_irid;
	double cst;
	double scoor;
	double nsm;
	double f[8];
} SecBeam;

typedef struct 
{
	unsigned long secid;
	unsigned long elform;
	double shrf;
	double nip;
	double propt;
	double or_irid;
	unsigned long icomp;
	unsigned long setyp;

	double t[4];
	double nloc;
	double marea;

} SecShell;

typedef struct 
{
	unsigned long secid;
	unsigned long elform;
	unsigned long aet;
} SecSolid;

typedef struct
{
	//card 1
	unsigned long secid;
	unsigned long dro;
	double kd;
	double v0;
	double cl;
	double fd;
	//card 2
	double cdl;
	double tdl;
} SecDiscrete;

typedef struct
{
	unsigned long eid;
	unsigned long pid;
	unsigned long n[8];
} ElemSolid;

typedef struct
{
	unsigned long eid;
	unsigned long pid;
	unsigned long n[4];
	double t[4];
	double psi;
    double sign;
} ElemShell;

typedef struct
{
	unsigned long eid;
	unsigned long pid;
	unsigned long n1;
	unsigned long n2;
	unsigned long n3;
	unsigned long rt1;
	unsigned long rr1;
	unsigned long rt2;
	unsigned long rr2;
	unsigned long local;
} ElemBeam;

typedef struct
{
	//card 1
	unsigned int SID;
	double DA1;
	double DA2;
	double DA3;
	double DA4;
	//cards 2,3,4 ...
	std::vector<unsigned int> NID;
} NodeList;

typedef struct
{
	std::string op;
	unsigned int LCID;
	double SF;
	unsigned int LCIDDR;
	double XC;
	double YC;
	double ZC;
} LoadBody;

typedef struct
{
	//card 1
	unsigned int LCID;
	unsigned int SIDR;
	double SFA;
	double SFO;
	double OFFA;
	double OFFO;
	int DATTYP;
	//card 2
   std::vector<std::pair<double, double> > AO;
} Curve;

typedef struct
{
	//card 1
	unsigned int NSID;
	unsigned int NSIDEX;
	unsigned int BOXID;
	double OFFSET;
	//card 2
	double XT;
	double YT;
	double ZT;
	double XH;
	double YH;
	double ZH;
	double FRIC;
	double WVEL;

} RigidWallPlanar;

typedef struct
{
	//card1
	unsigned int ID;
	unsigned int STYP;
	double OMEGA;
	double VX;
	double VY;
	double VZ;
	//card 2
	double XC;
	double YC;
	double ZC;
	double NX;
	double NY;
	double NZ;
	double PHASE;

}InitailVelocityGeneration;

typedef struct
{
	std::string option1;
	//card ID tile
	unsigned int CID;
	//card 1
	int SSID;
	int MSID;
	int SSTYP;
	int MSTYP;
	int SBOXID;
	int MBOXID;
	int SPR;
	int MPR;
	//card 2
	double FS;
	double FD;
	double DC;
	double VC;
	double VDC;
	int PENCHK;
	double BT;
	double DT;
	//card 3
	double SFS;
	double SFM;
	double SST;
	double MST;
	double SFST;
	double SFMT;
	double FSF;
	double VSF; 
	
} ContactOption1Title;

typedef struct
{
	int N[4];
	double A[4];
} SetSegmentCard2;

typedef struct
{
	unsigned int SID;
	double DA[4];
	std::vector<SetSegmentCard2> card2;
} SetSegment;

typedef struct
{
	unsigned int VID;
	int IOP;
	double XT;
	double YT;
	double ZT;
	int NID1;
	int NID2;

} SDOrientation;

typedef struct
{
	//card 1
	unsigned int SECID;
	int DRO;
	double KD;
	double V0;
	double CL;
	double FD;
	//card 2
	double CDL;
	double TDL;
} SectionDiscrete;

typedef struct
{
	//card 1
	unsigned int SID;
	int SIDTYP;
	int RBID;
	double VSCA;
	double PSCA;
	double VINI;
	double MWD;
	double SPSF;
	//card 2
	double CV;
	double CP;
	double T;
	int LCID;
	double MU;
	double A;
	double PE;
	double RO;
	//card 3
	int LOU;
	double TEXT;
	double Acard2;
	double B;
	double MW;
	double GASC;
} AirbagSimpleAirbagModel;

typedef struct
{
	unsigned int EID;
	unsigned long PID;
	int N1;
	int N2;
	int VID;
	double S;
	int PF;
	double OFFSET;
} ElementDiscrete;

typedef struct
{
	unsigned int EID;
	int NID;
	double MASS;
	int PID;
} ElementMass;

typedef struct
{
	unsigned int SBACID;
	int NID1;
	int NID2;
	int NID3;
	int IGRAV;
	int INTOPT;
} ElementSeatbeltAccelerometer;

class Node
{
public:
	unsigned long nid;
	double x[3];
	double tc;
	double rc;

   unsigned long nidloc; // part node number used for tri files in solid elements

};

typedef struct
{
	std::string option;
	unsigned int PID;
	unsigned int NIDNSID;
} ConstrainedExtraNodes;

#define JOINT_SIZE 6
typedef struct
{
	std::string option;
	unsigned int N[JOINT_SIZE];
	double RPS;
	double DAMP;
} ConstrainedJoint;

typedef struct
{
	unsigned int PID;
	unsigned int CID;
	unsigned int NSID;
} ConstrainedNodalRigidBody;

typedef struct
{
	unsigned int PIDM;
	unsigned int PIDS;
} ConstrainedRigidBody;

typedef struct
{
	unsigned int N1;
	unsigned int N2;
	double SN;
	double SS;
	double N;
	double M;
	double TF;
	double EP;
} ConstrainedSpotweld;

typedef struct
{
	unsigned long n[4];
   std::vector<unsigned long> en;
} face;

typedef struct
{
	unsigned long n[2];
   std::vector<unsigned long> en; 
} edge;

class Part
{
public:
   ~Part();
  std::string title; //card 1
  
  unsigned long pid; //card 2
  unsigned long secid;
  unsigned long mid;
  unsigned long eosid;
  unsigned long hgid;
  unsigned long grav;
  unsigned long adpopt;
  unsigned long tmid;

  unsigned long numsolid;
  unsigned long numshell;

  std::map<std::string, face> faces;
  std::map<unsigned long, ElemShell*> elemShells_m;
  std::vector<face> faces_v;
  std::vector<Node> nodes_v;
  std::map<unsigned int,Node> nodes_m;

};

class Dyna3DWriter;

class k_inp_reader
{

	friend class Dyna3DWriter;

private:
	
  bool match(const std::string& s1, const std::string& s2);	
  std::string getTok(const char* buffer, int start_pos, int len);
  bool isComments(const char* current_line);

 public:
  k_inp_reader();
  //check if this is a key word for the input file	
  int is_inp_keyw(std::string oword);
  //parse the input file to set up the gas system
  int parse_inp(const char* filename);

  void setOutputDir(const std::string& new_outputDir) { outputDir = new_outputDir+"/"; }
  void create_tri();
  void create_tri_initialization();
  void create_tri_shell_startup();
  void create_tri_shells();
  void create_tri_solids();
  struct TriFileContext{
    std::vector<std::string> ptsfiles;
    std::vector<std::string> trifiles;
    std::vector<std::string> names;
  };
  // Pass in a context that can be used to pass information on to create_tri_mpmice().
  // I made this function const, because it helped the optimizer maintain the same
  // performance as when it was part of create_tri().
  void create_tri_write_tris(TriFileContext& context) const;
  void create_tri_mpmice(const TriFileContext& context);
  void create_tri_epilogue();
  
  void dump();
  void cross_prod(const double* v1, const double* v2, double* v3);
  void read_vtcs(std::string ptsfile, int& cnt);
  void read_elems(std::string trifile, int cnt);
  void write_uns();
  void swap_4(char* data);
  void beneathTires();  
  
  ~k_inp_reader();
 
 public: //Data

  ///////////////////////////////////////////////
  std::string outputDir;   // directory for output files

	 double endTime; //The only thing we care about CONTROL_TERMINATION

	std::vector<NodeList*> nodeLists;

    std::vector<SecBeam*> secBeams;
    std::vector<SecShell*> secShells;
    std::vector<SecSolid*> secSolids;
    
    std::vector<ElemSolid*> elemSolids;
    std::vector<ElemShell*> elemShells;
    std::vector<ElemBeam*> elemBeams;
    std::vector<ElementDiscrete*> elementDiscretes;
	std::vector<ElementMass*> elementMass;
	std::vector<ElementSeatbeltAccelerometer*> elementSeatbeltAccelerometers;
	
	std::vector<Node*> nodes;
    std::vector<Part*> parts;

	std::vector<Hourglass*> hourglass;
	std::vector<MatElastic*> matElastics;
	std::vector<MatPlasticKinematic*> matPlasticKinematics;
	std::vector<MatRigid*> matRigids;
	std::vector<MatPiecewiseLinearPlasticity*> matPiecewiseLinearPlasticitys;
	std::vector<MatSpringNonlinearElastic*> matSpringNonlinearElastics;
	std::vector<MatBlatzKORubber*> matBlatzKORubbers;
	std::vector<MatDamperViscous*> matDamperViscous;
	std::vector<MatSpringElastic*> matSpringElastics;
	std::vector<MatHoneycomb*> matHoneycombs;
	std::vector<InitailVelocityGeneration*> initailVelocityGenerations;

	std::vector<LoadBody*> loadBodys;
	std::vector<Curve*> curves;
	std::vector<RigidWallPlanar*> rigidWallPlanars;
	std::vector<ContactOption1Title*> contactOption1Titles;
	std::vector<SetSegment*> setSegments;
	std::vector<SDOrientation*> sDOrientations;
	std::vector<SectionDiscrete*> sectionDiscretes;
	std::vector<AirbagSimpleAirbagModel*> airbagSimpleAirbagModels;
	std::vector<ConstrainedExtraNodes*> constrainedExtraNodes;
	std::vector<ConstrainedJoint*> constrainedJoints;
	std::vector<ConstrainedNodalRigidBody*> constrainedNodalRigidBodies;
	std::vector<ConstrainedRigidBody*> constrainedRigidBodies;
	std::vector<ConstrainedSpotweld*> constrainedSpotwelds;

   std::map<unsigned int,Part*> parts_m;
   std::map<unsigned int,Node*> nodes_m;
   std::map<unsigned int,SecShell*> secShells_m;
   std::map<unsigned int,NodeList*> nodeLists_m;
   
   std::vector<std::vector<int> > elements;
   std::vector<std::vector<float> > vertices;
   
   std::set<unsigned int> midLeaveOut;
   
   int errcnt;
   
   double TCORRECTION, convert_length, convert_mass, convert_time, translate[3];
  double xymin[4][2], xymax[4][2], xmin[3], xmax[3], beneathTireDepth, mat5min[2], mat5max[2];
};

#endif
