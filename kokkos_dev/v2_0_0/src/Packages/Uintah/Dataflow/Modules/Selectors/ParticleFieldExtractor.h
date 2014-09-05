/****************************************
CLASS
    ParticleFieldExtractor

    Visualization control for simulation data that contains
    information on both a regular grid in particle sets.

OVERVIEW TEXT
    This module receives a ParticleGridReader object.  The user
    interface is dynamically created based information provided by the
    ParticleGridReader.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
****************************************/
#ifndef PARTICLEFIELDEXTRACTOR_H
#define PARTICLEFIELDEXTRACTOR_H 1


#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Packages/Uintah/Dataflow/Ports/VectorParticlesPort.h>
#include <Packages/Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Dataflow/Network/Module.h> 
#include <Core/GuiInterface/GuiVar.h> 
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Thread/Runnable.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>


namespace SCIRun {
  class Semaphore;
  class Mutex;
}

namespace Uintah {
  class PFEThread;

using namespace SCIRun;

//class Patch;
class ParticleFieldExtractor;

#define PARTICLE_FIELD_EXTRACTOR_BOGUS_PART_ID -1
  
class ParticleFieldExtractor : public Module { 
  friend class PFEThread;
public: 

  // GROUP: Constructors
  //////////
  ParticleFieldExtractor(GuiContext* ctx);

  // GROUP: Destructors
  //////////
  virtual ~ParticleFieldExtractor(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  //  void tcl_command( TCLArgs&, void* );

  //////////
  // callback taking
  // [in] particleID--an index into the particle set.
  void callback(long64 particleID);

  //////////
  // command from the tcl code
  void tcl_command(GuiArgs& args, void* userdata);
  
protected:
  
private:

  static Mutex module_lock;

  GuiString tcl_status;

  GuiString psVar;
  GuiString pvVar;
  GuiString ptVar;

  GuiInt pNMaterials;


  ArchiveIPort *in;
  VectorParticlesOPort *pvout;
  ScalarParticlesOPort *psout;
  TensorParticlesOPort *ptout;

  
  std::string positionName;
  std::string particleIDs;

  struct VarInfo
  {
     VarInfo()
	: name(""), matls() {}
     VarInfo(std::string name, ConsecutiveRangeSet matls)
	: name(name), matls(matls), wasShown(false) { }
     VarInfo& operator=(const VarInfo& v)
     { name = v.name; matls = v.matls; return *this; }
     std::string name;
     ConsecutiveRangeSet matls;
     bool wasShown;
  };
  std::list<VarInfo> scalarVars;
  std::list<VarInfo> vectorVars;
  std::list<VarInfo> tensorVars;
  std::list<VarInfo> pointVars;
  VarInfo particleIDVar;

  ArchiveHandle archiveH;
  void add_type(string &type_list, const TypeDescription *subtype);
  bool setVars(DataArchive& archive);
  void showVarsForMatls();
  std::string getVarsForMaterials(std::list<VarInfo>& vars,
				  const ConsecutiveRangeSet& matls,
				  bool& needToUpdate);
  void buildData(DataArchive& archive, double time,
		 ScalarParticles*& sp,
		 VectorParticles*& vp,
		 TensorParticles*& tp);

  void addGraphingVars(long64 particleID, const list<VarInfo> &vars,
		       string type);
  int get_matl_from_particleID(long64 particleID);
   
  //  void graph(string varname, vector<string> mat_list, string particleID);

  vector< double > times;
  vector< int > indices;
  vector< string > names;
  vector< const TypeDescription *> types;
  double time;
  int num_materials;

  string vector_to_string(vector< int > data);
  string vector_to_string(vector< string > data);
  string vector_to_string(vector< double > data);
  string vector_to_string(vector< Vector > data, string type);
  string vector_to_string(vector< Matrix3 > data, string type);

  bool is_cached(string name, string& data);
  void cache_value(string where, vector<double>& values, string &data);
  void cache_value(string where, vector<int>& values, string &data);
  void cache_value(string where, vector<Vector>& values);
  void cache_value(string where, vector<Matrix3>& values);
  map< string, string > material_data_list;

  void graph(string varname, vector<string> mat_list,
	     vector<string> type_list, string particleID);
  //  void graph(string, string);

 
};

 class  PFEThread : public Runnable
    {
    public:
      PFEThread( ParticleFieldExtractor *pfe, DataArchive& archive,
		 Patch *patch,
		 ScalarParticles*& sp, VectorParticles*& vp,
		 TensorParticles*& tp, PSet* pset,
		 int scalar_type, bool have_sp,
		 bool have_vp, bool have_tp, bool have_ids,
		 Semaphore *sema, Mutex *smutex,
		 Mutex *vmutex, Mutex *tmutex, Mutex *imutex,
		 GuiInterface* gui):
	pfe(pfe), archive(archive), patch(patch), 
	sp(sp), vp(vp), tp(tp),
	pset(pset), scalar_type(scalar_type), have_sp(have_sp),
	have_vp(have_vp), have_tp(have_tp), have_ids(have_ids), sema(sema),
	smutex(smutex), vmutex(vmutex), tmutex(tmutex), imutex(imutex),
      gui(gui){}
      
      void  run();
    private:
      ParticleFieldExtractor *pfe;
      DataArchive&  archive;
      LevelP level;
      Patch *patch;
      ScalarParticles*& sp;
      VectorParticles*& vp;
      TensorParticles*& tp;
      PSet* pset;
      int scalar_type;
      bool have_sp;
      bool have_vp;
      bool have_tp;
      bool have_ids;
      Semaphore *sema;
      Mutex *smutex;
      Mutex *vmutex;
      Mutex *tmutex;
      Mutex *imutex;
      GuiInterface* gui;
    };
} // End namespace Uintah

#endif
