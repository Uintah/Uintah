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
#include <Packages/Uintah/Core/Datatypes/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/VectorParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/TensorParticlesPort.h>
#include <Dataflow/Network/Module.h> 
#include <Core/GuiInterface/GuiVar.h> 
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <string>
#include <vector>



namespace Uintah {

using namespace SCIRun;

class ParticleFieldExtractor : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  ParticleFieldExtractor(const clString& id); 

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
  void callback(long particleID);

  //////////
  // command from the tcl code
  void tcl_command(TCLArgs& args, void* userdata);
  
protected:
  
private:

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
  void setVars(ArchiveHandle ar);
  void showVarsForMatls();
  std::string getVarsForMaterials(std::list<VarInfo>& vars,
				  const ConsecutiveRangeSet& matls,
				  bool& needToUpdate);
  void buildData(DataArchive& archive, double time,
		 ScalarParticles*& sp,
		 VectorParticles*& vp,
		 TensorParticles*& tp);

  void addGraphingVars(long particleID, const list<VarInfo> &vars,
		       string type);
   
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
  void cache_value(string where, vector<Vector>& values);
  void cache_value(string where, vector<Matrix3>& values);
  map< string, string > material_data_list;

  void graph(string varname, vector<string> mat_list,
	     vector<string> type_list, string particleID);
  //  void graph(clString, clString);
};
} // End namespace Uintah

#endif
