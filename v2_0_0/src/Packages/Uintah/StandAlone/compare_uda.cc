/*
 *  puda.cc: Print out a uintah data archive
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 *  Copyright (C) 2000 U of U
 */

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <math.h>

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)
// NECESSARY FOR LINKING BUT NOT REALLY USED.
SCIRun::Mutex cerrLock( "cerr lock" );

using namespace SCIRun;
using namespace std;
using namespace Uintah;

// serr Vector specialization below
template <class T>
void print(std::ostream& out, const T& t)
{
  out << t;
}

// must override Vector's output in order to use the ostream's precision
void print(std::ostream& out, const SCIRun::Vector& t)
{
  out << "[" << t.x() << ", " << t.y() << ", " << t.z() << "]";
}

// must override Vector's output in order to use the ostream's precision
void print(std::ostream& out, const SCIRun::Point& t)
{
  out << "[" << t.x() << ", " << t.y() << ", " << t.z() << "]";
}

void usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
	cerr << "\nError parsing argument: " << badarg << '\n';
    cerr << "\nUsage: " << progname 
	 << " [options] <archive file 1> <archive file 2>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h[elp]\n";
    cerr << "  -abs_tolerance [double] (allowable abs diff of any numbers)\n";
    cerr << "  -rel_tolerance [double] (allowable rel diff of any numbers)\n";
    cerr << "  -as_warnings (treat tolerance errors as warnings and continue)\n";
    cerr << "  -skip_unknown_types (skip variable comparisons" 
	 << " of unknown types without error)\n";
    cerr << "\nNote: The absolute and relative tolerance tests must both fail\n"
	 << "      for a comparison to fail.\n\n";
    Thread::exitAll(1);
}

// I don't want to have to pass these parameters all around, so
// I've made them file-scope.
string filebase1;
string filebase2;
bool tolerance_as_warnings = false;
bool tolerance_error = false;
bool strict_types = true;

void abort_uncomparable()
{
  cerr << "\nThe uda directories may not be compared.\n";
  Thread::exitAll(5);
}

void tolerance_failure()
{
  if (tolerance_as_warnings) {
    tolerance_error = true;
    cerr << endl;
  }
  else
    Thread::exitAll(2);
}

void displayProblemLocation(const string& var, int matl,
			    const Patch* patch, double time)
{
  cerr << "Time: " << time << endl <<
    "Variable: " << var << endl <<
  "Material: " << matl << endl;
  if (patch != 0)
    cerr << "Patch: " << patch->getID() <<endl;   

}

void displayProblemLocation(const string& var, int matl,
			    const Patch* patch, const Patch* patch2,
			    double time)
{
  cerr << "Time: " << time << " "<<
          "Patch1: " << patch->getID() << " " <<
          "Patch2: " << patch2->getID() << " " <<
          "Material: " << matl << " " <<
          "Variable: " << var << endl;
}

bool compare(double a, double b, double abs_tolerance, double rel_tolerance)
{
  double max_abs = fabs(a);
  if (fabs(b) > max_abs) max_abs = fabs(b);
  if (fabs(a - b) > abs_tolerance) {
    if (max_abs > 0 && (fabs(a-b) / max_abs) > rel_tolerance)
      return false;
    else
      return true;
  }
  else
    return true;
}

bool compare(long64 a, long64 b, double /* abs_tolerance */,
	     double /* rel_tolerance */)
{
  return (a == b); // longs should use an exact comparison
}

bool compare(int a, int b, double /* abs_tolerance */,
	     double /* rel_tolerance */)
{
  return (a == b); // int should use an exact comparison
}

bool compare(Vector a, Vector b, double abs_tolerance, double rel_tolerance)
{
  return compare(a.x(), b.x(), abs_tolerance, rel_tolerance) &&
    compare(a.y(), b.y(), abs_tolerance, rel_tolerance) &&
    compare(a.z(), b.z(), abs_tolerance, rel_tolerance);
}

bool compare(Point a, Point b, double abs_tolerance, double rel_tolerance)
{ return compare(a.asVector(), b.asVector(), abs_tolerance, rel_tolerance); }

bool compare(const Matrix3& a, const Matrix3& b, double abs_tolerance,
	     double rel_tolerance)
{
  for (int i = 1; i <= 3; i++)
    for (int j = 1; j <= 3; j++)
      if (!compare(a(i,j), b(i, j), abs_tolerance, rel_tolerance))
	return false;
  return true;
}


/**********************************************************************
 * MaterialParticleVarData and MaterialParticleData are for comparing
 * ParticleVariables when the patch distributions are different in the
 * different uda's -- p.particleID must be a supplied variable for this
 * to work.
 *********************************************************************/

class MaterialParticleVarData
{
public:
  MaterialParticleVarData()
    : name_(""), particleIDData_(0), patchMap_(0) {}
  MaterialParticleVarData(const string& name)
    : name_(name), particleIDData_(0), patchMap_(0) {}
  
  ~MaterialParticleVarData(); // needs to delete the particleVars_
                              // and patchMap if "p.particleID"

  void setVarName(const string& varName)
  { name_ = varName; }
  
  const string& getName() { return name_; }
  
  // add for each patch
  void add(ParticleVariableBase* pvb, const Patch* patch);

  // gather all patches (vectors) into one
  void gather(ParticleSubset* gatherSubset);

  bool compare(MaterialParticleVarData& data2, int matl,
	       double time1, double time2,
	       double abs_tolerance, double rel_tolerance);

  void createPatchMap();
  
  void setParticleIDData(MaterialParticleVarData* particleIDData)
  {
    particleIDData_ = particleIDData;
    patchMap_ = particleIDData->patchMap_;
  }

  const vector<ParticleVariableBase*>& getParticleVars()
  { return particleVars_; }

  long64 getParticleID(particleIndex index);
  const Patch* getPatch(particleIndex index);
private:
  template <class T> bool 
  compare(MaterialParticleVarData& data2, ParticleVariable<T>* value1,
	  ParticleVariable<T>* value2, int matl,
	  double time1, double time2,
	  double abs_tolerance, double rel_tolerance);
  
  string name_;
  // vector elements each represent a patch -- doesn't matter which
  vector<ParticleVariableBase*> particleVars_;
  vector<ParticleSubset*> subsets_;
  vector<const Patch*> patches_;

  MaterialParticleVarData* particleIDData_;
  map<long64, const Patch*>* patchMap_;
};

class MaterialParticleData
{
public:
  MaterialParticleData()
    : matl_(-999), particleIDs_(0) {}
  MaterialParticleData(int matl)
    : matl_(matl), particleIDs_(0) {}
  MaterialParticleData(const MaterialParticleData& copy)
    : matl_(copy.matl_), vars_(copy.vars_), particleIDs_(copy.particleIDs_) {}

  ~MaterialParticleData() {}

  MaterialParticleVarData& operator[](const string& varName)
  {
    MaterialParticleVarData& result = vars_[varName];
    vars_[varName].setVarName(varName);
    if (varName == "p.particleID") {
      particleIDs_ = &result;
    }
    return result;
  }
      
  void compare(MaterialParticleData& data2, double time1, double time2,
	       double abs_tolerance, double rel_tolerance);
  
  void setMatl(int matl)
  { matl_ = matl; }
private:
  void createPatchMap();
  void gather(ParticleSubset* gatherSubset);
  void sort();
  int matl_;
  map<string, MaterialParticleVarData> vars_;
  MaterialParticleVarData* particleIDs_; // will point to one of vars_
};

MaterialParticleVarData::~MaterialParticleVarData()
{
  vector<ParticleVariableBase*>::iterator iter = particleVars_.begin();
  for ( ; iter != particleVars_.end(); iter++)
  {
    delete *iter;
  }
  if (name_ == "p.particleID")
    delete patchMap_;
}

void MaterialParticleData::createPatchMap()
{
  ASSERT(particleIDs_ != 0); // should check for this before this point
  particleIDs_->createPatchMap();

  map<string, MaterialParticleVarData>::iterator varIter = vars_.begin();
  for ( ; varIter != vars_.end(); varIter++)
  {
    (*varIter).second.setParticleIDData(particleIDs_);
  }
}

void MaterialParticleVarData::createPatchMap()
{
  ASSERT(name_ == "p.particleID");
  if (patchMap_)
    delete patchMap_;
  patchMap_ = scinew map<long64, const Patch*>();
  for (unsigned int patch = 0; patch < particleVars_.size(); patch++) {
    particleIndex count =
      particleVars_[patch]->getParticleSet()->numParticles();
    ParticleVariable<long64>* particleID =
      dynamic_cast< ParticleVariable<long64>* >(particleVars_[patch]);
    if (particleID == 0) {
      cerr << "p.particleID must be a ParticleVariable<long64>\n";
      abort_uncomparable();
    }
    for (int i = 0; i < count; i++) {
      (*patchMap_)[(*particleID)[i]] = patches_[patch];
    }
  }
}

void MaterialParticleData::compare(MaterialParticleData& data2, double time1,
				   double time2, double abs_tolerance,
				   double rel_tolerance)
{
  if (vars_.size() == 0)
    return; // nothing to compare -- all good
  
  // map particle id's to their patches
  createPatchMap(); // also calls setParticleIDData
  data2.createPatchMap();
  
  sort();
  data2.sort();

  if (!particleIDs_->compare(*data2.particleIDs_, matl_, time1, time2,
			     abs_tolerance, rel_tolerance))
  {
    cerr << "ParticleIDs do not match\n";
    abort_uncomparable();
  }
  
  map<string, MaterialParticleVarData>::iterator varIter = vars_.begin();
  map<string, MaterialParticleVarData>::iterator varIter2 =
    data2.vars_.begin();
  for ( ; (varIter != vars_.end()) && (varIter2 != data2.vars_.end()) ;
	varIter++, varIter2++)
  {
    // should catch this earlier -- vars/materials do not match
    ASSERT((*varIter).first == (*varIter2).first); 

    if ((*varIter).first == "p.particleID")
      continue; // already compared
    
    (*varIter).second.compare((*varIter2).second, matl_, time1, time2,
			      abs_tolerance, rel_tolerance);
  }
  // should catch this earlier -- vars/materials do not match
  ASSERT((varIter == vars_.end()) && (varIter2 == data2.vars_.end()));
}

struct ID_Index : public pair<long64, particleIndex>
{
  ID_Index(long64 l, particleIndex i)
    : pair<long64, particleIndex>(l, i) {}
  bool operator<(ID_Index id2)
  { return first < id2.first; }
};

void MaterialParticleData::sort()
{
  // should have made this check earlier -- particleIDs not output
  ASSERT(particleIDs_->getParticleVars().size() != 0);

  vector< ID_Index > idIndices;
  particleIndex base = 0;
  
  for (unsigned int i = 0; i < particleIDs_->getParticleVars().size(); i++) {
    ParticleVariable<long64>* pIDs = dynamic_cast<ParticleVariable<long64>*>(particleIDs_->getParticleVars()[i]);
    if (pIDs == 0) {
      cerr << "p.particleID must be a ParticleVariable<long64>\n";
      abort_uncomparable();
    }

    long64* pID = (long64*)pIDs->getBasePointer();
    ParticleSubset* subset = pIDs->getParticleSubset();
    for (ParticleSubset::iterator iter = subset->begin();
	 iter != subset->end(); iter++) {
      idIndices.push_back(ID_Index(*(pID++), base + *iter));
    }
    base = (particleIndex)idIndices.size();
  }

  // sort by particle id and find out what happens to the particle indices.
  ::sort(idIndices.begin(), idIndices.end());

  vector<particleIndex> subsetIndices(idIndices.size());
  for (particleIndex i = 0; i < (particleIndex)idIndices.size(); i++) {
    ASSERT(subsetIndices[idIndices[i].second] == 0);
    subsetIndices[idIndices[i].second] = i;
  }

  ParticleSet* set = scinew ParticleSet((particleIndex)subsetIndices.size());
  ParticleSubset* subset = scinew ParticleSubset(set, false, matl_, 0,
						 subsetIndices.size());
  for (unsigned int i = 0; i < subsetIndices.size(); i++) {
    subset->addParticle(subsetIndices[i]);
  }
  gather(subset);
}

void MaterialParticleData::gather(ParticleSubset* gatherSubset)
{
  map<string, MaterialParticleVarData>::iterator iter;
  for (iter = vars_.begin(); iter != vars_.end(); iter++)
    (*iter).second.gather(gatherSubset);
}

void MaterialParticleVarData::add(ParticleVariableBase* pvb,
				  const Patch* patch)
{
  particleVars_.push_back(pvb);
  subsets_.push_back(pvb->getParticleSubset());
  patches_.push_back(patch);
}

void MaterialParticleVarData::gather(ParticleSubset* gatherSubset)
{
  ASSERT(particleVars_.size() > 0);
  ParticleVariableBase* pvb = particleVars_[0]->clone();
  pvb->gather(gatherSubset, subsets_, particleVars_, 0);
  particleVars_.clear();
  subsets_.clear();
  patches_.clear();
  add(pvb, 0 /* all patches */);
}

bool MaterialParticleVarData::
compare(MaterialParticleVarData& data2, int matl, double time1, double time2,
	double abs_tolerance, double rel_tolerance)
{
  cerr << "\tVariable: " << name_ << ", comparing via particle ids" << endl;
  ASSERT(particleVars_.size() == 1 && subsets_.size() == 1 &&
	 data2.particleVars_.size() == 1 && data2.subsets_.size() == 1);
  ParticleVariableBase* pvb1 = particleVars_[0];
  ParticleVariableBase* pvb2 = data2.particleVars_[0];

  // type checks should have been made earlier
  ASSERT(pvb1->virtualGetTypeDescription() ==
	 pvb2->virtualGetTypeDescription());

  switch (pvb1->virtualGetTypeDescription()->getSubType()->getType())
  {
  case Uintah::TypeDescription::double_type:
    return compare(data2, dynamic_cast<ParticleVariable<double>*>(pvb1),
		   dynamic_cast<ParticleVariable<double>*>(pvb2), matl,
		   time1, time2, abs_tolerance, rel_tolerance);
  case Uintah::TypeDescription::long64_type:
    return compare(data2, dynamic_cast<ParticleVariable<long64>*>(pvb1),
		   dynamic_cast<ParticleVariable<long64>*>(pvb2), matl,
		   time1, time2, abs_tolerance, rel_tolerance);
  case Uintah::TypeDescription::int_type:
    return compare(data2, dynamic_cast<ParticleVariable<int>*>(pvb1),
		   dynamic_cast<ParticleVariable<int>*>(pvb2), matl,
		   time1, time2, abs_tolerance, rel_tolerance);
  case Uintah::TypeDescription::Point:
    return compare(data2, dynamic_cast<ParticleVariable<Point>*>(pvb1),
		   dynamic_cast<ParticleVariable<Point>*>(pvb2), matl,
		   time1, time2, abs_tolerance, rel_tolerance);
  case Uintah::TypeDescription::Vector:
    return compare(data2, dynamic_cast<ParticleVariable<Vector>*>(pvb1),
		   dynamic_cast<ParticleVariable<Vector>*>(pvb2), matl,
		   time1, time2, abs_tolerance, rel_tolerance);
  case Uintah::TypeDescription::Matrix3:
    return compare(data2, dynamic_cast<ParticleVariable<Matrix3>*>(pvb1),
		   dynamic_cast<ParticleVariable<Matrix3>*>(pvb2), matl,
		   time1, time2, abs_tolerance, rel_tolerance);
  default:
    cerr << "ParticleVariable of unsupported type: " << pvb1->virtualGetTypeDescription()->getName() << '\n';
    Thread::exitAll(-1);
  }
  return 0;
}

template <class T>
bool MaterialParticleVarData::
compare(MaterialParticleVarData& data2, ParticleVariable<T>* value1,
	ParticleVariable<T>* value2, int matl,
	double time1, double /*time2*/, double abs_tolerance,
	double rel_tolerance)
{
  bool passes = true;
  ParticleSubset* pset1 = value1->getParticleSubset();
  ParticleSubset* pset2 = value2->getParticleSubset();
  if (pset1->numParticles() != pset2->numParticles()) {
    cerr << "Inconsistent number of particles.\n";
    displayProblemLocation(name_, matl, 0, time1);    
    cerr << filebase1 << " has " << pset1->numParticles() << " particles.\n";
    cerr << filebase2 << " has " << pset2->numParticles() << " particles.\n";
    abort_uncomparable();
  }

  // Assumes that the particleVariables are in corresponding order --
  // not necessarily by their particle set order.  This is what the
  // sort/gather achieves.
  for (int i = 0; i < pset1->numParticles(); i++) {
    if (!(::compare((*value1)[i], (*value2)[i], abs_tolerance,
		    rel_tolerance))) {
      if (name_ != "p.particleID") {
	ASSERT(getParticleID(i) == data2.getParticleID(i));
      }
      cerr << "DIFFERENCE on particle id= " << getParticleID(i) << endl;
      IntVector origin((int)(getParticleID(i) >> 16) & 0xffff,
		       (int)(getParticleID(i) >> 32) & 0xffff,
		       (int)(getParticleID(i) >> 48) & 0xffff);
      cerr << "(Originating from " << origin << ")\n";
      const Patch* patch1 = getPatch(i);
      const Patch* patch2 = data2.getPatch(i);
      displayProblemLocation(name_, matl, patch1, patch2, time1);  
         
      cerr << filebase1 << ":\n" << (*value1)[i] << endl;
      cerr << filebase2 << ":\n" << (*value2)[i] << endl;
      tolerance_failure();
      passes = false;
    }
  }

  return passes;
}

long64 MaterialParticleVarData::getParticleID(particleIndex index)
{
  ASSERT(particleIDData_ != 0);
  ASSERT(particleIDData_->particleVars_.size() == 1);
  ParticleVariable<long64>* particleIDs =
    dynamic_cast<ParticleVariable<long64>*>(particleIDData_->particleVars_[0]);
  ASSERT(particleIDs != 0);
  
  return (*particleIDs)[index];
}

const Patch* MaterialParticleVarData::getPatch(particleIndex index)
{
  ASSERT(patchMap_ != 0);
  return (*patchMap_)[getParticleID(index)];
}

/*
typedef struct{
  vector<ParticleVariable<double> > pv_double_list;
  vector<ParticleVariable<Point> > pv_point_list;
  vector<ParticleVariable<Vector> > pv_vector_list;
  vector<ParticleVariable<Matrix3> > pv_matrix3_list;
  ParticleVariable<Point> p_x;
} MaterialPatchParticleData;
*/

typedef map<int, MaterialParticleData> MaterialParticleDataMap;

// takes a string and replaces all occurances of old with newch
string replaceChar(string s, char old, char newch) {
  string result;
  for (int i = 0; i<(int)s.size(); i++)
    if (s[i] == old)
      result += newch;
    else
      result += s[i];
  return result;
}


void addParticleData(MaterialParticleDataMap& matlParticleDataMap,
		     DataArchive* da, vector<string> vars,
		     vector<const Uintah::TypeDescription*> types,
		     LevelP level, double time)
{
  Level::const_patchIterator iter;
  for(iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
    const Patch* patch = *iter;
    for(int v=0;v<(int)vars.size();v++){
      std::string var = vars[v];
      const Uintah::TypeDescription* td = types[v];
      const Uintah::TypeDescription* subtype = td->getSubType();
      if (td->getType() == Uintah::TypeDescription::ParticleVariable) {
	ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
	     matlIter != matls.end(); matlIter++){
	  int matl = *matlIter;
	  // Add a new MaterialPatchData for each matl for this next patch.
	  MaterialParticleData& data = matlParticleDataMap[matl];
	  data.setMatl(matl);
	  ParticleVariableBase* pvb = NULL;
	  switch(subtype->getType()){
	  case Uintah::TypeDescription::double_type:
	    pvb = scinew ParticleVariable<double>();
	    break;
	  case Uintah::TypeDescription::long64_type:
	    pvb = scinew ParticleVariable<long64>();
	    break;
	  case Uintah::TypeDescription::int_type:
	    pvb = scinew ParticleVariable<int>();
	    break;
	  case Uintah::TypeDescription::Point:
	    pvb = scinew ParticleVariable<Point>();
	    break;
	  case Uintah::TypeDescription::Vector:
	    pvb = scinew ParticleVariable<Vector>(); 
	    break;
	  case Uintah::TypeDescription::Matrix3:
	    pvb = scinew ParticleVariable<Matrix3>();
	    break;
	  default:
	    cerr << "ParticleVariable of unsupported type: " 
		 << subtype->getName() << '\n';
	    Thread::exitAll(-1);
	  }
	  da->query(*pvb, var, matl, patch, time);
	  data[var].add(pvb, patch); // will add one for each patch
	}
      }
    }
  }
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // template parameter not used in declaring arguments
#endif  

template <class T>
void compareParticles(DataArchive* da1, DataArchive* da2, const string& var,
		      int matl, const Patch* patch1, const Patch* patch2,
		      double time, double time2, double abs_tolerance,
		      double rel_tolerance)
{
  ParticleVariable<T> value1;
  ParticleVariable<T> value2;
  da1->query(value1, var, matl, patch1, time);
  da2->query(value2, var, matl, patch2, time2);

  ParticleSubset* pset1 = value1.getParticleSubset();
  ParticleSubset* pset2 = value2.getParticleSubset();
  if (pset1->numParticles() != pset2->numParticles()) {
    cerr << "Inconsistent number of particles.\n";
    displayProblemLocation(var, matl, patch1, time);    
    cerr << filebase1 << " has " << pset1->numParticles() << " particles.\n";
    cerr << filebase2 << " has " << pset2->numParticles() << " particles.\n";
    abort_uncomparable();
  }
  
  ParticleSubset::iterator iter1 = pset1->begin();
  ParticleSubset::iterator iter2 = pset2->begin();
  
  for ( ; iter1 != pset1->end() && iter2 != pset2->end(); iter1++, iter2++) {
    if (!compare(value1[*iter1], value2[*iter2], abs_tolerance,
		 rel_tolerance)) {
      cerr << "\nValues differ too much.\n";
      displayProblemLocation(var, matl, patch1, time);    
      cerr << filebase1 << ":\n";
      print(cerr, value1[*iter1]);
      cerr << endl << filebase2 << ":\n";
      print(cerr, value2[*iter2]);
      cerr << endl;
      tolerance_failure();
    }
  }

  // this should be true if both sets are the same size
  ASSERT(iter1 == pset1->end() && iter2 == pset2->end());
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif  

/*
template <class Field, class Iterator>
void compareFields(DataArchive* da1, DataArchive* da2, const string& var,
		   int matl, const Patch* patch1, const Patch* patch2,
		   double time, double time2, double abs_tolerance,
		   double rel_tolerance, Iterator iter1, Iterator iter2)
{
  Field value1;
  Field value2;
  da1->query(value1, var, matl, patch1, time);
  da2->query(value2, var, matl, patch2, time2);
  IntVector dx1(value1.getHighIndex() - value1.getLowIndex());
  IntVector dx2(value2.getHighIndex() - value2.getLowIndex());
  if (value1.getHighIndex() != value2.getHighIndex() ||
      value1.getLowIndex() != value2.getLowIndex()) {
    cerr << "Inconsistent variable grid ranges.\n";
    displayProblemLocation(var, matl, patch1, time);    
    cerr << filebase1 << ": " << value1.getLowIndex() << " - " <<
      value1.getHighIndex() << endl;
    cerr << filebase2 << ": " << value2.getLowIndex() << " - " <<
      value2.getHighIndex() << endl;
    abort_uncomparable();
  }
  
  for ( ; !iter1.done() && !iter2.done(); iter1++, iter2++ ) {
    if (!compare(value1[*iter1], value2[*iter2], abs_tolerance,
		 rel_tolerance)) {
      cerr << "\nValues differ too much.\n";
      displayProblemLocation(var, matl, patch1, time);    
      cerr << filebase1 << ":\n" << value1[*iter1] << endl;
      cerr << filebase2 << ":\n" << value2[*iter2] << endl;
      tolerance_failure();
    }
  }

  // this should be true if both sets are the same size
  ASSERT(iter1.done() && iter2.done());
}
*/

class FieldComparator
{
public:
  virtual void
  compareFields(DataArchive* da1, DataArchive* da2, const string& var,
		ConsecutiveRangeSet matls, const Patch* patch,
		const Array3<const Patch*>& patch2Map,
		double time, double time2, double abs_tolerance,
		double rel_tolerance) = 0;

  static FieldComparator*
  makeFieldComparator(const Uintah::TypeDescription* td,
		      const Uintah::TypeDescription* subtype,
		      const Patch* patch);
};

template <class Field, class Iterator>
class SpecificFieldComparator : public FieldComparator
{
public:
  SpecificFieldComparator(Iterator begin)
    : begin_(begin) { }
  virtual ~SpecificFieldComparator() {}
  
  virtual void
  compareFields(DataArchive* da1, DataArchive* da2, const string& var,
		ConsecutiveRangeSet matls, const Patch* patch,
		const Array3<const Patch*>& patch2Map,
		double time, double time2, double abs_tolerance,
		double rel_tolerance);
private:
  Iterator begin_;
};

FieldComparator* FieldComparator::
makeFieldComparator(const Uintah::TypeDescription* td,
		    const Uintah::TypeDescription* subtype, const Patch* patch)
{
  switch(td->getType()){
  case Uintah::TypeDescription::ParticleVariable:
    // Particles handled differently (and previously)
    break;
  case Uintah::TypeDescription::NCVariable: {
    NodeIterator iter = patch->getNodeIterator();
    switch(subtype->getType()){
    case Uintah::TypeDescription::double_type:
      return scinew
	SpecificFieldComparator<NCVariable<double>, NodeIterator>(iter);
    case Uintah::TypeDescription::int_type:
      return scinew
	SpecificFieldComparator<NCVariable<int>, NodeIterator>(iter);
    case Uintah::TypeDescription::Point:
      return scinew
	SpecificFieldComparator<NCVariable<Point>, NodeIterator>(iter);
    case Uintah::TypeDescription::Vector:
      return scinew
	SpecificFieldComparator<NCVariable<Vector>, NodeIterator>(iter);
    case Uintah::TypeDescription::Matrix3:
      return scinew
	SpecificFieldComparator<NCVariable<Matrix3>, NodeIterator>(iter);
    default:
      cerr << "NC Variable of unsupported type: " << subtype->getName() << '\n';
      Thread::exitAll(-1);
    }
  }
  case Uintah::TypeDescription::CCVariable: {
    CellIterator iter = patch->getCellIterator();
    switch(subtype->getType()){
    case Uintah::TypeDescription::double_type:
      return scinew
	SpecificFieldComparator<CCVariable<double>, CellIterator>(iter);
    case Uintah::TypeDescription::int_type:
      return scinew
	SpecificFieldComparator<CCVariable<int>, CellIterator>(iter);
    case Uintah::TypeDescription::Point:
      return scinew
	SpecificFieldComparator<CCVariable<Point>, CellIterator>(iter);
    case Uintah::TypeDescription::Vector:
      return scinew
	SpecificFieldComparator<CCVariable<Vector>, CellIterator>(iter);
    case Uintah::TypeDescription::Matrix3:
      return scinew
	SpecificFieldComparator<CCVariable<Matrix3>, CellIterator>(iter);
    default:
      cerr << "CC Variable of unsupported type: " << subtype->getName() << '\n';
      Thread::exitAll(-1);
    }
  }
  case Uintah::TypeDescription::SFCXVariable: {
    CellIterator iter = patch->getSFCXIterator();
    switch(subtype->getType()){
    case Uintah::TypeDescription::double_type:
      return scinew
	SpecificFieldComparator<SFCXVariable<double>, CellIterator>(iter);
    case Uintah::TypeDescription::int_type:
      return scinew
	SpecificFieldComparator<SFCXVariable<int>, CellIterator>(iter);
    case Uintah::TypeDescription::Point:
      return scinew
	SpecificFieldComparator<SFCXVariable<Point>, CellIterator>(iter);
    case Uintah::TypeDescription::Vector:
      return scinew
	SpecificFieldComparator<SFCXVariable<Vector>, CellIterator>(iter);
    case Uintah::TypeDescription::Matrix3:
      return scinew
	SpecificFieldComparator<SFCXVariable<Matrix3>, CellIterator>(iter);
    default:
      cerr << "SFCX Variable of unsupported type: " << subtype->getName() << '\n';
      Thread::exitAll(-1);
    }
  }
  case Uintah::TypeDescription::SFCYVariable: {
    CellIterator iter = patch->getSFCYIterator();
    switch(subtype->getType()){
    case Uintah::TypeDescription::double_type:
      return scinew
	SpecificFieldComparator<SFCYVariable<double>, CellIterator>(iter);
    case Uintah::TypeDescription::int_type:
      return scinew
	SpecificFieldComparator<SFCYVariable<int>, CellIterator>(iter);
    case Uintah::TypeDescription::Point:
      return scinew
	SpecificFieldComparator<SFCYVariable<Point>, CellIterator>(iter);
    case Uintah::TypeDescription::Vector:
      return scinew
	SpecificFieldComparator<SFCYVariable<Vector>, CellIterator>(iter);
    case Uintah::TypeDescription::Matrix3:
      return scinew
	SpecificFieldComparator<SFCYVariable<Matrix3>, CellIterator>(iter);
    default:
      cerr << "SFCY Variable of unsupported type: " << subtype->getName() << '\n';
      Thread::exitAll(-1);
    }
  }
  case Uintah::TypeDescription::SFCZVariable: {
    CellIterator iter = patch->getSFCZIterator();
    switch(subtype->getType()){
    case Uintah::TypeDescription::double_type:
      return scinew
	SpecificFieldComparator<SFCZVariable<double>, CellIterator>(iter);
    case Uintah::TypeDescription::int_type:
      return scinew
	SpecificFieldComparator<SFCZVariable<int>, CellIterator>(iter);
    case Uintah::TypeDescription::Point:
      return scinew
	SpecificFieldComparator<SFCZVariable<Point>, CellIterator>(iter);
    case Uintah::TypeDescription::Vector:
      return scinew
	SpecificFieldComparator<SFCZVariable<Vector>, CellIterator>(iter);
    case Uintah::TypeDescription::Matrix3:
      return scinew
	SpecificFieldComparator<SFCZVariable<Matrix3>, CellIterator>(iter);
    default:
      cerr << "SFCZ Variable of unsupported type: " << subtype->getName() << '\n';
      Thread::exitAll(-1);
    }
  }
  default:
    cerr << "Variable of unsupported type: " << td->getName() << '\n';
    Thread::exitAll(-1);
  }
  return 0;
}

template <class Field, class Iterator>
void SpecificFieldComparator<Field, Iterator>::
compareFields(DataArchive* da1, DataArchive* da2, const string& var,
	      ConsecutiveRangeSet matls, const Patch* patch,
	      const Array3<const Patch*>& patch2Map,
	      double time1, double time2, double abs_tolerance,
	      double rel_tolerance)
{
  Field* pField2;
  bool firstMatl = true;
  
  for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
		matlIter != matls.end(); matlIter++){
    int matl = *matlIter;
    Field field;
    da1->query(field, var, matl, patch, time1);

    map<const Patch*, Field*> patch2FieldMap;
    typename map<const Patch*, Field*>::iterator findIter;
    for (Iterator iter = begin_ ; !iter.done(); iter++ ) {
      const Patch* patch2 = patch2Map[*iter];
      findIter = patch2FieldMap.find(patch2);
      if (findIter == patch2FieldMap.end()) {
	if (firstMatl) { // check only needs to be made the first round
	  ConsecutiveRangeSet matls2 = da2->queryMaterials(var, patch2, time2);
	  ASSERT(matls == matls2); // check should have been made previously
	}
	pField2 = scinew Field();
	patch2FieldMap[patch2] = pField2;
	da2->query(*pField2, var, matl, patch2, time2);
      }
      else {
	pField2 = (*findIter).second;
      }
      if (!compare(field[*iter], (*pField2)[*iter], abs_tolerance,
		   rel_tolerance)) {
	cerr << "DIFFERENCE " << *iter << "  ";
	displayProblemLocation(var, matl, patch, patch2, time1);
 
       cerr << filebase1 << " (1)\t\t" << filebase2 << " (2)"<<endl;
       print(cerr, field[*iter]);
       cerr << "\t\t";
       print(cerr, (*pField2)[*iter]);
       cerr << endl;

	tolerance_failure();
      }
    }

    typename map<const Patch*, Field*>::iterator iter = patch2FieldMap.begin();
    for ( ; iter != patch2FieldMap.end(); iter++) {
      delete (*iter).second;
    }
    firstMatl = false;
  }
}


// map nodes to their owning patch in a level.
// Nodes are used because I am assuming that whoever owns the node at
// that index also owns the cell, or whatever face at that same index.
// The same doesn't work if you used cells because nodes can go beyond
// cells (when there is no neighbor on the greater side).
void buildPatchMap(LevelP level, const string& filebase,
		   Array3<const Patch*>& patchMap, double time)
{
  const PatchSet* allPatches = level->allPatches();
  const PatchSubset* patches = allPatches->getUnion();
  if (patches->size() == 0)
    return;

  IntVector low = patches->get(0)->getNodeLowIndex();
  IntVector high = patches->get(0)->getNodeHighIndex();

  for (int i = 1; i < patches->size(); i++) {
    low = Min(low, patches->get(i)->getNodeLowIndex());
    high = Max(high, patches->get(i)->getNodeHighIndex());
  }
  
  patchMap.resize(low, high);
  patchMap.initialize(0);

  Level::const_patchIterator iter;
  for(iter = level->patchesBegin();
      iter != level->patchesEnd(); iter++) {
    const Patch* patch = *iter;
    ASSERT(Min(patch->getNodeLowIndex(), low) == low);
    ASSERT(Max(patch->getNodeHighIndex(), high) == high);
    patchMap.rewindow(patch->getNodeLowIndex(),
		      patch->getNodeHighIndex());
    for (Array3<const Patch*>::iterator iter = patchMap.begin();
	 iter != patchMap.end(); iter++) {
      if (*iter != 0) {
	cerr << "Patches " << patch->getID() << " and " << (*iter)->getID()
	     << " overlap on the same file at time " << time
	     << " in " << filebase << endl;
	cerr << "Cannot be handled\n";
	abort_uncomparable();
      }
      else
	*iter = patch;
    }
  }
  patchMap.rewindow(low, high);
}


int
main(int argc, char** argv)
{
  double rel_tolerance = 1e-6; // Default 
  double abs_tolerance = 1e-9; //   values...

  // Parse Args:
  for( int i = 1; i < argc; i++ ) {
    string s = argv[i];
    if(s == "-abs_tolerance"){
      if (++i == argc)
	usage("-abs_tolerance, no value given", argv[0]);
      else
	abs_tolerance = atof(argv[i]);
    }
    else if(s == "-rel_tolerance"){
      if (++i == argc)
	usage("-rel_tolerance, no value given", argv[0]);
      else
	rel_tolerance = atof(argv[i]);
    }
    else if(s == "-as_warnings") {
      tolerance_as_warnings = true;
    }
    else if(s == "-skip_unknown_types") {
      strict_types = false;
    }
    else if(s[0] == '-' && s[1] == 'h' ) { // lazy check for -h[elp] option
      usage( "", argv[0] );
    }
    else {
      if (filebase1 != "") {
	if (filebase2 != "")
	  usage(s, argv[0]);
	else
	  filebase2 = argv[i];
      }
      else
	filebase1 = argv[i];
    }
  }

  if( filebase2 == "" ){
    cerr << "\nYou must specify two archive directories.\n";
    usage("", argv[0]);
  }

  cerr << "Using absolute tolerance: " << abs_tolerance << endl;
  cerr << "Using relative tolerance: " << rel_tolerance << endl;
  
  if (rel_tolerance <= 0) {
    cerr << "Must have a positive value rel_tolerance.\n";
    Thread::exitAll(1);
  }

  int digits_precision = (int)ceil(-log10(rel_tolerance)) + 1;
  cerr << setprecision(digits_precision);
  cout << setprecision(digits_precision);

  try {
    DataArchive* da1 = scinew DataArchive(filebase1);
    DataArchive* da2 = scinew DataArchive(filebase2);

    vector<string> vars;    
    vector<const Uintah::TypeDescription*> types;
    vector< pair<string, const Uintah::TypeDescription*> > vartypes1;
    vector<string> vars2;
    vector<const Uintah::TypeDescription*> types2;
    vector< pair<string, const Uintah::TypeDescription*> > vartypes2;    
    da1->queryVariables(vars, types);
    
    ASSERTEQ(vars.size(), types.size());
    da2->queryVariables(vars2, types2);
    ASSERTEQ(vars2.size(), types2.size());
    
    if (vars.size() != vars2.size()) {
      cerr << filebase1 << " has " << vars.size() << " variables\n";
      cerr << filebase2 << " has " << vars2.size() << " variables\n";
      abort_uncomparable();
    }

    vartypes1.resize(vars.size());
    vartypes2.resize(vars.size());
    for (unsigned int i = 0; i < vars.size(); i++) {
      vartypes1[i] = make_pair(vars[i], types[i]);
      vartypes2[i] = make_pair(vars2[i], types2[i]);      
    }
    // sort vars so uda's can be compared if their index files have
    // different orders of variables.
    // Assuming that there are no duplicates in the var names, these will
    // sort alphabetically by varname.
    sort(vartypes1.begin(), vartypes1.end());
    sort(vartypes2.begin(), vartypes2.end());    
    for (unsigned int i = 0; i < vars.size(); i++) {
      vars[i] = vartypes1[i].first; types[i] = vartypes1[i].second;
      vars2[i] = vartypes2[i].first; types2[i] = vartypes2[i].second;      
    }    
    
    for (unsigned int i = 0; i < vars.size(); i++) {
      if (vars[i] != vars2[i]) {
	cerr << "Variable " << vars[i] << " in " << filebase1
	     << " does not match\n";
	cerr << "variable " << vars2[i] << " in " << filebase2 << endl;
	abort_uncomparable();
      }
      
      if (types[i] != types2[i]) {
	cerr << "Variable " << vars[i]
	     << " does not have the same type in both uda directories.\n";
	cerr << "In " << filebase1 << " its type is " << types[i]->getName()
	     << endl;
	cerr << "In " << filebase2 << " its type is " << types2[i]->getName()
	     << endl;
	abort_uncomparable();
      }	
    }
      
    vector<int> index;
    vector<double> times;
    vector<int> index2;
    vector<double> times2;
    da1->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    da2->queryTimesteps(index2, times2);
    ASSERTEQ(index2.size(), times2.size());

    for(unsigned long t = 0; t < times.size() && t < times2.size(); t++){
      if (!compare(times[t], times2[t], abs_tolerance, rel_tolerance)) {
	cerr << "Timestep at time " << times[t] << " in " << filebase1
	     << " does not match\n";
	cerr << "timestep at time " << times2[t] << " in " << filebase2
	     << " within the allowable tolerance.\n";
	abort_uncomparable();
      }
      
      double time1 = times[t];
      double time2 = times2[t];
      cerr << "time = " << time1 << "\n";
      GridP grid = da1->queryGrid(time1);
      GridP grid2 = da2->queryGrid(times2[t]);

      if (grid->numLevels() != grid2->numLevels()) {
	cerr << "Grid at time " << time1 << " in " << filebase1
	     << " has " << grid->numLevels() << " levels.\n";
	cerr << "Grid at time " << time2 << " in " << filebase2
	     << " has " << grid->numLevels() << " levels.\n";
	abort_uncomparable();
      }

      // do some consistency checking first
      bool hasParticleIDs = false;
      bool hasParticleData = false;
      for(int v=0;v<(int)vars.size();v++){
	std::string var = vars[v];
	if (var == "p.particleID")
	  hasParticleIDs = true;
	if (types[v]->getType() == Uintah::TypeDescription::ParticleVariable)
	  hasParticleData = true;

	for(int l=0;l<grid->numLevels();l++){
	  LevelP level = grid->getLevel(l);
	  LevelP level2 = grid2->getLevel(l);
	  ConsecutiveRangeSet matls;
	  bool first = true;
	  Level::const_patchIterator iter;
	  for(iter = level->patchesBegin();
	      iter != level->patchesEnd(); iter++) {
	    const Patch* patch = *iter;
	    if (first) {
	      matls = da1->queryMaterials(var, patch, time1);
	    }
	    else if (matls != da1->queryMaterials(var, patch, time1)) {
	      cerr << "The material set is not consistent for variable "
		   << var << " across patches at time " << time1 << endl;
	      cerr << "Previously was: " << matls << endl;
	      cerr << "But on patch " << patch->getID() << ": " <<
		da1->queryMaterials(var, patch, time1) << endl;
	      abort_uncomparable();
	    }
	    first = false;
	  }
	  ASSERT(!first); /* More serious problems would show up if this
			     assertion would fail */
	  for(iter = level2->patchesBegin();
	      iter != level2->patchesEnd(); iter++) {
	    const Patch* patch = *iter;
	    if (matls != da2->queryMaterials(var, patch, time2)) {
	      cerr << "Inconsistent material sets for variable "
		   << var << " on patch2 = " << patch->getID()
		   << ", time " << time1 << endl;
	      cerr << filebase1 << " (1) has material set: " << matls << ".\n";
	      cerr << filebase2 << " (2) has material set: "
		   << da2->queryMaterials(var, patch, time2) << ".\n";
	      abort_uncomparable();  
	    }
	  }
	}
      }

      /* COMPARE PARTICLE VARIABLES */
      if (hasParticleData && !hasParticleIDs) {
	// Compare particle variables without p.particleID -- patches
	// must be consistent.
	cerr << "Particle data exists without p.particleID output.\n";
	cerr << "There must be patch consistency in order to do this comparison.\n";
	cerr << "In order to make a comparison between udas with different\n"
	     << "number or distribution of patches, you must either output\n"
	     << "p.particleID or don't output any particle variables at all.\n";
	cerr << endl;
	
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  const Uintah::TypeDescription* td = types[v];
	  const Uintah::TypeDescription* subtype = td->getSubType();
	  cerr << "\tVariable: " << var << ", type " << td->getName() << "\n";
	  if (td->getName() == string("-- unknown type --")) {
	    cerr << "\t\tParticleVariable of unknown type";
	    if (strict_types) {
	      cerr << ".\nQuitting.\n";
	      Thread::exitAll(-1);
	    }
	    cerr << "; skipping comparison...\n";
	    continue;
	  }
	  for(int l=0;l<grid->numLevels();l++){
	    LevelP level = grid->getLevel(l);
	    LevelP level2 = grid2->getLevel(l);
	    if (level->numPatches() != level2->numPatches()) {
	      cerr << "Inconsistent number of patches on level " << l <<
		" at time " << time1 << ":" << endl;
	      cerr << filebase1 << " has " << level->numPatches()
		   << " patches.\n";
	      cerr << filebase2 << " has " << level2->numPatches()
		   << " patches.\n";
	      abort_uncomparable();
	    }

	  
	    Level::const_patchIterator iter2 = level2->patchesBegin();
	    for(Level::const_patchIterator iter = level->patchesBegin();
		iter != level->patchesEnd(); iter++, iter2++){
	      const Patch* patch = *iter;
	      const Patch* patch2 = *iter2;

	      if (patch->getID() != patch2->getID()) {
		cerr << "Inconsistent patch ids on level " << l
		     << " at time " << time1 << endl;
		cerr << filebase1 << " has patch id " << patch->getID()
		     << " where\n";
		cerr << filebase2 << " has patch id " << patch2->getID() << endl;
		abort_uncomparable();  
	      }
	    
	      cerr << "\t\tPatch: " << patch->getID() << "\n";

	      if (!compare(patch->getBox().lower(), patch2->getBox().lower(),
			   abs_tolerance, rel_tolerance) ||
		  !compare(patch->getBox().upper(), patch2->getBox().upper(),
			   abs_tolerance, rel_tolerance)) {
		cerr << "Inconsistent patch bounds on patch " << patch->getID()
		     << " at time " << time1 << endl;
		cerr << filebase1 << " has bounds " << patch->getBox().lower()
		     << " - " << patch->getBox().upper() << ".\n";
		cerr << filebase2 << " has bounds " << patch2->getBox().lower()
		     << " - " << patch2->getBox().upper() << ".\n";
		cerr << "Difference is: " << patch->getBox().lower() - patch2->getBox().lower() << " - " << patch->getBox().upper() - patch2->getBox().upper() << endl;
		abort_uncomparable();  
	      }

	      ConsecutiveRangeSet matls = da1->queryMaterials(var, patch,
							      time1);
	      ConsecutiveRangeSet matls2 = da2->queryMaterials(var, patch2,
							       time2);
	      ASSERT(matls == matls2); // should have already been checked
	      // loop over materials
	      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		  matlIter != matls.end(); matlIter++){
		int matl = *matlIter;
		cerr << "\t\t\tMaterial: " << matl << "\n";
		if (td->getType() == Uintah::TypeDescription::ParticleVariable) {
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    compareParticles<double>(da1, da2, var, matl, patch, patch2,
					     time1, time2, abs_tolerance, rel_tolerance);
		    break;
		  case Uintah::TypeDescription::Point:
		    compareParticles<Point>(da1, da2, var, matl, patch, patch2,
					    time1, time2, abs_tolerance, rel_tolerance);
		    break;
		  case Uintah::TypeDescription::Vector:
		    compareParticles<Vector>(da1, da2, var, matl, patch, patch2,
					     time1, time2, abs_tolerance, rel_tolerance);
		    break;
		  case Uintah::TypeDescription::Matrix3:
		    compareParticles<Matrix3>(da1, da2, var, matl, patch, patch2,
					      time1, time2, abs_tolerance, rel_tolerance);
		    break;
		  default:
		    cerr << "ParticleVariable of unsupported type: " << subtype->getType() << '\n';
		    Thread::exitAll(-1);
		  }
		}
	      }
	    }
	  }
	}
      }
      else if (hasParticleIDs) {
	// Compare Particle variables with p.particleID -- patches don't
	// need to be cosistent.  It will gather and sort the particles
	// so they can be compared in particleID order.
	for(int l=0;l<grid->numLevels();l++){
	  LevelP level = grid->getLevel(l);
	  LevelP level2 = grid2->getLevel(l);
	  MaterialParticleDataMap matlParticleDataMap1;
	  MaterialParticleDataMap matlParticleDataMap2;
	  addParticleData(matlParticleDataMap1, da1, vars, types, level,
			  time1);
	  addParticleData(matlParticleDataMap2, da2, vars2, types2, level2,
			  time2);
	  MaterialParticleDataMap::iterator matlIter;
	  MaterialParticleDataMap::iterator matlIter2;
	  
	  matlIter = matlParticleDataMap1.begin();
	  matlIter2 = matlParticleDataMap2.begin();
	  for (; (matlIter != matlParticleDataMap1.end()) &&
		 (matlIter2 != matlParticleDataMap2.end());
	       matlIter++, matlIter2++) {
	    // This assert should already have been check above whan comparing
	    // material sets.
	    ASSERT((*matlIter).first == (*matlIter).first);
	    (*matlIter).second.compare((*matlIter2).second, time1, time2,
				       abs_tolerance, rel_tolerance);
	  }
	  // This assert should already have been check above whan comparing
	  // material sets.
	  ASSERT(matlIter == matlParticleDataMap1.end() &&
		 matlIter2 == matlParticleDataMap2.end());
	}
      }

	
      for(int v=0;v<(int)vars.size();v++){
	std::string var = vars[v];
	const Uintah::TypeDescription* td = types[v];
	const Uintah::TypeDescription* subtype = td->getSubType();
	if (td->getType() == Uintah::TypeDescription::ParticleVariable)
	  continue;
	cerr << "\tVariable: " << var << ", type " << td->getName() << "\n";
	if (td->getName() == string("-- unknown type --")) {
	  cerr << "\t\tParticleVariable of unknown type";
	  if (strict_types) {
	    cerr << ".\nQuitting.\n";
	    Thread::exitAll(-1);
	  }
	  cerr << "; skipping comparison...\n";
	  continue;
	}
	
	for(int l=0;l<grid->numLevels();l++){
	  LevelP level = grid->getLevel(l);
	  LevelP level2 = grid2->getLevel(l);
	  
	  // map nodes to patches in level and level2 respectively
	  Array3<const Patch*> patchMap;
	  Array3<const Patch*> patch2Map;
	  
	  buildPatchMap(level, filebase1, patchMap, time1);
	  buildPatchMap(level2, filebase2, patch2Map, time2);
	  
	  if (patchMap.getLowIndex() != patch2Map.getLowIndex() ||
	      patchMap.getHighIndex() != patch2Map.getHighIndex()) {
	    cerr << "Inconsistent patch coverage on level " << l
		 << " at time " << time1 << endl;
	    cerr << "On " << filebase1 << "\n"
		 << "\tRange: " << patchMap.getLowIndex() << " - "
		 << patchMap.getHighIndex() << endl;
	    cerr << "On " << filebase2 << "\n"
		 << "\tRange: " << patch2Map.getLowIndex() << " - "
		 << patch2Map.getHighIndex() << endl;	    
	    abort_uncomparable();
	  }
	  
	  for (Array3<const Patch*>::iterator nodePatchIter = patchMap.begin();
	       nodePatchIter != patchMap.end(); nodePatchIter++) {
	    IntVector index = nodePatchIter.getIndex();
	    if ((patchMap[index] == 0 && patch2Map[index] != 0) ||
		(patch2Map[index] == 0 && patchMap[index] != 0)) {
	      cerr << "Inconsistent patch coverage on level " << l
		   << " at time " << time1 << endl;
	      if (patchMap[index] != 0) {
		cerr << index << " is covered by " << filebase1 << endl
		     << " and not " << filebase2 << endl;
	      }
	      else {
		cerr << index << " is covered by " << filebase2 << endl
		     << " and not " << filebase1 << endl;
	      }
	      abort_uncomparable();
	    }
	  }
	  
	  Level::const_patchIterator iter;
	  
	  for(iter = level->patchesBegin();
	      iter != level->patchesEnd(); iter++) {
	    const Patch* patch = *iter;
 
	    ConsecutiveRangeSet matls = da1->queryMaterials(var, patch, time1);

	    FieldComparator* comparator =
	      FieldComparator::makeFieldComparator(td, subtype, patch);
	    if (comparator != 0) {
	      comparator->compareFields(da1, da2, var, matls, patch,
					patch2Map, time1, time2,
					abs_tolerance, rel_tolerance);
	      delete comparator;
	    }
	  }
	}
      }
    }
    if (times.size() != times2.size()) {
      cerr << endl;
      cerr << filebase1 << " has " << times.size() << " timesteps\n";
      cerr << filebase2 << " has " << times2.size() << " timesteps\n";
      abort_uncomparable();
    }
    delete da1;
    delete da2;
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }

  if (tolerance_error) {
    cerr << "\nComparison did NOT fully pass.\n";
    Thread::exitAll(2);
  }
  else
    cerr << "\nComparison fully passed!\n";

  return 0;
}
