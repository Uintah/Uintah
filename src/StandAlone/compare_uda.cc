/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  compare_uda.cc: compare results of 2 uintah data archive
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 */

#include <Core/DataArchive/DataArchive.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/MinMax.h>
#include <Core/OS/Dir.h>
#include <Core/Util/FileUtils.h>
#include <Core/Util/ProgressiveWarning.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <iostream>

#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace Uintah;

// serr Vector specialization below
template <class T>
void
print(std::ostream& out, const T& t)
{
  out << t;
}

// must override Vector's output in order to use the ostream's precision
void
print(std::ostream& out, const Uintah::Vector& t)
{
  out << "[" << t.x() << ", " << t.y() << ", " << t.z() << "]";
}

// must override Vector's output in order to use the ostream's precision
void
print(std::ostream& out, const Uintah::Point& t)
{
  out << "[" << t.x() << ", " << t.y() << ", " << t.z() << "]";
}

void
usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "")
    cerr << "\nError parsing argument: " << badarg << '\n';
  cerr << "\nUsage: " << progname
       << " [options] <UDA archive directory 1> <UDA archive directory 2>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -h[elp]\n";
  cerr << "  -abs_tolerance [double]  (Allowable absolute difference of any number, default: 1e-9)\n";
  cerr << "  -rel_tolerance [double]  (Allowable relative difference of any number, default: 1e-6)\n";
  cerr << "  -exact                   (Perform an exact comparison, absolute/relative tolerance = 0)\n";
  cerr << "  -levels     [int int]    (Optional:  level index for uda 1 and uda 2)\n";
  cerr << "  -as_warnings             (Treat tolerance errors as warnings and continue)\n";
  cerr << "  -concise                 (With '-as_warnings', only print first incidence of error per var.)\n";
  cerr << "  -skip_unknown_types      (Skip variable comparisons of unknown types without error)\n";
  cerr << "  -ignoreVariable [string] (Skip this variable)\n";
  cerr << "  -dont_sort               (Don't sort the variable names before comparing them)";
  cerr << "\nNote: The absolute and relative tolerance tests must both fail\n"
       << "      for a comparison to fail.\n\n";
  cerr << "  Exit values:\n";
  cerr << "    -1:      Comparison failed, variable type not supported.\n";
  cerr << "     0:      Comparison passed.\n";
  cerr << "     1:      Error in input parameters.\n"; 
  cerr << "     2:      Comparison failed, tolerances exceeded. \n";
  cerr << "     5:      The uda directories may not be compared.\n";

  
  Parallel::exitAll(1);
}

// I don't want to have to pass these parameters all around, so
// I've made them file-scope.
string d_filebase1;
string d_filebase2;
bool d_tolerance_as_warnings = false;
bool d_tolerance_error       = false;
bool d_concise               = false; // If true (and d_tolerance_error), only print 1st error per var.
bool d_strict_types          = true;

void
abort_uncomparable(std::ostringstream& warn)
{
  cerr << "\n_______________ERROR:compare_uda___________________\n";
  
  int error = -9;
  if (d_tolerance_error){
    cerr << " The differences have exceeded the tolerances to the point that the udas can no longer be compared.\n";
    cerr << " Now exiting (2).\n";
    error = 2;
  } else{
  
    cerr << warn.str();
    cerr << "\n  The uda directories may not be compared.\n";
    cerr << " Now exiting (5).\n";
    error = 5;
  }
  cerr << "\n_______________ERROR:compare_uda___________________\n";
  Parallel::exitAll(error);
}

void
tolerance_failure()
{
  if (d_tolerance_as_warnings) {
    d_tolerance_error = true;
    cerr << endl;
  }
  else
    Parallel::exitAll(2);
}

void
displayProblemLocation( std::ostream& out,
                        const string& var,
                        int matl,
                        const Patch* patch,
                        double time )
{
  out << "Time: "     << time << "\n"
       << "Variable: " << var  << "\n"
       << "Material: " << matl << "\n";
  if (patch != 0) {
    out << "Patch: " << patch->getID() << "\n";
  }
}

void
displayProblemLocation( std::ostream& out,
                        const string& var,
                        int matl,
                        const Patch* patch,
                        const Patch* patch2,
                        double time )
{
  out << "Time: " << time << " "<<
    "Level: " <<patch->getLevel()->getIndex() << " " <<
    "Patch1: " << patch->getID() << " " <<
    "Patch2: " << patch2->getID() << " " <<
    "Material: " << matl << " " <<
    "Variable: " << var << endl;
}

bool
compare( double a, double b, double abs_tolerance, double rel_tolerance )
{
  // Return false only if BOTH absolute and relative comparisons fail.

  if(std::isnan(a) || std::isnan(b)){
    return false;
  }

  double max_abs = fabs(a);
  if (fabs(b) > max_abs){
    max_abs = fabs(b);
  }
  if (fabs(a - b) > abs_tolerance) {
    if (max_abs > 0 && (fabs(a-b) / max_abs) > rel_tolerance){
      return false;
    }else{
      return true;
    }
  }else{
    return true;
  }
}

bool
compare( float a, float b, double abs_tolerance, double rel_tolerance )
{
  if(std::isnan(a) || std::isnan(b)){
    return false;
  }

  return compare((double)a, (double)b, abs_tolerance, rel_tolerance);
}

bool
compare( long64 a,
         long64 b,
         double /* abs_tolerance */,
         double /* rel_tolerance */)
{
  if(std::isnan(a) || std::isnan(b)){
    return false;
  }

  return (a == b); // longs should use an exact comparison
}

bool
compare( int a, int b,
         double /* abs_tolerance */,
         double /* rel_tolerance */ )
{
  if(std::isnan(a) || std::isnan(b)){
    return false;
  }

  return (a == b); // int should use an exact comparison
}

bool
compare( Vector a, Vector b, double abs_tolerance, double rel_tolerance )
{
  if(std::isnan(a.length()) || std::isnan(b.length())){
    return false;
  }

  return compare(a.x(), b.x(), abs_tolerance, rel_tolerance) &&
         compare(a.y(), b.y(), abs_tolerance, rel_tolerance) &&
         compare(a.z(), b.z(), abs_tolerance, rel_tolerance);
}

bool
compare( IntVector a, IntVector b, double abs_tolerance, double rel_tolerance )
{
//  if(std::isnan(a.length()) || std::isnan(b.length())){
  if(std::isnan(a.x()*a.x()+a.y()*a.y()+a.z()*a.z()) || 
     std::isnan(b.x()*b.x()+b.y()*b.y()+b.z()*b.z())){
    return false;
  }

  return compare(a.x(), b.x(), abs_tolerance, rel_tolerance) &&
         compare(a.y(), b.y(), abs_tolerance, rel_tolerance) &&
         compare(a.z(), b.z(), abs_tolerance, rel_tolerance);
}

bool
compare( Point a, Point b, double abs_tolerance, double rel_tolerance )
{
  return compare(a.asVector(), b.asVector(), abs_tolerance, rel_tolerance);
}


bool
compare( Stencil7& a, Stencil7& b, double abs_tolerance, double rel_tolerance )
{
  return compare(a.p, b.p, abs_tolerance, rel_tolerance)  &&
         compare(a.n, b.n, abs_tolerance, rel_tolerance)  &&
         compare(a.s, b.s, abs_tolerance, rel_tolerance)  &&
         compare(a.e, b.e, abs_tolerance, rel_tolerance)  &&
         compare(a.w, b.w, abs_tolerance, rel_tolerance)  &&
         compare(a.t, b.t, abs_tolerance, rel_tolerance)  &&
         compare(a.b, b.b, abs_tolerance, rel_tolerance);
}

bool
compare( const Matrix3 & a,
         const Matrix3 & b,
         double abs_tolerance,
         double rel_tolerance )
{
  if(std::isnan(a.Norm()) || std::isnan(b.Norm())){
    return false;
  }

  // for (int i = 0; i < 3; i++)
  //   for (int j = 0; j < 3; j++)
  //     if (!compare(a(i,j), b(i, j), abs_tolerance, rel_tolerance))
  // Comparing element by element is overly sensitive to code changes
  // The following is a hopefully more informative metric of agreement
  if( !compare(a.Norm(), b.Norm(), abs_tolerance, rel_tolerance) ){
    return false;
  }
  else {
    return true;
  }
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
    : d_name(""), d_particleIDData(0), d_patchMap(0) {}
  MaterialParticleVarData(const string& name)
    : d_name(name), d_particleIDData(0), d_patchMap(0) {}

  ~MaterialParticleVarData(); // needs to delete the d_particleVars
                              // and patchMap if "p.particleID"

  void setVarName(const string& varName)
  { d_name = varName; }

  const string& getName() { return d_name; }

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
    d_particleIDData = particleIDData;
    d_patchMap = particleIDData->d_patchMap;
  }

  const vector<ParticleVariableBase*>& getParticleVars()
  { return d_particleVars; }

  long64 getParticleID(particleIndex index);
  const Patch* getPatch(particleIndex index);
private:
  template <class T> bool
  compare(MaterialParticleVarData& data2, ParticleVariable<T>* value1,
          ParticleVariable<T>* value2, int matl,
          double time1, double time2,
          double abs_tolerance, double rel_tolerance);

  string d_name;
  // vector elements each represent a patch -- doesn't matter which
  vector<ParticleVariableBase*> d_particleVars;
  vector<ParticleSubset*> subsets_;
  vector<const Patch*> d_patches;

  MaterialParticleVarData* d_particleIDData;
  map<long64, const Patch*>* d_patchMap;
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

  void compare(MaterialParticleData& data2,
               double time1, double time2,
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

//__________________________________
MaterialParticleVarData::~MaterialParticleVarData()
{
  vector<ParticleVariableBase*>::iterator iter = d_particleVars.begin();
  for ( ; iter != d_particleVars.end(); iter++){
    delete *iter;
  }

  if (d_name == "p.particleID")
    delete d_patchMap;
}

//__________________________________
void MaterialParticleData::createPatchMap()
{
  ASSERT(particleIDs_ != 0); // should check for this before this point
  particleIDs_->createPatchMap();

  map<string, MaterialParticleVarData>::iterator varIter = vars_.begin();
  for ( ; varIter != vars_.end(); varIter++){
    (*varIter).second.setParticleIDData(particleIDs_);
  }
}

//__________________________________
void MaterialParticleVarData::createPatchMap()
{
  ASSERT(d_name == "p.particleID");
  if (d_patchMap)
    delete d_patchMap;

  d_patchMap = scinew map<long64, const Patch*>();

  for (unsigned int patch = 0; patch < d_particleVars.size(); patch++) {
    particleIndex count = d_particleVars[patch]->getParticleSubset()->numParticles();

    ParticleVariable<long64>* particleID = dynamic_cast< ParticleVariable<long64>* >(d_particleVars[patch]);

    if (particleID == 0) {
      ostringstream warn;
      warn << "    p.particleID must be a ParticleVariable<long64>\n";
      abort_uncomparable( warn );
    }

    for (int i = 0; i < count; i++) {
      (*d_patchMap)[(*particleID)[i]] = d_patches[patch];
    }
  }
}

//__________________________________
void MaterialParticleData::compare(MaterialParticleData& data2,
                                   double time1,
                                   double time2,
                                   double abs_tolerance,
                                   double rel_tolerance)
{
  if (vars_.size() == 0)
    return; // nothing to compare -- all good

  // map particle id's to their patches
  createPatchMap(); // also calls setParticleIDData
  data2.createPatchMap();

  sort();
  data2.sort();

  if (!particleIDs_->compare(*data2.particleIDs_, matl_,
                             time1, time2,
                             abs_tolerance, rel_tolerance)){
    ostringstream warn;
    warn << "    ParticleIDs do not match\n";
    abort_uncomparable( warn );
  }

  map<string, MaterialParticleVarData>::iterator varIter  = vars_.begin();
  map<string, MaterialParticleVarData>::iterator varIter2 = data2.vars_.begin();

  for ( ; (varIter != vars_.end()) && (varIter2 != data2.vars_.end()) ;
        varIter++, varIter2++) {

    // should catch this earlier -- vars/materials do not match
    ASSERT((*varIter).first == (*varIter2).first);

    if ((*varIter).first == "p.particleID")
      continue; // already compared

    (*varIter).second.compare((*varIter2).second, matl_,
                               time1, time2,
                               abs_tolerance, rel_tolerance);
  }
  // should catch this earlier -- vars/materials do not match
  ASSERT((varIter == vars_.end()) && (varIter2 == data2.vars_.end()));
}

//__________________________________
struct ID_Index : public pair<long64, particleIndex>
{
  ID_Index(long64 l, particleIndex i)
    : pair<long64, particleIndex>(l, i) {}

  bool operator<(ID_Index id2) {
    return first < id2.first;
  }
};

//__________________________________
void MaterialParticleData::sort()
{
  // should have made this check earlier -- particleIDs not output
  ASSERT(particleIDs_->getParticleVars().size() != 0);

  vector< ID_Index > idIndices;
  particleIndex base = 0;

  for (unsigned int i = 0; i < particleIDs_->getParticleVars().size(); i++) {
    ParticleVariable<long64>* pIDs = dynamic_cast<ParticleVariable<long64>*>(particleIDs_->getParticleVars()[i]);

    if (pIDs == 0) {
      ostringstream warn;
      warn << "    p.particleID must be a ParticleVariable<long64>\n";
      abort_uncomparable( warn );
    }

    long64         * pID    = (long64*)pIDs->getBasePointer();
    ParticleSubset * subset = pIDs->getParticleSubset();

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

  ParticleSubset* subset = scinew ParticleSubset(0, matl_, 0);
  subset->expand(subsetIndices.size());
  for (unsigned int i = 0; i < subsetIndices.size(); i++) {
    subset->addParticle(subsetIndices[i]);
  }
  gather(subset);
}

//__________________________________
void MaterialParticleData::gather(ParticleSubset* gatherSubset)
{
  map<string, MaterialParticleVarData>::iterator iter;
  for (iter = vars_.begin(); iter != vars_.end(); iter++)
    (*iter).second.gather(gatherSubset);
}

//__________________________________
void MaterialParticleVarData::add(ParticleVariableBase* pvb,
                                  const Patch* patch)
{
  d_particleVars.push_back(pvb);
  subsets_.push_back(pvb->getParticleSubset());
  d_patches.push_back(patch);
}

//__________________________________
void MaterialParticleVarData::gather(ParticleSubset* gatherSubset)
{
  ASSERT(d_particleVars.size() > 0);
  ParticleVariableBase* pvb = d_particleVars[0]->clone();
  pvb->gather(gatherSubset, subsets_, d_particleVars, 0);
  d_particleVars.clear();

  subsets_.clear();
  d_patches.clear();
  add(pvb, 0 /* all patches */);
}

//__________________________________
bool MaterialParticleVarData::
compare(MaterialParticleVarData& data2, int matl, double time1, double time2,
        double abs_tolerance, double rel_tolerance)
{
  cerr << "\tVariable: " << d_name << ", comparing via particle ids" << endl;
  ASSERT(d_particleVars.size() == 1 && subsets_.size() == 1 &&
         data2.d_particleVars.size() == 1 && data2.subsets_.size() == 1);

  ParticleVariableBase* pvb1 = d_particleVars[0];
  ParticleVariableBase* pvb2 = data2.d_particleVars[0];

  // type checks should have been made earlier
  ASSERT(pvb1->virtualGetTypeDescription() ==
         pvb2->virtualGetTypeDescription());

  switch (pvb1->virtualGetTypeDescription()->getSubType()->getType())
    {
    case Uintah::TypeDescription::double_type:
      return compare(data2, dynamic_cast<ParticleVariable<double>*>(pvb1),
                     dynamic_cast<ParticleVariable<double>*>(pvb2), matl,
                     time1, time2, abs_tolerance, rel_tolerance);
    case Uintah::TypeDescription::float_type:
      return compare(data2, dynamic_cast<ParticleVariable<float>*>(pvb1),
                     dynamic_cast<ParticleVariable<float>*>(pvb2), matl,
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
    case Uintah::TypeDescription::IntVector:
      return compare(data2, dynamic_cast<ParticleVariable<IntVector>*>(pvb1),
                     dynamic_cast<ParticleVariable<IntVector>*>(pvb2), matl,
                     time1, time2, abs_tolerance, rel_tolerance);
    case Uintah::TypeDescription::Matrix3:
      return compare(data2, dynamic_cast<ParticleVariable<Matrix3>*>(pvb1),
                     dynamic_cast<ParticleVariable<Matrix3>*>(pvb2), matl,
                     time1, time2, abs_tolerance, rel_tolerance);
    default:
      cerr << "MaterialParticleVarData::gather: ParticleVariable of unsupported type: " << pvb1->virtualGetTypeDescription()->getName() << '\n';
      Parallel::exitAll(-1);
    }
  return 0;
}

//______________________________________________________________________
//
template <class T>
bool
MaterialParticleVarData::compare( MaterialParticleVarData & data2,
                                  ParticleVariable<T>     * value1,
                                  ParticleVariable<T>     * value2,
                                  int                       matl,
                                  double                    time1,
                                  double                  /*time2*/,
                                  double                    abs_tolerance,
                                  double                    rel_tolerance )
{
  bool passes = true;
  ParticleSubset* pset1 = value1->getParticleSubset();
  ParticleSubset* pset2 = value2->getParticleSubset();

  if (pset1->numParticles() != pset2->numParticles()) {
    ostringstream warn;
    warn << "Inconsistent number of particles.\n";

    displayProblemLocation(warn, d_name, matl, 0, time1);
    
    
    warn << "    " << d_filebase1 << " has " << pset1->numParticles() << " particles.\n";
    warn << "    " << d_filebase2 << " has " << pset2->numParticles() << " particles.\n";
    abort_uncomparable( warn );
  }

  // Assumes that the particleVariables are in corresponding order --
  // not necessarily by their particle set order.  This is what the
  // sort/gather achieves.
  for( unsigned int i = 0; i < pset1->numParticles(); i++ ) {
    if (!(::compare((*value1)[i], (*value2)[i], abs_tolerance, rel_tolerance))) {
      if (d_name != "p.particleID") {
        ASSERT(getParticleID(i) == data2.getParticleID(i));
      }

      cerr << setprecision(18) << endl;
      cerr << "DIFFERENCE on particle id= " << getParticleID(i) << endl;

      IntVector origin((int)(getParticleID(i) >> 16) & 0xffff,
                       (int)(getParticleID(i) >> 32) & 0xffff,
                       (int)(getParticleID(i) >> 48) & 0xffff);

      cerr << "(Originating from " << origin << ")\n";

      const Patch* patch1 = getPatch(i);
      const Patch* patch2 = data2.getPatch(i);

      displayProblemLocation(cerr, d_name, matl, patch1, patch2, time1);

      cerr << d_filebase1 << ":\n" << (*value1)[i] << endl;
      cerr << d_filebase2 << ":\n" << (*value2)[i] << endl;

      tolerance_failure();
      if( d_concise ) {
        break;
      }

      passes = false;
    }
  }

  return passes;
}

//__________________________________
long64 MaterialParticleVarData::getParticleID(particleIndex index)
{
  ASSERT(d_particleIDData != 0);
  ASSERT(d_particleIDData->d_particleVars.size() == 1);
  ParticleVariable<long64>* particleIDs =
    dynamic_cast<ParticleVariable<long64>*>(d_particleIDData->d_particleVars[0]);
  ASSERT(particleIDs != 0);

  return (*particleIDs)[index];
}

//__________________________________
const Patch* MaterialParticleVarData::getPatch(particleIndex index)
{
  ASSERT(d_patchMap != 0);
  return (*d_patchMap)[getParticleID(index)];
}

/*
  typedef struct{
  vector<ParticleVariable<double> > pv_double_list;
  vector<ParticleVariable<float> > pv_float_list;
  vector<ParticleVariable<Point> > pv_point_list;
  vector<ParticleVariable<Vector> > pv_vector_list;
  vector<ParticleVariable<Matrix3> > pv_matrix3_list;
  ParticleVariable<Point> p_x;
  } MaterialPatchParticleData;
*/

typedef map<int, MaterialParticleData> MaterialParticleDataMap;

// replaceChar():
//   Takes a string and replaces all occurrences of 'old' with 'newch'.
string
replaceChar( const string & s, char old, char newch )
{
  string result;
  for (int i = 0; i<(int)s.size(); i++) {
    if (s[i] == old)
      result += newch;
    else
      result += s[i];
  }
  return result;
}

//__________________________________

void
addParticleData( MaterialParticleDataMap                & matlParticleDataMap,
                 DataArchive                            * da,
                 vector<string>                           var_names,
                 vector<const Uintah::TypeDescription*>   types,
                 LevelP                                   level,
                 int                                      timestep )
{
  Level::const_patch_iterator iter;
  for(iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
    const Patch* patch = *iter;

    for( int vni = 0; vni < (int)var_names.size(); vni++ ){
      std::string var_name = var_names[ vni ];

      const Uintah::TypeDescription * td      = types[ vni ];
      const Uintah::TypeDescription * subtype = td->getSubType();

      if (td->getType() == Uintah::TypeDescription::ParticleVariable) {
        ConsecutiveRangeSet matls = da->queryMaterials( var_name, patch, timestep );

        for( ConsecutiveRangeSet::iterator matlIter = matls.begin(); matlIter != matls.end(); matlIter++){

          int matl = *matlIter;
          // Add a new MaterialPatchData for each matl for this next patch.
          MaterialParticleData & data = matlParticleDataMap[ matl ];
          data.setMatl( matl );
          ParticleVariableBase* pvb = nullptr;
          switch(subtype->getType()){
          case Uintah::TypeDescription::double_type:
            pvb = scinew ParticleVariable<double>();
            break;
          case Uintah::TypeDescription::float_type:
            pvb = scinew ParticleVariable<float>();
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
          case Uintah::TypeDescription::IntVector:
            pvb = scinew ParticleVariable<IntVector>();
            break;
          case Uintah::TypeDescription::Matrix3:
            pvb = scinew ParticleVariable<Matrix3>();
            break;
          default:
            cerr << "addParticleData: ParticleVariable of unsupported type: " << subtype->getName() << '\n';
            Parallel::exitAll(-1);
          }

          da->query( *pvb, var_name, matl, patch, timestep );
          data[ var_name ].add( pvb, patch ); // will add one for each patch
        }
      }
    }
  }
} // end addParticleData()

//__________________________________

template <class T>
void
compareParticles( DataArchive  * da1,
                  DataArchive  * da2,
                  const string & var_name,
                  int            matl,
                  const Patch  * patch1,
                  const Patch  * patch2,
                  double         time,
                  int            timestep,
                  double         abs_tolerance,
                  double         rel_tolerance )
{
  ParticleVariable<T> var1;
  ParticleVariable<T> var2;
  da1->query( var1, var_name, matl, patch1, timestep );
  da2->query( var2, var_name, matl, patch2, timestep );

  ParticleSubset* pset1 = var1.getParticleSubset();
  ParticleSubset* pset2 = var2.getParticleSubset();

  if (pset1->numParticles() != pset2->numParticles()) {
    ostringstream warn;
    warn << "Inconsistent number of particles.\n";
    displayProblemLocation( warn, var_name, matl, patch1, time );
    warn << "    " << d_filebase1 << " has " << pset1->numParticles() << " particles.\n";
    warn << "    " << d_filebase2 << " has " << pset2->numParticles() << " particles.\n";
    abort_uncomparable( warn );
  }

  ParticleSubset::iterator iter1 = pset1->begin();
  ParticleSubset::iterator iter2 = pset2->begin();

  for ( ; iter1 != pset1->end() && iter2 != pset2->end(); iter1++, iter2++) {
    if (!compare( var1[*iter1], var2[*iter2], abs_tolerance, rel_tolerance)) {
      cerr << "\nValues differ too much.\n";
      displayProblemLocation( cerr, var_name, matl, patch1, time );
      cerr << d_filebase1 << ":\n";
      print( cerr, var1[ *iter1 ] );
      cerr << endl << d_filebase2 << ":\n";
      print( cerr, var2[ *iter2 ] );
      cerr << endl;
      tolerance_failure();
      if( d_concise ) {
        break;
      }
    }
  }

  // this should be true if both sets are the same size
  ASSERT( iter1 == pset1->end() && iter2 == pset2->end() );
}

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
  cerr << d_filebase1 << ": " << value1.getLowIndex() << " - " <<
  value1.getHighIndex() << endl;
  cerr << d_filebase2 << ": " << value2.getLowIndex() << " - " <<
  value2.getHighIndex() << endl;
  abort_uncomparable();
  }

  for ( ; !iter1.done() && !iter2.done(); iter1++, iter2++ ) {
  if (!compare(value1[*iter1], value2[*iter2], abs_tolerance,
  rel_tolerance)) {
  cerr << "\nValues differ too much.\n";
  displayProblemLocation(var, matl, patch1, time);
  cerr << d_filebase1 << ":\n" << value1[*iter1] << endl;
  cerr << d_filebase2 << ":\n" << value2[*iter2] << endl;
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

  virtual ~FieldComparator() {};
  virtual void
  compareFields(DataArchive* da1,
                DataArchive* da2,
                const string& var,
                ConsecutiveRangeSet matls,
                const Patch* patch,
                const Array3<const Patch*>& patch2Map,
                double time,
                int timestep,
                double abs_tolerance,
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
    : d_begin(begin) { }
  virtual ~SpecificFieldComparator() {}

  virtual void
  compareFields(DataArchive* da1,
                DataArchive* da2,
                const string& var,
                ConsecutiveRangeSet matls,
                const Patch* patch,
                const Array3<const Patch*>& patch2Map,
                double time, int timestep,
                double abs_tolerance,
                double rel_tolerance);
private:
  Iterator d_begin;
};

//__________________________________
FieldComparator*
FieldComparator::makeFieldComparator( const Uintah::TypeDescription * td,
                                      const Uintah::TypeDescription * subtype,
                                      const Patch                   * patch )
{
  switch( td->getType() ){
  case Uintah::TypeDescription::ParticleVariable:
    // Particles handled differently (and previously)
    break;
  case Uintah::TypeDescription::NCVariable: {
    NodeIterator iter = patch->getNodeIterator();
    switch( subtype->getType() ){
    case Uintah::TypeDescription::double_type : return scinew SpecificFieldComparator<NCVariable<double>,  NodeIterator>( iter );
    case Uintah::TypeDescription::float_type  : return scinew SpecificFieldComparator<NCVariable<float>,   NodeIterator>( iter );
    case Uintah::TypeDescription::int_type    : return scinew SpecificFieldComparator<NCVariable<int>,     NodeIterator>( iter );
    case Uintah::TypeDescription::Point       : return scinew SpecificFieldComparator<NCVariable<Point>,   NodeIterator>( iter );
    case Uintah::TypeDescription::Vector      : return scinew SpecificFieldComparator<NCVariable<Vector>,  NodeIterator>( iter );
    case Uintah::TypeDescription::Matrix3     : return scinew SpecificFieldComparator<NCVariable<Matrix3>, NodeIterator>( iter );
    default:
      cerr << "FieldComparator::makeFieldComparator: NC Variable of unsupported type: " << subtype->getName() << '\n';
      Parallel::exitAll( -1 );
    }
  }

  case Uintah::TypeDescription::CCVariable: {
    CellIterator iter = patch->getCellIterator();
    switch( subtype->getType() ){
    case Uintah::TypeDescription::double_type : return scinew SpecificFieldComparator<CCVariable<double>,   CellIterator>( iter );
    case Uintah::TypeDescription::float_type  : return scinew SpecificFieldComparator<CCVariable<float>,    CellIterator>( iter );
    case Uintah::TypeDescription::int_type    : return scinew SpecificFieldComparator<CCVariable<int>,      CellIterator>( iter );
    case Uintah::TypeDescription::Point       : return scinew SpecificFieldComparator<CCVariable<Point>,    CellIterator>( iter );
    case Uintah::TypeDescription::Vector      : return scinew SpecificFieldComparator<CCVariable<Vector>,   CellIterator>( iter );
    case Uintah::TypeDescription::Matrix3     : return scinew SpecificFieldComparator<CCVariable<Matrix3>,  CellIterator>( iter );
    case Uintah::TypeDescription::Stencil7    : return scinew SpecificFieldComparator<CCVariable<Stencil7>, CellIterator>( iter );
    default:
      cerr << "FieldComparator::makeFieldComparator: CC Variable of unsupported type: " << subtype->getName() << '\n';
      Parallel::exitAll( -1 );
    }
  }

  case Uintah::TypeDescription::SFCXVariable: {
    CellIterator iter = patch->getSFCXIterator();
    switch( subtype->getType() ){
    case Uintah::TypeDescription::double_type : return scinew SpecificFieldComparator<SFCXVariable<double>,  CellIterator>( iter );
    case Uintah::TypeDescription::float_type  : return scinew SpecificFieldComparator<SFCXVariable<float>,   CellIterator>( iter );
    case Uintah::TypeDescription::int_type    : return scinew SpecificFieldComparator<SFCXVariable<int>,     CellIterator>( iter );
    case Uintah::TypeDescription::Point       : return scinew SpecificFieldComparator<SFCXVariable<Point>,   CellIterator>( iter );
    case Uintah::TypeDescription::Vector      : return scinew SpecificFieldComparator<SFCXVariable<Vector>,  CellIterator>( iter );
    case Uintah::TypeDescription::Matrix3     : return scinew SpecificFieldComparator<SFCXVariable<Matrix3>, CellIterator>( iter );
    default:
      cerr << "FieldComparator::makeFieldComparator: SFCX Variable of unsupported type: " << subtype->getName() << '\n';
      Parallel::exitAll( -1 );
    }
  }

  case Uintah::TypeDescription::SFCYVariable: {
    CellIterator iter = patch->getSFCYIterator();
    switch( subtype->getType() ){
    case Uintah::TypeDescription::double_type : return scinew SpecificFieldComparator<SFCYVariable<double>,  CellIterator>( iter );
    case Uintah::TypeDescription::float_type  : return scinew SpecificFieldComparator<SFCYVariable<float>,   CellIterator>( iter );
    case Uintah::TypeDescription::int_type    : return scinew SpecificFieldComparator<SFCYVariable<int>,     CellIterator>( iter );
    case Uintah::TypeDescription::Point       : return scinew SpecificFieldComparator<SFCYVariable<Point>,   CellIterator>( iter );
    case Uintah::TypeDescription::Vector      : return scinew SpecificFieldComparator<SFCYVariable<Vector>,  CellIterator>( iter );
    case Uintah::TypeDescription::Matrix3     : return scinew SpecificFieldComparator<SFCYVariable<Matrix3>, CellIterator>( iter );
    default:
      cerr << "FieldComparator::makeFieldComparator: SFCY Variable of unsupported type: " << subtype->getName() << '\n';
      Parallel::exitAll( -1 );
    }
  }

  case Uintah::TypeDescription::SFCZVariable: {
    CellIterator iter = patch->getSFCZIterator();
    switch( subtype->getType() ){
    case Uintah::TypeDescription::double_type : return scinew SpecificFieldComparator<SFCZVariable<double>,   CellIterator>( iter );
    case Uintah::TypeDescription::float_type  : return scinew SpecificFieldComparator<SFCZVariable<float>,    CellIterator>( iter );
    case Uintah::TypeDescription::int_type    : return scinew SpecificFieldComparator<SFCZVariable<int>,      CellIterator>( iter );
    case Uintah::TypeDescription::Point       : return scinew SpecificFieldComparator<SFCZVariable<Point>,    CellIterator>( iter );
    case Uintah::TypeDescription::Vector      : return scinew SpecificFieldComparator<SFCZVariable<Vector>,   CellIterator>( iter );
    case Uintah::TypeDescription::Matrix3     : return scinew SpecificFieldComparator<SFCZVariable<Matrix3>,  CellIterator>( iter );
    default:
      cerr << "FieldComparator::makeFieldComparator: SFCZ Variable of unsupported type: " << subtype->getName() << '\n';
      Parallel::exitAll( -1 );
    }
  }
  default:
    cerr << "FieldComparator::makeFieldComparator: Variable of unsupported type: " << td->getName() << '\n';
    Parallel::exitAll( -1 );
  }
  return nullptr;
}

//__________________________________
template <class Field, class Iterator>
void
SpecificFieldComparator<Field, Iterator>::compareFields( DataArchive                * da1,
                                                         DataArchive                * da2,
                                                         const string               & var_name,
                                                         ConsecutiveRangeSet          matls,
                                                         const Patch                * patch,
                                                         const Array3<const Patch*> & patch2Map,
                                                         double                       time1,
                                                         int                          timestep,
                                                         double                       abs_tolerance,
                                                         double                       rel_tolerance )
{
  Field* field2;
  bool firstMatl = true;

  //__________________________________
  //  Matl loop
  for( ConsecutiveRangeSet::iterator matlIter = matls.begin(); matlIter != matls.end(); matlIter++ ) {

    int matl = *matlIter;

    Field field;
    bool found = da1->query( field, var_name, matl, patch, timestep );

    if( !found ) {
      cout << "Skipping comparison of " << var_name << " as it was not found in DataArchive1.\n";
      continue;
    }

    map<const Patch*, Field*> patch2FieldMap;
    typename map<const Patch*, Field*>::iterator findIter;

    for( Iterator iter = d_begin ; !iter.done(); iter++ ) {
      const Patch* patch2 = patch2Map[*iter];

      findIter = patch2FieldMap.find( patch2 );

      if (findIter == patch2FieldMap.end()) {
        if (firstMatl) { // check only needs to be made the first round
          ConsecutiveRangeSet matls2 = da2->queryMaterials( var_name, patch2, timestep );
          ASSERT(matls == matls2); // check should have been made previously
        }
        field2 = scinew Field();
        patch2FieldMap[ patch2 ] = field2;
        found = da2->query( *field2, var_name, matl, patch2, timestep );
        if( !found ) {
          cout << "Skipping comparison of " << var_name << " as it was not found in DataArchive2.\n";
          continue;
        }
      }
      else {
        field2 = (*findIter).second;
      }

      if (!compare(field[*iter], (*field2)[*iter], abs_tolerance, rel_tolerance)) {

        cerr << "DIFFERENCE " << *iter << "  ";
        displayProblemLocation( cerr, var_name, matl, patch, patch2, time1 );

        cerr << d_filebase1 << " (1)\t\t" << d_filebase2 << " (2)"<<endl;
        print(cerr, field[*iter]);
        cerr << "\t\t";
        print(cerr, (*field2)[*iter]);
        cerr << endl;

        tolerance_failure();
        if( d_concise ) {
          break; // Exit for() loop as we are only displaying first error per variable.
        }
      }
    }

    typename map<const Patch*, Field*>::iterator iter = patch2FieldMap.begin();
    for ( ; iter != patch2FieldMap.end(); iter++) {
      delete (*iter).second;
    }
    firstMatl = false;
  }
}


//______________________________________________________________________
// map nodes to their owning patch in a level.
// Nodes are used because I am assuming that whoever owns the node at
// that index also owns the cell, or whatever face at that same index.
// The same doesn't work if you used cells because nodes can go beyond
// cells (when there is no neighbor on the greater side).
void
buildPatchMap( LevelP                 level,
               const string         & filebase,
               Array3<const Patch*> & patchMap,
               double                 time,
               Patch::VariableBasis   basis )
{
  const PatchSet* allPatches = level->allPatches();
  const PatchSubset* patches = allPatches->getUnion();
  if (patches->size() == 0)
    return;

  IntVector bl   = IntVector(0,0,0);
  IntVector low  = patches->get(0)->getExtraLowIndex(basis,bl);
  IntVector high = patches->get(0)->getExtraHighIndex(basis,bl);

  for (int i = 1; i < patches->size(); i++) {
    low  = Min(low,  patches->get(i)->getExtraLowIndex(basis,bl));
    high = Max(high, patches->get(i)->getExtraHighIndex(basis,bl));
  }

  patchMap.resize(low, high);
  patchMap.initialize(0);

  Level::const_patch_iterator iter;
  for(iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
    const Patch* patch = *iter;

    ASSERT(Min(patch->getExtraLowIndex(basis,bl), low)   == low);
    ASSERT(Max(patch->getExtraHighIndex(basis,bl), high) == high);

    patchMap.rewindow(patch->getExtraLowIndex(basis,bl),
                      patch->getExtraHighIndex(basis,bl));

    serial_for( patchMap.range(), [&](int i, int j, int k) {
      if (patchMap(i,j,k) != nullptr) {
        static ProgressiveWarning pw("Two patches on the same grid overlap", 10);
        if (pw.invoke())
          cerr << "Patches " << patch->getID() << " and "
               << (*iter)->getID() << " overlap on the same file at time " << time
               << " in " << filebase << " at index " << IntVector(i,j,k) << endl;
        //abort_uncomparable();

        // in some cases, we can have overlapping patches, where an extra cell/node
        // overlaps an interior cell/node of another patch.  We prefer the interior
        // one.  if there are two overlapping interior ones (nodes or face centers only),
        // they should have the same value.  However, since this patchMap is also used
        // for cell centered variables give priority to the patch that has this index within
        // its interior cell centered variables
        IntVector in_low  = patch->getLowIndex(basis);
        IntVector in_high = patch->getHighIndex(basis);

        if (i >= in_low.x() && j >= in_low.y() && k >= in_low.z() &&
            i < in_high.x() && j < in_high.y() && k < in_high.z()) {
          patchMap(i,j,k) = patch;
        }
      }
      else
      {
        patchMap(i,j,k) = patch;
      }
    });
  }
  patchMap.rewindow(low, high);
}

//______________________________________________________________________

int
main( int argc, char** argv )
{
  Uintah::Parallel::initializeManager(argc, argv);

  double rel_tolerance  = 1e-6; // Default
  double abs_tolerance  = 1e-9; //   values...
  string ignoreVar      = "none";
  bool sortVariables    = true;
  int  udaLevels[2];                // user can override and specify the levels to compare.  Useful for 1L vs N level comparison
  udaLevels[0]          =-9;
  udaLevels[1]          =-9;

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
      d_tolerance_as_warnings = true;
    }
    else if(s == "-concise") {
      d_concise = true;
    }
    else if(s == "-dont_sort") {
      sortVariables = false;
    }
    else if(s == "-skip_unknown_types") {
      d_strict_types = false;
    }
    else if(s == "-exact") {
      abs_tolerance = 0;
      rel_tolerance = 0;
    }
    else if(s == "-levels") {
      udaLevels[0] = atoi(argv[++i]);
      cout << "udaLevels: " << udaLevels[0];
      udaLevels[1] = atoi(argv[++i]);
      cout << "  " << udaLevels[1];
    }
    else if(s == "-ignoreVariable") {
      if (++i == argc){
        usage("-ignoreVariable, no variable given", argv[0]);
      }else{
        ignoreVar = argv[i];
      }
    }
    else if(s[0] == '-' && s[1] == 'h' ) { // lazy check for -h[elp] option
      usage( "", argv[0] );
    }
    else if(s[0] == '-' ) {
      usage( s, argv[0] );
    }
    else {
      if (d_filebase1 != "") {
        if (d_filebase2 != "") {
          usage(s, argv[0]);
        }
        else {
          d_filebase2 = argv[i];
        }
      }
      else {
        d_filebase1 = argv[i];
      }
    }
  }

  if( d_filebase2 == "" ){
    cerr << "\nYou must specify two archive directories.\n";
    usage("", argv[0]);
  }

  if( !validDir( d_filebase1 ) ) {
    cerr << "\nParameter '" << d_filebase1 << "' is not a valid directory.\n";
    usage("", argv[0]);
  }
  if( !validDir( d_filebase2 ) ) {
    cerr << "\nParameter '" << d_filebase2 << "' is not a valid directory.\n";
    usage("", argv[0]);
  }

  cerr << "Using absolute tolerance: " << abs_tolerance << endl;
  cerr << "Using relative tolerance: " << rel_tolerance << endl;

  if( udaLevels[0] != -9 ){
    cerr << "Comparing uda1: Level("<< udaLevels[0] << ") against uda2: level(" << udaLevels[1] << ")" << endl;
  }

  if (rel_tolerance < 0) {
    cerr << "Must have a non-negative value rel_tolerance.\n";
    Parallel::exitAll(1);
  }

  // default to 16 digits of precision when using exact comparison (i.e. rel_tolerance = 0)
  int digits_precision = (rel_tolerance > 0 ) ? (int)ceil(-log10(rel_tolerance)) + 1 : 16;
  cerr << setprecision(digits_precision);
  cout << setprecision(digits_precision);

  try {
    DataArchive* da1 = scinew DataArchive(d_filebase1);
    DataArchive* da2 = scinew DataArchive(d_filebase2);

    vector<string>                                         vars,      vars2;
    vector<int>                                            num_matls, num_matls2;
    vector<const Uintah::TypeDescription*>                 types,     types2;
    vector< pair<string, const Uintah::TypeDescription*> > vartypes1, vartypes2;

    da1->queryVariables( vars, num_matls, types );
    ASSERTEQ(vars.size(), types.size());

    da2->queryVariables( vars2, num_matls2, types2 );
    ASSERTEQ(vars2.size(), types2.size());

    if (vars.size() != vars2.size() && ignoreVar.size() == 0) {
      ostringstream warn;
      warn << "    " << d_filebase1 << " has " << vars.size() << " variables\n";
      warn << "    " << d_filebase2 << " has " << vars2.size() << " variables\n";
      abort_uncomparable( warn );
    }

    vartypes1.resize( vars.size() );
    vartypes2.resize( vars2.size() );
    int count = 0;
    //__________________________________
    //  eliminate the variable to be ignored
    // Create a list of ignored variables
    // uda 1
    stringstream iV(ignoreVar);
    vector<string> vs;
    copy(istream_iterator<string>(iV), istream_iterator<string>(), back_inserter(vs));

    for (unsigned int i = 0; i < vars.size(); i++) {
      vector<string>::iterator fs = find(vs.begin(),vs.end(),vars[i]);
      // if vars[i] is NOT in the ignore Variables list make a pair
      if (fs == vs.end()){
        vartypes1[count] = make_pair(vars[i], types[i]);
        count ++;
      }
    }
    vars.resize(count);
    vartypes1.resize(vars.size());

    // uda 2
    count =0;
    for (unsigned int i = 0; i < vars2.size(); i++) {
      vector<string>::iterator fs = find(vs.begin(),vs.end(),vars2[i]);
      // if vars[i] is NOT in the ignore Variables list make a pair
      if (fs == vs.end()){
        vartypes2[count] = make_pair(vars2[i], types2[i]);
        count ++;
      }
    }
    vars2.resize(count);
    vartypes2.resize(vars2.size());

    if (vartypes1.size() != vartypes2.size() )  {
      ostringstream warn;
      warn << "    " << d_filebase1 << " has " << vars.size()  << " variables\n";
      warn << "    " << d_filebase2 << " has " << vars2.size() << " variables\n";
      abort_uncomparable(warn);
    }

    //__________________________________
    // sort vars so uda's can be compared if their index files have
    // different orders of variables.
    // Assuming that there are no duplicates in the var names, these will
    // sort alphabetically by varname.
    if(sortVariables){
      sort(vartypes1.begin(), vartypes1.end());
      sort(vartypes2.begin(), vartypes2.end());
    }
    for (unsigned int i = 0; i < vars.size(); i++) {
      vars[i]   = vartypes1[i].first;
      types[i]  = vartypes1[i].second;
      vars2[i]  = vartypes2[i].first;
      types2[i] = vartypes2[i].second;
    }

    for (unsigned int i = 0; i < vars.size(); i++) {
      if (vars[i] != vars2[i]) {
        ostringstream warn;
        warn << "    Variable " << vars[i] << " in " << d_filebase1 << " does not match\n";
        warn << "    variable " << vars2[i] << " in " << d_filebase2 << "\n";
        abort_uncomparable(warn);
      }

      if (types[i] != types2[i]) {
        ostringstream warn;
        warn << "    Variable " << vars[i] << " does not have the same type in both uda directories.\n";
        warn << "    In " << d_filebase1 << " its type is " << types[i]->getName() << endl;
        warn << "    In " << d_filebase2 << " its type is " << types2[i]->getName() << endl;
        abort_uncomparable(warn);
      }
    }

    vector<int>     ts_index;
    vector<double>  times;
    vector<int>     ts_index2;
    vector<double>  times2;

    da1->queryTimesteps( ts_index, times );
    ASSERTEQ( ts_index.size(), times.size() );

    da2->queryTimesteps( ts_index2, times2 );
    ASSERTEQ( ts_index2.size(), times2.size() );

    for( unsigned long tstep = 0; tstep < times.size() && tstep < times2.size(); tstep++ ){
      if (!compare(times[tstep], times2[tstep], abs_tolerance, rel_tolerance)) {
        ostringstream warn;
        warn << "    Timestep at time " << times[tstep]  << " in " << d_filebase1 << " does not match\n";
        warn << "    Timestep at time " << times2[tstep] << " in " << d_filebase2 << " within the allowable tolerance.\n";
        abort_uncomparable(warn);
      }

      double time1 = times[tstep];
      double time2 = times2[tstep];
      cerr << "time = " << time1 << "\n";

      GridP grid  = da1->queryGrid(tstep);
      GridP grid2 = da2->queryGrid(tstep);

      int maxLevels[2];
      maxLevels[0] = grid->numLevels();
      maxLevels[1] = grid2->numLevels();

      int minLevel[2];
      minLevel[0] = 0;
      minLevel[1] = 0;

       // override if user has specified the levels to compare
      if( udaLevels[0] != -9 ){
        minLevel[0]  = udaLevels[0];
        minLevel[1]  = udaLevels[1];
        maxLevels[0] = minLevel[0] +1;
        maxLevels[1] = minLevel[1] +1;

        if( maxLevels[0] > grid->numLevels() || maxLevels[1] > grid2->numLevels() ) {
          ostringstream warn;
          warn << "    The level(s) specified (uda:" << maxLevels[0] << " , uda2: " << maxLevels[1] << ") are invalid.\n";
          warn << "    The maximum level index that are valid are (uda:" << grid->numLevels() << " , uda2: " << grid2->numLevels() << ").\n";
          abort_uncomparable(warn);
        }

      } else if ( maxLevels[0] != maxLevels[1] ) {
        ostringstream warn;
        warn << "    Grid at time " << time1 << " in " << d_filebase1 << " has " << grid->numLevels() << " levels.\n";
        warn << "    Grid at time " << time2 << " in " << d_filebase2 << " has " << grid2->numLevels() << " levels.\n";
        abort_uncomparable(warn);
      }


      // do some consistency checking first
      bool hasParticleIDs  = false;
      bool hasParticleData = false;

#if 0 
      // Bullet proofing checks for each variable:
      for( int v = 0; v < (int)vars.size();v++ ){
        std::string var = vars[v];

        if (var == "p.particleID"){
          hasParticleIDs = true;
        }
        if (types[v]->getType() == Uintah::TypeDescription::ParticleVariable){
          hasParticleData = true;
        }

        for( int l1 = minLevel[0], l2 = minLevel[1]; (l1 < maxLevels[0] &&  l2 < maxLevels[1]); l1++, l2++) {

          LevelP level  = grid->getLevel(l1);
          LevelP level2 = grid2->getLevel(l2);

          ConsecutiveRangeSet matls;

          bool first = true;
          Level::const_patch_iterator iter;

          //__________________________________
          //  bulletproofing does the variable exist in both DAs on this timestep?
          //  This problem mainly occurs if <outputInitTimestep> has been specified.
          bool existsDA1 = true;
          bool existsDA2 = true;
          for(iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
            const Patch* patch = *iter;
            if ( ! da1->exists( var, patch, tstep ) ){
              existsDA1 = false;
            }
          }
          for(iter = level2->patchesBegin(); iter != level2->patchesEnd(); iter++) {
            const Patch* patch = *iter;
            if ( ! da2->exists( var, patch, tstep ) ){
              existsDA2 = false;
            }
          }
          if( existsDA1 != existsDA2 ) {
            ostringstream warn;
            warn << "    The variable ("<< var << ") was not found on timestep (" << index[tstep]
                 <<  "), Level-" << level->getIndex() << ", in both udas (" << existsDA1 << ", " << existsDA2 << ").\n";
                 << "    If this occurs on timestep 0 then ("<< var<< ") was not computed in the initialization task.\n";
            abort_uncomparable(warn);
          }

          //__________________________________
          //  bulletproof are material sets consistent over DA 1
          for(iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
            const Patch* patch = *iter;

            if (first) {
              matls = da1->queryMaterials( var, patch, tstep );
            }
            else if (matls != da1->queryMaterials( var, patch, tstep )) {
              ostringstream warn;
              warn << "    The material set is not consistent for variable "
                   << var << " across patches at time " << time1 << ".\n";
              warn << "    Previously was: " << matls << endl;
              warn << "    But on patch " << patch->getID() << ": " <<
                     da1->queryMaterials(var, patch, tstep) << ".\n";
              abort_uncomparable(warn);
            }
            first = false;
          }

          ASSERT( !first ); /* More serious problems would show up if this assertion would fail */

          //__________________________________
          //  bulletproof are material sets consistent between DA1 & DA2
          for( Level::const_patch_iterator iter = level2->patchesBegin(); iter != level2->patchesEnd(); iter++ ) {
            const Patch* patch = *iter;

            ConsecutiveRangeSet matls2 = da2->queryMaterials( var, patch, tstep );

            if ( matls != matls2 ) {
              ostringstream warn;
              warn << "Inconsistent material sets for variable " << var << " on patch = " << patch->getID() << ", time " << time1 << "\n";
              warn << "    " << d_filebase1 << " (1) has material set: " << matls << ".\n";
              warn << "    " << d_filebase2 << " (2) has material set: " << da2->queryMaterials(var, patch, tstep) << ".\n";
              // If this is timestep 0 and one or both of the DAs are using PIDX format:
              if( ts_index[ tstep ] == 0 && ( da1->isPIDXFormat() || da2->isPIDXFormat() ) ) {
                cout << "Ignoring the following warning because this is timestep 0 with a PIDX UDA and thus probably not a real issue...\n";
                cout << warn.str() << "\n";
              }
              else {
                abort_uncomparable( warn );
              }
            }
          }
        }
      }
#endif

      //______________________________________________________________________
      // COMPARE PARTICLE VARIABLES
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

            if (d_strict_types) {
              cerr << ".\nQuitting.\n";
              Parallel::exitAll(-1);
            }
            cerr << "; skipping comparison...\n";
            continue;
          }

          for( int l1 = minLevel[0], l2 = minLevel[1]; (l1 < maxLevels[0] &&  l2 < maxLevels[1]); l1++, l2++) {

            LevelP level  = grid->getLevel(l1);
            LevelP level2 = grid2->getLevel(l2);

            if (level->numPatches() != level2->numPatches()) {
              ostringstream warn;
              warn << "    Inconsistent number of patches on level " << l1 << " at time " << time1 << ":" << endl;
              warn << "    " << d_filebase1 << " has " << level->numPatches()  << " patches.\n";
              warn << "    " << d_filebase2 << " has " << level2->numPatches() << " patches.\n";
              abort_uncomparable(warn);
            }


            Level::const_patch_iterator iter2 = level2->patchesBegin();
            for(Level::const_patch_iterator iter = level->patchesBegin();
                iter != level->patchesEnd(); iter++, iter2++){

              const Patch* patch  = *iter;
              const Patch* patch2 = *iter2;

              if (patch->getID() != patch2->getID()) {
                ostringstream warn;
                warn << "    Inconsistent patch ids on level " << l1 << " at time " << time1 << endl;
                warn << "    " << d_filebase1 << " has patch id " << patch->getID() << " where\n";
                warn << "    " << d_filebase2 << " has patch id " << patch2->getID() << ".\n";
                abort_uncomparable(warn);
              }

              cerr << "\t\tPatch: " << patch->getID() << "\n";

              if (!compare(patch->getExtraBox().lower(), patch2->getExtraBox().lower(), abs_tolerance, rel_tolerance) ||
                  !compare(patch->getExtraBox().upper(), patch2->getExtraBox().upper(), abs_tolerance, rel_tolerance)) {
                
                ostringstream warn;
                warn << "    Inconsistent patch bounds on patch " << patch->getID()
                     << " at time " << time1 << ".\n";

                warn << "    " << d_filebase1 << " has bounds " << patch->getExtraBox().lower()
                     << " - " << patch->getExtraBox().upper() << ".\n";

                warn << "    " << d_filebase2 << " has bounds " << patch2->getExtraBox().lower()
                     << " - " << patch2->getExtraBox().upper() << ".\n";

                warn << "    " << "Difference is: " << patch->getExtraBox().lower() - patch2->getExtraBox().lower() << " - " 
                                          << patch->getExtraBox().upper() - patch2->getExtraBox().upper() << ".\n";
                abort_uncomparable(warn);
              }

              ConsecutiveRangeSet matls  = da1->queryMaterials( var, patch,  tstep );
              ConsecutiveRangeSet matls2 = da2->queryMaterials( var, patch2, tstep );
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
                                             time1, tstep, abs_tolerance, rel_tolerance);
                    break;
                  case Uintah::TypeDescription::float_type:
                    compareParticles<float>(da1, da2, var, matl, patch, patch2,
                                            time1, tstep, abs_tolerance, rel_tolerance);
                    break;
                  case Uintah::TypeDescription::int_type:
                    compareParticles<int>(da1, da2, var, matl, patch, patch2,
                                          time1, tstep, abs_tolerance, rel_tolerance);
                    break;
                  case Uintah::TypeDescription::Point:
                    compareParticles<Point>(da1, da2, var, matl, patch, patch2,
                                            time1, tstep, abs_tolerance, rel_tolerance);
                    break;
                  case Uintah::TypeDescription::Vector:
                    compareParticles<Vector>(da1, da2, var, matl, patch, patch2,
                                             time1, tstep, abs_tolerance, rel_tolerance);
                    break;
                  case Uintah::TypeDescription::IntVector:
                    compareParticles<IntVector>(da1, da2, var, matl, patch, patch2,
                                             time1, tstep, abs_tolerance, rel_tolerance);
                    break;
                  case Uintah::TypeDescription::Matrix3:
                    compareParticles<Matrix3>(da1, da2, var, matl, patch, patch2,
                                              time1, tstep, abs_tolerance, rel_tolerance);
                    break;
                  default:
                    cerr << "main: ParticleVariable of unsupported type: " << subtype->getName() << '\n';
                    Parallel::exitAll(-1);
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
        for( int l1 = minLevel[0], l2 = minLevel[1]; (l1 < maxLevels[0] &&  l2 < maxLevels[1]); l1++, l2++) {

          LevelP level  = grid->getLevel(l1);
          LevelP level2 = grid2->getLevel(l2);

          MaterialParticleDataMap matlParticleDataMap1;
          MaterialParticleDataMap matlParticleDataMap2;

          addParticleData( matlParticleDataMap1, da1, vars,  types,  level,  tstep );
          addParticleData( matlParticleDataMap2, da2, vars2, types2, level2, tstep );

          MaterialParticleDataMap::iterator matlIter;
          MaterialParticleDataMap::iterator matlIter2;

          matlIter  = matlParticleDataMap1.begin();
          matlIter2 = matlParticleDataMap2.begin();

          for (; (matlIter  != matlParticleDataMap1.end()) &&
                 (matlIter2 != matlParticleDataMap2.end()); matlIter++, matlIter2++) {

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
          if (d_strict_types) {
            cerr << ".\nQuitting.\n";
            Parallel::exitAll(-1);
          }
          cerr << "; skipping comparison...\n";
          continue;
        }

        Patch::VariableBasis basis=Patch::translateTypeToBasis(td->getType(),false);

        for( int l1 = minLevel[0], l2 = minLevel[1]; (l1 < maxLevels[0] &&  l2 < maxLevels[1]); l1++, l2++) {

          LevelP level  = grid->getLevel(l1);
          LevelP level2 = grid2->getLevel(l2);

          //check patch coverage
          vector<Region> region1, region2, difference1, difference2;

          for( int i=0; i < level->numPatches(); i++ ) {
            const Patch* patch=level->getPatch(i);
            region1.push_back(Region(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex()));
          }
          for( int i=0; i < level2->numPatches(); i++ ) {
            const Patch* patch=level2->getPatch(i);
            region2.push_back(Region(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex()));
          }

          difference1 = Region::difference(region1,region2);
          difference2 = Region::difference(region1,region2);

          if(!difference1.empty() || !difference2.empty()){
            ostringstream warn;
            warn << "    Patches on level:" << l1 << " do not cover the same area.\n";
            abort_uncomparable(warn);
          }

          // map nodes to patches in level and level2 respectively
          Array3<const Patch*> patchMap;
          Array3<const Patch*> patch2Map;

          buildPatchMap(level,  d_filebase1, patchMap,  time1, basis);
          buildPatchMap(level2, d_filebase2, patch2Map, time2, basis);

          serial_for( patchMap.range(), [&](int i, int j, int k) {
            // bulletproofing
            if ((patchMap(i,j,k)  == nullptr && patch2Map(i,j,k) != nullptr) ||
                (patch2Map(i,j,k) == nullptr && patchMap(i,j,k) != nullptr)) {
              ostringstream warn;
              warn << "    Inconsistent patch coverage on level " << l1
                   << " at time " << time1 << ".\n";

              if (patchMap(i,j,k) != nullptr) {
                warn << "    " << IntVector(i,j,k) << " is covered by " << d_filebase1 << endl
                     << " and not " << d_filebase2 << ".\n";
              }
              else {
                warn << "    " << IntVector(i,j,k) << " is covered by " << d_filebase2 << endl
                     << " and not " << d_filebase1 << ".\n";
              }
              abort_uncomparable(warn);
            }
          });

          Level::const_patch_iterator iter;

          for(iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
            const Patch* patch = *iter;

            ConsecutiveRangeSet matls = da1->queryMaterials(var, patch, tstep);

            FieldComparator* comparator = FieldComparator::makeFieldComparator(td, subtype, patch);

            if (comparator != 0) {
              comparator->compareFields( da1, da2, var, matls, patch,
                                         patch2Map, time1, tstep,
                                         abs_tolerance, rel_tolerance );
              delete comparator;
            }
          }
        } // end for (l)
      } // end for (v)
    } // end for(tstep)
    //__________________________________
    //
    if (times.size() != times2.size()) {
      ostringstream warn;
      warn << "\n";
      warn << "    " << d_filebase1 << " has " << times.size() << " timesteps\n";
      warn << "    " << d_filebase2 << " has " << times2.size() << " timesteps\n";
      abort_uncomparable(warn);
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

  if (d_tolerance_error) {
    cerr << "\nComparison did NOT fully pass.\n";
    Parallel::exitAll(2);
  }
  else
    cerr << "\nComparison fully passed!\n";

  return 0;
}
