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

#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <math.h>


using namespace SCIRun;
using namespace std;
using namespace Uintah;

typedef struct{
  vector<ParticleVariable<double> > pv_double_list;
  vector<ParticleVariable<Point> > pv_point_list;
  vector<ParticleVariable<Vector> > pv_vector_list;
  vector<ParticleVariable<Matrix3> > pv_matrix3_list;
  ParticleVariable<Point> p_x;
} MaterialData;

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

void usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
	cerr << "Error parsing argument: " << badarg << '\n';
    cerr << "Usage: " << progname << " [options] <archive file 1> <archive file 2\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h[elp]\n";
    cerr << "  -abs_tolerance [double] (allowable absolute difference of any numbers)\n";
    cerr << "  -rel_tolerance [double] (allowable relative difference of any numbers)\n";
    cerr << "\nNote: The absolute and relative tolerance tests must both fail\n      for a comparison to fail.\n";
    exit(1);
}

// I don't want to have to pass these parameters all around, so
// I've made them file-scope.
string filebase1;
string filebase2;
bool tolerance_as_warnings = false;
bool tolerance_error = false;

void abort_uncomparable()
{
  cerr << "\nThe uda directories may not be compared.\n";
  exit(5);
}

void tolerance_failure()
{
  if (tolerance_as_warnings) {
    tolerance_error = true;
    cerr << endl;
  }
  else
    exit(2);
}

void displayProblemLocation(const string& var, int matl,
			      const Patch* patch, double time)
{
  cerr << "Time: " << time << endl;
  cerr << "Patch: " << patch->getID() << endl;  
  cerr << "Variable: " << var << endl;
  cerr << "Material: " << matl << endl << endl;
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
      cerr << "Values differ too much.\n";
      displayProblemLocation(var, matl, patch1, time);    
      cerr << filebase1 << ":\n" << value1[*iter1] << endl;
      cerr << filebase2 << ":\n" << value2[*iter2] << endl;
      tolerance_failure();
    }
  }

  // this should be true if both sets are the same size
  ASSERT(iter1 == pset1->end() && iter2 == pset2->end());
}

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
      cerr << "Values differ too much.\n";
      displayProblemLocation(var, matl, patch1, time);    
      cerr << filebase1 << ":\n" << value1[*iter1] << endl;
      cerr << filebase2 << ":\n" << value2[*iter2] << endl;
      tolerance_failure();
    }
  }

  // this should be true if both sets are the same size
  ASSERT(iter1.done() && iter2.done());
}


int main(int argc, char** argv)
{
  /*
   * Default values
   */
  double rel_tolerance = 1e-5;
  double abs_tolerance = 1e-8;

  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
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
    } else {
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
  
  if (filebase2 == ""){
    cerr << "Must specify two archive directories.\n";
    usage("", argv[0]);
  }

  if (rel_tolerance <= 0) {
    cerr << "Must have a positive value rel_tolerance.\n";
    exit(1);
  }

  int digits_precision = (int)ceil(-log10(rel_tolerance)) + 1;
  cerr << setprecision(digits_precision);
  cout << setprecision(digits_precision);

  try {
    XMLPlatformUtils::Initialize();
  } catch(const XMLException& toCatch) {
    cerr << "Caught XML exception: " << toString(toCatch.getMessage()) 
	 << '\n';
    exit( 1 );
  }
  
  try {
    DataArchive* da1 = scinew DataArchive(filebase1);
    DataArchive* da2 = scinew DataArchive(filebase2);

    vector<string> vars;
    vector<const TypeDescription*> types;
    vector<string> vars2;
    vector<const TypeDescription*> types2;
    da1->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    da2->queryVariables(vars2, types2);
    ASSERTEQ(vars2.size(), types2.size());

    if (vars.size() != vars2.size()) {
      cerr << filebase1 << " has " << vars.size() << " variables\n";
      cerr << filebase2 << " has " << vars2.size() << " variables\n";
      abort_uncomparable();
    }

    for (int i = 0; i < vars.size(); i++) {
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

    if (times.size() != times2.size()) {
      cerr << filebase1 << " has " << times.size() << " timesteps\n";
      cerr << filebase2 << " has " << times2.size() << " timesteps\n";
      abort_uncomparable();
    }

    for (int i = 0; i < times.size(); i++) {
      if (!compare(times[i], times2[i], abs_tolerance, rel_tolerance)) {
	cerr << "Timestep at time " << times[i] << " in " << filebase1
	     << " does not match\n";
	cerr << "timestep at time " << times2[i] << " in " << filebase2
	     << " within the allowable tolerance.\n";
	tolerance_failure();
      }
    }

    for(unsigned long t = 0; t < times.size(); t++){
      double time = times[t];
      double time2 = times2[t];
      cout << "time = " << time << "\n";
      GridP grid = da1->queryGrid(time);
      GridP grid2 = da2->queryGrid(times2[t]);

      if (grid->numLevels() != grid2->numLevels()) {
	cerr << "Grid at time " << time << " in " << filebase1
	     << " has " << grid->numLevels() << " levels.\n";
	cerr << "Grid at time " << time2 << " in " << filebase2
	     << " has " << grid->numLevels() << " levels.\n";
	abort_uncomparable();
      }
      
      for(int v=0;v<(int)vars.size();v++){
	std::string var = vars[v];
	const TypeDescription* td = types[v];
	const TypeDescription* subtype = td->getSubType();
	cout << "\tVariable: " << var << ", type " << td->getName() << "\n";
	for(int l=0;l<grid->numLevels();l++){
	  LevelP level = grid->getLevel(l);
	  LevelP level2 = grid2->getLevel(l);
	  if (level->numPatches() != level2->numPatches()) {
	    cerr << "Inconsistent number of patches on level " << l <<
	      " at time " << time << ":" << endl;
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
		   << " at time " << time << endl;
	      cerr << filebase1 << " has patch id " << patch->getID()
		   << " where\n";
	      cerr << filebase2 << " has patch id " << patch2->getID() << endl;
	      abort_uncomparable();  
	    }
	    
	    cout << "\t\tPatch: " << patch->getID() << "\n";

	    if (!compare(patch->getBox().lower(), patch2->getBox().lower(),
			 abs_tolerance, rel_tolerance) ||
		!compare(patch->getBox().upper(), patch2->getBox().upper(),
			 abs_tolerance, rel_tolerance)) {
	      cerr << "Inconsistent patch bounds on patch " << patch->getID()
		   << " at time " << time << endl;
	      cerr << filebase1 << " has bounds " << patch->getBox().lower()
		   << " - " << patch->getBox().upper() << ".\n";
	      cerr << filebase2 << " has bounds " << patch2->getBox().lower()
		   << " - " << patch2->getBox().upper() << ".\n";
	      cerr << "Difference is: " << patch->getBox().lower() - patch2->getBox().lower() << " - " << patch->getBox().upper() - patch2->getBox().upper() << endl;
	      abort_uncomparable();  
	    }
	    

	    ConsecutiveRangeSet matls = da1->queryMaterials(var, patch, time);
	    ConsecutiveRangeSet matls2 = da2->queryMaterials(var, patch2,
							     time2);
	    if (matls != matls2) {
	      cerr << "Inconsistent material sets for variable "
		   << var << " on patch " << patch->getID()
		   << " and time " << time << endl;
	      cerr << filebase1 << " has material set: " << matls << ".\n";
	      cerr << filebase2 << " has material set: " << matls2 << ".\n";
	      abort_uncomparable();  
	    }
	    
	    // loop over materials
	    for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		matlIter != matls.end(); matlIter++){
	      int matl = *matlIter;
	      cout << "\t\t\tMaterial: " << matl << "\n";
	      switch(td->getType()){
	      case TypeDescription::ParticleVariable:
		switch(subtype->getType()){
		case TypeDescription::double_type:
		  compareParticles<double>(da1, da2, var, matl, patch, patch2,
					   time, time2, abs_tolerance, rel_tolerance);
		  break;
		case TypeDescription::Point:
		  compareParticles<Point>(da1, da2, var, matl, patch, patch2,
					  time, time2, abs_tolerance, rel_tolerance);
		  break;
		case TypeDescription::Vector:
		  compareParticles<Vector>(da1, da2, var, matl, patch, patch2,
					   time, time2, abs_tolerance, rel_tolerance);
		  break;
		case TypeDescription::Matrix3:
		  compareParticles<Matrix3>(da1, da2, var, matl, patch, patch2,
					    time, time2, abs_tolerance, rel_tolerance);
		  break;
		default:
		  cerr << "ParticleVariable of unknown type: " << subtype->getType() << '\n';
		  exit(-1);
		}
		break;
	      case TypeDescription::NCVariable:
		switch(subtype->getType()){
		case TypeDescription::double_type:
		  compareFields< NCVariable<double> >
		    (da1, da2, var, matl, patch, patch2, time,time2, abs_tolerance, rel_tolerance,
		     patch->getNodeIterator(), patch2->getNodeIterator());
		  break;
		case TypeDescription::Point:
		  compareFields< NCVariable<Point> >
		    (da1, da2, var, matl, patch, patch2, time,time2, abs_tolerance, rel_tolerance,
		     patch->getNodeIterator(), patch2->getNodeIterator());
		  break;
		case TypeDescription::Vector:
		  compareFields< NCVariable<Vector> >
		    (da1, da2, var, matl, patch, patch2, time,time2, abs_tolerance, rel_tolerance,
		     patch->getNodeIterator(), patch2->getNodeIterator());
		  break;
		case TypeDescription::Matrix3:
		  compareFields< NCVariable<Matrix3> >
		    (da1, da2, var, matl, patch, patch2, time,time2, abs_tolerance, rel_tolerance,
		     patch->getNodeIterator(), patch2->getNodeIterator());
		  break;
		default:
		  cerr << "NC Variable of unknown type: " << subtype->getType() << '\n';
		  exit(-1);
		}
		break;
	      case TypeDescription::CCVariable:
		switch(subtype->getType()){
		case TypeDescription::double_type:
		  compareFields< CCVariable<double> >
		    (da1, da2, var, matl, patch, patch2, time,time2, abs_tolerance, rel_tolerance,
		     patch->getNodeIterator(), patch2->getNodeIterator());
		  break;
		case TypeDescription::Point:
		  compareFields< CCVariable<Point> >
		    (da1, da2, var, matl, patch, patch2, time,time2, abs_tolerance, rel_tolerance,
		     patch->getNodeIterator(), patch2->getNodeIterator());
		  break;
		case TypeDescription::Vector:
		  compareFields< CCVariable<Vector> >
		    (da1, da2, var, matl, patch, patch2, time,time2, abs_tolerance, rel_tolerance,
		     patch->getNodeIterator(), patch2->getNodeIterator());
		  break;
		case TypeDescription::Matrix3:
		  compareFields< CCVariable<Matrix3> >
		    (da1, da2, var, matl, patch, patch2, time,time2, abs_tolerance, rel_tolerance,
		     patch->getNodeIterator(), patch2->getNodeIterator());
		  break;
		default:
		  cerr << "CC Variable of unknown type: " << subtype->getType() << '\n';
		  exit(-1);
		}
		break;
	      default:
		cerr << "Variable of unknown type: " << td->getType() << '\n';
		exit(-1);
	      }
	    }
	  }
	}
      }
    }
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }

  if (tolerance_error)
    exit(2);

  return 0;
}
