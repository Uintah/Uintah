/*
 *  dumpfields.cc: Print out a uintah data archive
 *
 *  The fault of 
 *   Andrew D. Brydon
 *   Los Alamos National Laboratory
 *   Mar 2004
 *
 *  Based on puda, written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 *  Copyright (C) 2000 U of U
 */

#include <assert.h>

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/Endian.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/Array3.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

Mutex cerrLock( "cerrLock" );

vector<string> 
split(const string & s, char sep, bool skipmult=false)
{
  vector<string> res;
  
  string::size_type ls(s.size());
  string::size_type p0(0), p1;
  
  while( (p1=s.find(sep,p0))!=string::npos )
    {
      // Assert(p1>=0 && p1<ls, "string separator pointer in range");
      // Assert(p1>=p0,         "non-negative sub-string length");
      
      res.push_back(s.substr(p0, p1-p0));
      p0 = p1+1;
      if(skipmult)
	{
	  while(p0<ls && s[p0]==sep)
	    p0++;
	}
    }
  
  if(p0<s.size() || (p0>0&&p0==s.size()) ) // end case of splitting '1,2,' into three
    {
      res.push_back(s.substr(p0));
    }
  
  return res;
}

// -----------------------------------------------------------------------------
// generate number of scalar diagnostics from different element types

template <typename ElemT>
class ScalarDiagGen {
};

template <>
class ScalarDiagGen<float> {
public:
  static const int number = 1;
  const string name(int idiag) const { return ""; }
  double gen(double v, int idiag) const { return v; }
};

template <>
class ScalarDiagGen<double> {
public:
  static const int number = 1;
  const string name(int idiag) const { return ""; }
  double gen(double v, int idiag) const { return v; }
};

template <>
class ScalarDiagGen<Vector> {
public: 
  static const int number = 1;
  const string name(int idiag) const { return ""; }
  double gen(const Vector & v, int idiag) const { return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
};

template <>
class ScalarDiagGen<Point> {
public:
  static const int number = 1;
  const string name(int idiag) const { return "mag"; }
  double gen(const Point & v, int idiag) const { return sqrt(v(0)*v(0)+v(1)*v(1)+v(2)*v(2)); }
};

template <>
class ScalarDiagGen<Matrix3> {
public:
  static const int number = 5;
  const string name(int idiag) const 
  { 
    switch(idiag) {
    case 0:
      return "mag"; 
    case 1:
      return "equiv"; 
    case 2:
      return "trace"; 
    case 3:
      return "maxabs"; 
    case 4:
      return "princip"; 
    default:
      return "";
    }
  }
  double gen(const Matrix3 & t, int idiag) const 
  {
    switch(idiag) {
    case 0:
      return t.Norm();
    case 1:
      {
        const double  p  = t.Trace()/3.;
        const Matrix3 ti = t - Matrix3(p,0,0, 0,p,0, 0,0,p);
        return ti.Norm();
      }
    case 2:
      return t(0,0)+t(1,1)+t(2,2);
      // return t.Trace();
    case 3:
      return t.MaxAbsElem();
    case 4:
      {
        double e1,e2,e3;
        int ne = t.getEigenValues(e1,e2,e3);
        if(ne==3) {
          if(fabs(e1)>fabs(e3)) {
            return e1;
          } else {
            return e3;
          }
        } else {
          return 0.;
        }
      }
      return t.MaxAbsElem();
    default:
      return 0.;
    }
  }
};

// -----------------------------------------------------------------------------

class Dumper {
public:
  Dumper(DataArchive * da, string datadir);
  virtual ~Dumper();
  
  virtual void addField(string fieldname, const Uintah::TypeDescription * type) = 0;
  
  class Step {
  public:
    Step(int index_, double time_) : index(index_), time(time_) {}
    
    virtual ~Step();
    
    virtual void storeGrid () = 0;
    virtual void storeField(string fieldname, const Uintah::TypeDescription * type) = 0;
    virtual string infostr() const = 0;
    
    int    index;
    double time;
  };
  
  virtual Step * addStep(int index, double time) = 0;
  virtual void finishStep(Step *) = 0;

protected:
  DataArchive* da;
  string       datadir;
};

Dumper::Dumper(DataArchive * da_, string datadir_)
  : da(da_),datadir(datadir_)
{
}

Dumper::~Dumper() {}
Dumper::Step::~Step() {}

// -----------------------------------------------------------------------------

#define ONEDIM_DIM 2

class TextDumper : public Dumper 
{
public:
  TextDumper(DataArchive* da, string datadir, bool onedim=false, bool tseries=false);

  void addField(string fieldname, const Uintah::TypeDescription * type) {}
  
  class Step : public Dumper::Step {
  public:
    Step(DataArchive * da, string datadir, int index, double time, bool onedim, bool tseries);
    
    void storeGrid ();
    void storeField(string fieldname, const Uintah::TypeDescription * type);
    
    string infostr() const { return stepdname; }
    
  private:
    string stepdname;
    
  private:
    DataArchive* da;
    string datadir;
    bool onedim, tseries;
  };
  
  //
  Step * addStep(int index, double time);
  void   finishStep(Dumper::Step * s);
  
  static string time_string(double tval);
  static string mat_string (int    mval);

private:
  static string makeFileName(string raydatadir, string time_file="", string variable_file="", 
			     string materialType_file="", string extension="");
  
private:
  ofstream idxos;
  bool onedim, tseries;
  FILE*        filelist;
};

TextDumper::TextDumper(DataArchive* da_, string datadir_, bool onedim_, bool tseries_)
  : Dumper(da_, datadir_+"_text"), onedim(onedim_), tseries(tseries_)
{
  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  
  // Create a directory if it's not already there.
  // The exception occurs when the directory is already there
  // and the Dir.create fails.  This exception is ignored. 
  Dir dumpdir;
  try {
    dumpdir.create(datadir);
  } catch (Exception& e) {
    ;
  }
  
  // set up the file that contains a list of all the files
  string filelistname = datadir + string("/") + string("timelist");
  filelist = fopen(filelistname.c_str(),"w");
  if (!filelist) {
    cerr << "Can't open output file " << filelistname << endl;
    abort();
  }
}

string
TextDumper::time_string(double tval)
{
  ostringstream b;
  b.setf(ios::fixed,ios::floatfield);
  b.precision(8);
  b << setw(12) << setfill('0') << tval;
  string btxt(b.str());
  string r;
  for(string::const_iterator bit(btxt.begin());bit!=btxt.end();bit++) {
    char c = *bit;
    if(c=='.' || c==' ')
      r += '_';
    else
      r += c;
  }
  return r;
}

string
TextDumper::mat_string(int mval)
{
  ostringstream b;
  b << setw(4) << setfill('0') << mval;
  return b.str();
}

string
TextDumper::makeFileName(string datadir, string time_file, string variable_file, 
		     string materialType_file, string extension) {
  string datafile;
  if (datadir != "")
    datafile+= datadir + string("/");
  if (time_file!="")
    datafile+= string("TS_") + time_file + string("/");
  if (variable_file != "")
    datafile+= string("VAR_") + variable_file + string(".");
  if (materialType_file != "")
    datafile+= string("MT_") + materialType_file + string(".");
  if(extension != "")
    datafile+= extension;
  return datafile;
}

TextDumper::Step * 
TextDumper::addStep(int index, double time)
{
  return new Step(da, datadir, index, time, onedim, tseries);
}  

void
TextDumper::finishStep(Dumper::Step * s)
{
  fprintf(filelist, "%10d %16.8g  %20s\n", s->index, s->time, s->infostr().c_str());
}

TextDumper::Step::Step(DataArchive * da_, string datadir_, int index_, double time_,  bool onedim_, bool tseries_)
  : 
  Dumper::Step(index_, time_),
  da(da_), datadir(datadir_), onedim(onedim_), tseries(tseries_)
{
  if(tseries)
    stepdname = TextDumper::makeFileName(datadir, TextDumper::time_string(time));
  else
    stepdname = TextDumper::makeFileName(datadir);
  if(!tseries || index==1)
    {
      Dir stepdir;
      try {
        stepdir.create(stepdname);
      } catch (...) {
        ; // 
      }
    }
}

void
TextDumper::Step::storeGrid()
{
  GridP grid = da->queryGrid(time);
  
  // dont store grid info in flat text mode
}

bool _outside(IntVector p, IntVector mn, IntVector mx)
{
  return  ( p[0]<mn[0] || p[0]>=mx[0] ||
	    p[1]<mn[1] || p[1]>=mx[1] ||
	    p[2]<mn[2] || p[2]>=mx[2] );
}

void
TextDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
{
  GridP grid = da->queryGrid(time);
  
  cout << "   " << fieldname << endl;
  
  const Uintah::TypeDescription* subtype = td->getSubType();
  
  int nmats = 0;
  for(int l=0;l<=0;l++) {
    LevelP level = grid->getLevel(l);
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      ConsecutiveRangeSet matls= da->queryMaterials(fieldname, patch, time);
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
	int matl = *matlIter;
	if(matl>=nmats) nmats = matl+1;
      }
    }
  }
  
  vector<ofstream *> outfiles(nmats);
  for(int imat=0;imat<nmats;imat++) {
    string matname = TextDumper::mat_string(imat);
    string tname   = "";
    if(!tseries) tname = time_string(time);
    string fname   = TextDumper::makeFileName(datadir, tname, fieldname, matname, "txt");
    
    cout << "     " << fname << endl;
    if(!tseries || index==1)
      {
        outfiles[imat] = scinew ofstream(fname.c_str());
        *(outfiles[imat]) << "# time = " << time << ", field = " << fieldname << ", mat " << matname << endl;
      }
    else
      {
        outfiles[imat] = scinew ofstream(fname.c_str(), ios::app);
        *(outfiles[imat]) << endl << endl;
      }
  }
  
  // only support level 0 for now
  for(int l=0;l<=0;l++) {
    LevelP level = grid->getLevel(l);
    
    IntVector minind, maxind;
    level->findNodeIndexRange(minind, maxind);
    if(this->onedim) {
      IntVector ghostl(-minind);
      minind[0] += ghostl[0];
      maxind[0] -= ghostl[0];
      for(int id=0;id<3;id++) {
        if(id!=ONEDIM_DIM) {
          minind[id] = (maxind[id]+minind[id])/2;
          maxind[id] = minind[id]+1;
        }
      }
    }
    
    for(Level::const_patchIterator iter = level->patchesBegin();
	iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;
	    
      ConsecutiveRangeSet matls = da->queryMaterials(fieldname, patch, time);
	    
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
	  matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	      
	bool no_match = false;
	      
	ParticleVariable<Point> partposns;
	if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	  da->query(partposns, "p.x", matl, patch, time);
	}
	
	switch(subtype->getType()) {
	case Uintah::TypeDescription::float_type:
	  {
	    switch(td->getType()){
	    case Uintah::TypeDescription::NCVariable:
	      {
		NCVariable<float> value;
		da->query(value, fieldname, matl, patch, time);
		
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter] << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::CCVariable:
	      {
		CCVariable<float> value;
		da->query(value, fieldname, matl, patch, time);
		
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter] << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::ParticleVariable:
	      {
		ParticleVariable<float> value;
		da->query(value, fieldname, matl, patch, time);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << (value[*iter]) << " " 
				  << endl;
		}
	      } break;
	    default:
	      no_match = true;
	    }
	  } break;
	case Uintah::TypeDescription::double_type:
	  {
	    switch(td->getType()){
	    case Uintah::TypeDescription::NCVariable:
	      {
		NCVariable<double> value;
		da->query(value, fieldname, matl, patch, time);
			
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter] << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::CCVariable:
	      {
		CCVariable<double> value;
		da->query(value, fieldname, matl, patch, time);
			
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter] << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::ParticleVariable:
	      {
		ParticleVariable<double> value;
		da->query(value, fieldname, matl, patch, time);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << (value[*iter]) << " " 
                                  << endl;
		}
	      } break;
	    default:
	      no_match = true;
	    }
	  } break;
	case Uintah::TypeDescription::Point:
	  {
	    switch(td->getType()){
	    case Uintah::TypeDescription::NCVariable:
	      {
		NCVariable<Point> value;
		da->query(value, fieldname, matl, patch, time);
		    
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter](0) << " "
                                  << value[*iter](1) << " "
                                  << value[*iter](2) << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::CCVariable:
	      {
		CCVariable<Point> value;
		da->query(value, fieldname, matl, patch, time);
		    
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter](0) << " "
                                  << value[*iter](1) << " "
                                  << value[*iter](2) << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::ParticleVariable:
	      {
		ParticleVariable<Point> value;
		da->query(value, fieldname, matl, patch, time);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter](0) << " "
                                  << value[*iter](1) << " "
                                  << value[*iter](2) << " "
                                  << endl;
		}
	      } break;
	    default:
	      no_match = true;
	    }
	  } break;
	case Uintah::TypeDescription::Vector:
	  {
	    switch(td->getType()){
	    case Uintah::TypeDescription::NCVariable:
	      {
		NCVariable<Vector> value;
		da->query(value, fieldname, matl, patch, time);
		      
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter][0] << " "
                                  << value[*iter][1] << " "
                                  << value[*iter][2] << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::CCVariable:
	      {
		CCVariable<Vector> value;
		da->query(value, fieldname, matl, patch, time);
		      
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter][0] << " "
                                  << value[*iter][1] << " "
                                  << value[*iter][2] << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::ParticleVariable:
	      {
		ParticleVariable<Vector> value;
		da->query(value, fieldname, matl, patch, time);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter][0] << " "
                                  << value[*iter][1] << " "
                                  << value[*iter][2] << " "
                                  << endl;
		}
	      } break;
	    default:
	      no_match = true;
	    }
	  } break;
	case Uintah::TypeDescription::Matrix3:
	  {
	    switch(td->getType()){
	    case Uintah::TypeDescription::NCVariable:
	      {
		NCVariable<Matrix3> value;
		da->query(value, fieldname, matl, patch, time);
		    
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter](0,0) << " "
                                  << value[*iter](0,1) << " "
                                  << value[*iter](0,2) << " "
                                  << value[*iter](1,0) << " "
                                  << value[*iter](1,1) << " "
                                  << value[*iter](1,2) << " "
                                  << value[*iter](2,0) << " "
                                  << value[*iter](2,1) << " "
                                  << value[*iter](2,2) << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::CCVariable:
	      {
		CCVariable<Matrix3> value;
		da->query(value, fieldname, matl, patch, time);
		    
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(_outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter](0,0) << " "
                                  << value[*iter](0,1) << " "
                                  << value[*iter](0,2) << " "
                                  << value[*iter](1,0) << " "
                                  << value[*iter](1,1) << " "
                                  << value[*iter](1,2) << " "
                                  << value[*iter](2,0) << " "
                                  << value[*iter](2,1) << " "
                                  << value[*iter](2,2) << " "
                                  << endl;
		}
	      } break;
	    case Uintah::TypeDescription::ParticleVariable:
	      {
		ParticleVariable<Matrix3> value;
		da->query(value, fieldname, matl, patch, time);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries) *outfiles[matl] << time << " ";
                  *outfiles[matl]  << xpt(0) << " " 
                                   << xpt(1) << " " 
                                   << xpt(2) << " ";
		  *outfiles[matl] << value[*iter](0,0) << " "
                                  << value[*iter](0,1) << " "
                                  << value[*iter](0,2) << " "
                                  << value[*iter](1,0) << " "
                                  << value[*iter](1,1) << " "
                                  << value[*iter](1,2) << " "
                                  << value[*iter](2,0) << " "
                                  << value[*iter](2,1) << " "
                                  << value[*iter](2,2) << " "
                                  << endl;
		}
	      } break;
	    default:
	      no_match = true;
	    }
	  } break;
	default:
	  no_match = true;
	} // type switch
        
	if (no_match)
	  cerr << "Unexpected type of " << td->getName() << "," << subtype->getName() << endl;
        
      } // materials
    } // patches
  } // levels 
  
  for(int imat=0;imat<nmats;imat++) {
    delete outfiles[imat];
  }
}    

// -----------------------------------------------------------------------------

class EnsightDumper : public Dumper 
{
private:
  // dump field as text or binary
  struct FldDumper { 
    // ensight is very fussy about the format of the text, so it's nice
    // to have this all in one place
    FldDumper(bool bin_) : bin(bin_), os(0) {}
    
    void setstrm(ostream * os_) { os = os_; }
    void unsetstrm() { os = 0; }
    
    void textfld(string v, int width=80) {
      *os << setw(width) << setiosflags(ios::left) << v;
    }
    
    void textfld(string v, int width, int binwidth) {
      if(this->bin)
        *os << setw(binwidth)  << setiosflags(ios::left) << v;
      else
        *os << setw(width)  << setiosflags(ios::left) << v;
    }
    
    void textfld(int v, int width=10) {
      *os << setiosflags(ios::left) << setw(width) << v;
    }
    
    void textfld(float v) { 
      char b[13];
      snprintf(b, 13, "%12.5e", v);
      *os << b;
    }
    
    void textfld(double v) { 
      char b[13];
      snprintf(b, 13, "%12.5e", v);
      *os << b;
    }
    
    void numfld(int v, int wid=10) { 
      if(!this->bin) {
        textfld(v, wid);
      } else {
        os->write((char*)&v, sizeof(int));
      }
    }
    
    void endl() { 
      if(!this->bin) {
        *os << std::endl;
      }
    }
    
    void numfld(float v) { 
      v = (fabs(v)<FLT_MIN)?0.:v;
      if(!this->bin) {
        textfld(v);
      } else {
        os->write((char*)&v, sizeof(float));
      }
    }
    
    void numfld(double v) { 
      numfld((float)v);
    }
    
    bool bin;
    ostream * os;
  };

  struct Data {
    Data(DataArchive * da_, string dir_, FldDumper * dumper_, bool onemesh_, bool withpart_)
      : da(da_), dir(dir_), dumper(dumper_), onemesh(onemesh_), withpart(withpart_) {}
    
    DataArchive * da;
    string dir;
    FldDumper * dumper;
    bool onemesh, withpart;
  };
  
public:
  EnsightDumper(DataArchive* da, string datadir, bool binmode=false, bool onemesh=false, bool withpart=false);
  ~EnsightDumper();
  
  void addField(string fieldname, const Uintah::TypeDescription * type);
  
  class Step : public Dumper::Step {
    friend class EnsightDumper;
  private:
    Step(Data * data, int index, double time, int fileindex);
    
  public:
    void storeGrid ();
    void storeField(string filename, const Uintah::TypeDescription * type);
    
    string infostr() const { return stepdesc; }
    
  private:
    void storePartField(string filename, const Uintah::TypeDescription * type);
    void storeGridField(string filename, const Uintah::TypeDescription * type);
    
  private:
    int    fileindex;
    string stepdname;
    string stepdesc;
    
  private:
    Data * data;
    bool needmesh;
    IntVector vshape, minind;
  };
  friend class Step;
  
  //
  Step * addStep(int index, double time);
  void finishStep(Dumper::Step * step);
  
private:  
  static string tensor_diagname(int icomp) {
    switch(icomp) {
    case 0: return "_mag"; 
    case 1: return "_equiv"; 
    case 2: return "_trace"; 
    case 3: return "_maxabs"; 
    }
    return "_BLAH";
  }
  
private:
  int           nsteps;
  ofstream      casestrm;
  int           tscol;
  ostringstream tsstrm;
  FldDumper     flddumper;
  Data          data;
};

EnsightDumper::EnsightDumper(DataArchive* da_, string datadir_, bool bin_, bool onemesh_, bool withpart_)
  : 
  Dumper(da_, datadir_+"_ensight"), nsteps(0), flddumper(bin_), 
  data(da_,datadir_+"_ensight",&flddumper,onemesh_,withpart_)
{
  cerr << "using bin = " << bin_ << endl;
  
  // Create a directory if it's not already there.
  Dir dumpdir;
  try {
    dumpdir.create(data.dir);
  } catch (Exception& e) {
    ; // ignore failure
  }
  
  // set up the file that contains a list of all the files
  string casefilename = data.dir + string("/") + string("ensight.case");
  casestrm.open(casefilename.c_str());
  if (!casestrm) {
    cerr << "Can't open output file " << casefilename << endl;
    abort();
  }
  cout << "     " << casefilename << endl;
  
  // header
  casestrm << "FORMAT" << endl;
  casestrm << "type: ensight gold" << endl;
  casestrm << endl;
  casestrm << "GEOMETRY" << endl;
  if(data.onemesh)
    casestrm << "model:       gridgeo" << endl;
  else
    casestrm << "model:   1   gridgeo****" << endl;
  casestrm << endl;
  casestrm << "VARIABLE" << endl;
  
  // time step data
  tscol  = 0;
  tsstrm << "time values: ";;
}

EnsightDumper::~EnsightDumper()
{
  casestrm << endl;
  casestrm << "TIME" << endl;
  casestrm << "time set: 1 " << endl;
  casestrm << "number of steps: " << nsteps << endl;
  casestrm << "filename start number: 0" << endl;
  casestrm << "filename increment: 1" << endl;
  casestrm << tsstrm.str() << endl;
}

static string nodots(string n)
{
  string r;
  for(string::const_iterator nit(n.begin());nit!=n.end();nit++)
    if(*nit!='.')
      r += *nit;
  return r;
}

void
EnsightDumper::addField(string fieldname, const Uintah::TypeDescription * td)
{
  string subtypestr, typestr;
  if(td->getType()==Uintah::TypeDescription::NCVariable)
    typestr = "node";
  else if(td->getType()==Uintah::TypeDescription::CCVariable)
    typestr = "cell";
  else
    return; // dont dump particle data
  
  const Uintah::TypeDescription* subtype = td->getSubType();
  if(subtype->getType()==Uintah::TypeDescription::float_type || 
     subtype->getType()==Uintah::TypeDescription::double_type )
    subtypestr = "scalar";
  else if(subtype->getType()==Uintah::TypeDescription::Point || 
	  subtype->getType()==Uintah::TypeDescription::Vector )
    subtypestr = "vector";
  
  if(subtype->getType()==Uintah::TypeDescription::Matrix3)
    {
      for(int idiag=0;idiag<4;idiag++)
	casestrm << "scalar per " << typestr << ": 1 " 
                 << fieldname << "_" << tensor_diagname(idiag) 
                 << " " << nodots(fieldname) << tensor_diagname(idiag) << "****" << endl;
    }
  else
    {
      casestrm << subtypestr << " per " << typestr << ": 1 " << fieldname << " " << nodots(fieldname) << "****" << endl;
    }
}

EnsightDumper::Step * 
EnsightDumper::addStep(int index, double time)
{
  Step * res = new Step(&data, index, time, nsteps);
  nsteps++;
  return res;
}

void
EnsightDumper::finishStep(Dumper::Step * step)
{
  tsstrm << step->time << " ";
  if(++tscol==10) {
    tsstrm << endl << "  ";
    tscol = 0;
  }
}

EnsightDumper::Step::Step(Data * data_, int index_, double time_, int fileindex_)
  :
  Dumper::Step(index_, time_),
  fileindex(fileindex_),
  data(data_),
  needmesh(!data->onemesh||(fileindex_==0))
{
}

void
EnsightDumper::Step::storeGrid()
{
  GridP grid = data->da->queryGrid(this->time);
  FldDumper * fd = data->dumper;
  
  // only support level 0 for now
  int lnum = 0;
  LevelP level = grid->getLevel(lnum);
  
  // store to basename/grid.geo****
  char goutname[1024];
  char poutname[1024];
  if(data->onemesh) {
    snprintf(goutname, 1024, "%s/gridgeo", data->dir.c_str());
  } else {
    snprintf(goutname, 1024, "%s/gridgeo%04d", data->dir.c_str(), this->fileindex);
  }
  snprintf(poutname, 1024, "%s/partgeo%04d", data->dir.c_str(), this->fileindex);
  
  if(this->needmesh) {
    
    // find ranges
    // dont bother dumping ghost stuff to ensight
    IntVector minind, maxind;
    level->findNodeIndexRange(minind, maxind);
    this->minind = minind;
    this->vshape = (maxind-minind);
  
    cout << "  " << goutname << endl;
    cout << "   minind = " << minind << endl;
    cout << "   maxind = " << maxind << endl;
    cout << "   vshape = " << vshape << endl;
    cout << endl;
  
    ofstream gstrm(goutname);
    fd->setstrm(&gstrm);
  
    if(fd->bin) fd->textfld("C Binary",79,80);
    fd->textfld("grid description",79,80) ; fd->endl();
    fd->textfld("grid description",79,80) ; fd->endl();
    fd->textfld("node id off",     79,80) ; fd->endl();
    fd->textfld("element id off",  79,80) ; fd->endl();
  
    fd->textfld("part",            79,80) ; fd->endl();
    fd->numfld(1); fd->endl();
    fd->textfld("3d regular block",79,80); fd->endl();
    fd->textfld("block",79,80); fd->endl();
    fd->numfld(vshape(0),10);
    fd->numfld(vshape(1),10);
    fd->numfld(vshape(2),10);
    fd->endl();
    
    for(int id=0;id<3;id++) {
      for(int k=minind[2];k<maxind[2];k++) {
        for(int j=minind[1];j<maxind[1];j++) {
          for(int i=minind[0];i<maxind[0];i++) {
            fd->numfld(level->getNodePosition(IntVector(i,j,k))(id)) ; fd->endl();
          }
        }
      }
    }
    
    fd->unsetstrm();
  }
  
  if(data->withpart) {
    // store particles   
    cout << "  " << poutname << endl;
    
    ofstream pstrm(poutname);
    pstrm << "particle description" << endl;
    pstrm << "particle coordinates" << endl;
    streampos nspot = pstrm.tellp();
    pstrm << setw(8) << "XXXXXXXX" << endl;
    int nparts = 0;
  
    for(Level::const_patchIterator iter = level->patchesBegin();
        iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;
    
      ConsecutiveRangeSet matls = data->da->queryMaterials("p.x", patch, time);
    
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
          matlIter != matls.end(); matlIter++) {
        ParticleVariable<Matrix3> value;
        
        const int matl = *matlIter;
      
        ParticleVariable<Point> partposns;
        data->da->query(partposns, "p.x", matl, patch, time);
      
        ParticleSubset* pset = partposns.getParticleSubset();
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++) {
        
          Point xpt = partposns[*iter];
          pstrm << setw(8) << ++nparts;
          for(int id=0;id<3;id++) {
            char b[13];
            snprintf(b, 12, "%12.5f", xpt(id));
            pstrm << b << endl;
          }
          pstrm << endl;
        }
      
      }
    }
    cout << "   nparts = " << nparts << endl;
    cout << endl;
  
    pstrm.seekp(nspot);
    pstrm << setw(8) << setfill(' ') << nparts;
  }
  
}

void
EnsightDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
{
  if(td->getType()==Uintah::TypeDescription::ParticleVariable) 
    this->storePartField(fieldname, td);
  else 
    this->storeGridField(fieldname, td);
}

void
EnsightDumper::Step::storeGridField(string fieldname, const Uintah::TypeDescription * td)
{
  GridP grid = data->da->queryGrid(this->time);
  FldDumper * fd = data->dumper;
  
  const Uintah::TypeDescription* subtype = td->getSubType();
  int ndiags;
  if(subtype->getType()==Uintah::TypeDescription::Matrix3)
    ndiags = 4;
  else
    ndiags = 1;
  for(int idiag=0;idiag<ndiags;idiag++)
    {
      char outname[1024];
      
      if(subtype->getType()==Uintah::TypeDescription::Matrix3)
	snprintf(outname, 1024, "%s/%s%s%04d", 
		 data->dir.c_str(), nodots(fieldname).c_str(), 
		 tensor_diagname(idiag).c_str(), this->fileindex);
      else
	snprintf(outname, 1024, "%s/%s%04d", data->dir.c_str(), nodots(fieldname).c_str(), this->fileindex);
      
      int icol(0);
      
      cout << "  " << outname << endl;
      ofstream vstrm(outname);
      fd->setstrm(&vstrm);
      
      ostringstream descb;
      descb << "data field " << nodots(fieldname) << " at " << this->time;
      fd->textfld(descb.str(),79,80); fd->endl();
      fd->textfld("part",     79,80); fd->endl();
      fd->numfld(1); fd->endl();
      fd->textfld("block",    79,80); fd->endl();
      
      int nc = 0;
      if(subtype->getType()==Uintah::TypeDescription::float_type || 
	 subtype->getType()==Uintah::TypeDescription::double_type )
	nc = 1;
      else if(subtype->getType()==Uintah::TypeDescription::Point || 
	      subtype->getType()==Uintah::TypeDescription::Vector )
	nc = 3;
      else if(subtype->getType()==Uintah::TypeDescription::Matrix3)
	nc = 1;
      if(nc==0) return;
      
      // only support level 0 for now
      int lnum = 0;
      LevelP level = grid->getLevel(lnum);
      
      // const int iscell = (subtype->getType()==Uintah::TypeDescription::NCVariable)?1:0;
      
      for(int icomp=0;icomp<nc;icomp++) {
        
	Uintah::Array3<double> vals(vshape[0],vshape[1],vshape[2]);
	for(Uintah::Array3<double>::iterator vit(vals.begin());vit!=vals.end();vit++)
	  {
	    *vit = 0.;
	  }
        
	for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
	  const Patch* patch = *iter;
          cout << "loading patch " << patch->getID() << endl;
          
          IntVector ilow, ihigh;
          patch->computeVariableExtents(subtype->getType(), IntVector(0,0,0), Ghost::None, 0, ilow, ihigh);
          
          ConsecutiveRangeSet matls = data->da->queryMaterials(fieldname, patch, time);
          
          // loop over materials
          for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
            int i,j,k;
            const int matl = *matlIter;
            
            switch(subtype->getType()) {
            case Uintah::TypeDescription::float_type:
              {
                NCVariable<float> value;
                data->da->query(value, fieldname, matl, patch, time);
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  vals[ijk-minind] += value[ijk];
                }
	      } break;
	    case Uintah::TypeDescription::double_type:
	      {
                NCVariable<double> value;
                data->da->query(value, fieldname, matl, patch, time);
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  vals[ijk-minind] += value[ijk];
                }
	      } break;
	    case Uintah::TypeDescription::Point:
	      {
                NCVariable<Point> value;
                data->da->query(value, fieldname, matl, patch, time);
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  vals[ijk-minind] += value[ijk](icomp);
                }
	      } break;
	    case Uintah::TypeDescription::Vector:
	      {
                NCVariable<Vector> value;
                data->da->query(value, fieldname, matl, patch, time);
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  vals[ijk-minind] += value[ijk][icomp];
                }
	      } break;
	    case Uintah::TypeDescription::Matrix3:
	      {
		NCVariable<Matrix3> value;
		data->da->query(value, fieldname, matl, patch, time);
                
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  const Matrix3 t = value[ijk];
                  switch(idiag) {
                  case 0: // mag
                    vals[ijk-minind] = t.Norm();
                    break;
                  case 1: // equiv
                    {
                      const double  p  = t.Trace()/3.;
                      const Matrix3 ti = t - Matrix3(p,0,0, 0,p,0, 0,0,p);
                      vals[ijk-minind] = ti.Norm();
                    }
                    break;
                  case 2: // trace
                    vals[ijk-minind] = t.Trace();
                    break;
                  case 3: // max absolute element
                    vals[ijk-minind] = t.MaxAbsElem();
                    break;
		  }
		}
	      } break;
	    default:
	      ; // FIXME:
	    } // type switch
	  } // materials
	} // patches
	
	// dump this component as text
	for(int k=0;k<vshape[2];k++) 
	  for(int j=0;j<vshape[1];j++) 
	    for(int i=0;i<vshape[0];i++) {
	      fd->numfld(vals[IntVector(i,j,k)]);
	      if(++icol==1) 
		{
		  fd->endl();
		  icol = 0;
		}
	      if(icol!=0) fd->endl();
	    }
      } // components
      
      fd->unsetstrm();
      
    } // diags  
}

void
EnsightDumper::Step::storePartField(string fieldname, const Uintah::TypeDescription * td)
{
  // WRITEME
}

// -----------------------------------------------------------------------------

class DxDumper : public Dumper 
{
public:
  DxDumper(DataArchive* da, string datadir, bool binmode=false, bool onedim=false);
  ~DxDumper();
  
  void addField(string fieldname, const Uintah::TypeDescription * type);
  
  struct FldWriter { 
    FldWriter(string datadir, string fieldname);
    ~FldWriter();
    
    int dxobj;
    ofstream strm;
    list< pair<float,int> > timesteps;
  };
  
  class Step : public Dumper::Step {
  public:
    Step(DataArchive * da, string datadir, int index, double time, int fileindex, 
	 const map<string,FldWriter*> & fldwriters, bool bin, bool onedim);
    
    void storeGrid ();
    void storeField(string filename, const Uintah::TypeDescription * type);
    
    string infostr() const { return dxstrm.str()+"\n"+fldstrm.str()+"\n"; }
    
  private:
    DataArchive*  da;
    int           fileindex;
    string        datadir;
    ostringstream dxstrm, fldstrm;
    const map<string,FldWriter*> & fldwriters;
    bool bin, onedim;
  };
  friend class Step;
  
  //
  Step * addStep(int index, double time);
  void finishStep(Dumper::Step * step);
  
private:  
  int      nsteps;
  int      dxobj;
  ostringstream timestrm;
  ofstream dxstrm;
  map<string,FldWriter*> fldwriters;
  bool bin;
  bool onedim;
};

DxDumper::DxDumper(DataArchive* da_, string datadir_, bool bin_, bool onedim_)
  : Dumper(da_, datadir_+"_dx"), nsteps(0), dxobj(0), bin(bin_), onedim(onedim_)
{
  // Create a directory if it's not already there.
  Dir dumpdir;
  try {
    dumpdir.create(datadir);
  } catch (Exception& e) {
    ; // ignore failure
  }
  
  // set up the file that contains a list of all the files
  string indexfilename = datadir + string("/") + string("index.dx");
  dxstrm.open(indexfilename.c_str());
  if (!dxstrm) {
    cerr << "Can't open output file " << indexfilename << endl;
    abort();
  }
  cout << "     " << indexfilename << endl;
}

DxDumper::~DxDumper()
{
  dxstrm << "object \"udadata\" class series" << endl;
  dxstrm << timestrm.str();
  dxstrm << endl;
  dxstrm << "default \"udadata\"" << endl;
  dxstrm << "end" << endl;
  
  for(map<string,FldWriter*>::iterator fit(fldwriters.begin());fit!=fldwriters.end();fit++) {
    delete fit->second;
  }
}

void
DxDumper::addField(string fieldname, const Uintah::TypeDescription * td)
{
  fldwriters[fieldname] = new FldWriter(datadir, fieldname);
}

DxDumper::Step * 
DxDumper::addStep(int index, double time)
{
  DxDumper::Step * r = new Step(da, datadir, index, time, nsteps++, fldwriters, bin, onedim);
  return r;
}
  
void
DxDumper::finishStep(Dumper::Step * step)
{
  dxstrm << step->infostr() << endl;
  timestrm << "  member " << nsteps-1 << " " << step->time << " \"step " << nsteps << "\" " << endl;
}

DxDumper::FldWriter::FldWriter(string datadir, string fieldname)
  : dxobj(0)
{
  string outname = datadir+"/"+fieldname+".dx";
  strm.open(outname.c_str());
  if(!strm) {
    cerr << "Can't open output file " << outname << endl;
    abort();
  }
  cout << "     " << outname << endl;
}

DxDumper::FldWriter::~FldWriter()
{
  strm << "# time series " << endl;
  strm << "object " << ++dxobj << " series" << endl;
  int istep(0);
  for(list< pair<float,int> >::iterator tit(timesteps.begin());tit!=timesteps.end();tit++)
    {
      strm << "  member " << istep++ << " " << tit->first << " " << tit->second << endl;
    }
  strm << endl;
  
  strm << "default " << dxobj << endl;
  strm << "end" << endl;
}

DxDumper::Step::Step(DataArchive * da_, string datadir_, int index_, double time_, int fileindex_, 
		     const map<string,DxDumper::FldWriter*> & fldwriters_, bool bin_, bool onedim_)
  :
  Dumper::Step(index_, time_),
  da(da_), 
  fileindex(fileindex_),
  datadir(datadir_),
  fldwriters(fldwriters_),
  bin(bin_), onedim(onedim_)
{
  fldstrm << "object \"step " << fileindex_+1 << "\" class group" << endl;
}

static double REMOVE_SMALL(double v)
{
  if(fabs(v)<FLT_MIN) return 0;
  else return v;
}

void
DxDumper::Step::storeGrid()
{
}

void
DxDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
{
  FldWriter * fldwriter = fldwriters.find(fieldname)->second;
  ostream & os = fldwriter->strm;
  
  GridP grid = da->queryGrid(this->time);
  
  const Uintah::TypeDescription* subtype = td->getSubType();
  
  // only support level 0 for now
  int lnum = 0;
  LevelP level = grid->getLevel(lnum);
  
  string dmode;
  if(bin) {
    if(isLittleEndian()) dmode = " lsb";
    else                 dmode = " msb";
  } else                 dmode = " text";
  
  // build positions
  int posnobj(-1);
  int connobj(-1);
  int dataobj(-1);
  int nparts(0);
  bool iscell(false);
  IntVector nnodes, strides, minind, midind, ncells;
  if(td->getType()!=Uintah::TypeDescription::ParticleVariable) {
    IntVector indlow, indhigh;
    level->findNodeIndexRange(indlow, indhigh);
    Point x0 = level->getAnchor();
    Vector dx = level->dCell();
    
    iscell = (td->getType()==Uintah::TypeDescription::CCVariable);
    int celllen = iscell?1:0;
    nnodes  = IntVector(indhigh-indlow+IntVector(1,1,1));
    ncells  = IntVector(indhigh-indlow);
    strides = IntVector(indhigh-indlow+IntVector(1-celllen,1-celllen,1-celllen));
    minind  = indlow;
    midind  = IntVector(ncells[0]/2, ncells[1]/2, ncells[2]/2);
    
    os << "# step " << this->index << " positions" << endl;
    if(this->onedim)
      {
	os << "object " << ++fldwriter->dxobj << " class array type float items " 
	   << nnodes[0] << " data follows " << endl;
	for(int i=0;i<nnodes[0];i++)
	  {
	    os << x0(0)+dx[0]*(i-minind[0]) << endl;
	  }
      }
    else
      {
	os << "object " << ++fldwriter->dxobj << " class gridpositions counts " 
	   << nnodes[0] << " " << nnodes[1] << " " << nnodes[2] << endl;
	os << "origin " << x0(0)-minind[0]*dx[0] << " " << x0(1)-minind[1]*dx[1] << " " << x0(2)-minind[2]*dx[2] << endl;
	os << "delta " << dx[0] << " " << 0. << " " << 0. << endl;
	os << "delta " << 0. << " " << dx[1] << " " << 0. << endl;
	os << "delta " << 0. << " " << 0. << " " << dx[2] << endl;
	os << endl;
      }
    posnobj = fldwriter->dxobj;
    
    os << "# step " << this->index << " connections" << endl;
    if(this->onedim)
      {
	os << "object " << ++fldwriter->dxobj << " class gridconnections counts " 
	   << nnodes[0] << endl; // dx wants node counts here !
	os << "attribute \"element type\" string \"lines\"" << endl;
      } 
    else
      {
	os << "object " << ++fldwriter->dxobj << " class gridconnections counts " 
	   << nnodes[0] << " " << nnodes[1] << " " << nnodes[2] << endl; // dx wants node counts here !
	os << "attribute \"element type\" string \"cubes\"" << endl;
      }
    os << "attribute \"ref\" string \"positions\"" << endl;
    os << endl;
    connobj = fldwriter->dxobj;
    
    if(this->onedim)
      nparts = strides(0);
    else
      nparts = strides(0)*strides(1)*strides(2);
    
  } else {
    nparts = 0;
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      
      ConsecutiveRangeSet matls = da->queryMaterials("p.x", patch, time);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	
	ParticleVariable<Point> partposns;
	da->query(partposns, "p.x", matl, patch, time);
	ParticleSubset* pset = partposns.getParticleSubset();
	nparts += pset->numParticles();
      }
    }
    
    os << "# step " << this->index << " positions" << endl;
    os << "object " << ++fldwriter->dxobj << " class array rank 1 shape 3 items " << nparts;
    os << dmode << " data follows " << endl;;
    
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      
      ConsecutiveRangeSet matls = da->queryMaterials("p.x", patch, time);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	
	ParticleVariable<Point> partposns;
	da->query(partposns, "p.x", matl, patch, time);
	ParticleSubset* pset = partposns.getParticleSubset();
	for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
	  Point xpt = partposns[*iter];
	  if(!bin)
	    os << xpt(0) << " " << xpt(1) << " " << xpt(2) << endl;
	  else
	    {
	      for(int ic=0;ic<3;ic++) {
		float v = xpt(ic);
		os.write((char *)&v, sizeof(float));
	      }
	    }
	}
      } // materials
      
    } // patches
    os << endl;
    posnobj = fldwriter->dxobj;
  }
  
  int ncomps, rank;
  string shp, source;
  vector<float> minval, maxval;
  
  if(1) { // FIXME: skip p.x
    int nvals;
    switch (td->getType()) { 
    case Uintah::TypeDescription::NCVariable:
      if(this->onedim)
	nvals = strides(0);
      else
	nvals = strides(0)*strides(1)*strides(2);
      source = "nodes";
      break;
    case Uintah::TypeDescription::CCVariable:
      if(this->onedim)
	nvals = strides(0);
      else
	nvals = strides(0)*strides(1)*strides(2);
      source = "cells";
      break;
    case Uintah::TypeDescription::ParticleVariable:
      nvals = nparts;
      source = "particles";
      break;
    default:
      fprintf(stderr, "unexpected field type\n");
      abort();
    }
    
    cout << "     " << fieldname << endl;
    switch(subtype->getType()) {
    case Uintah::TypeDescription::float_type:  rank = 0; ncomps = 1; shp = " "; break;
    case Uintah::TypeDescription::double_type: rank = 0; ncomps = 1; shp = " "; break;
    case Uintah::TypeDescription::Point:       rank = 1; ncomps = 3; shp = "shape 3"; break;
    case Uintah::TypeDescription::Vector:      rank = 1; ncomps = 3; shp = "shape 3"; break;
    case Uintah::TypeDescription::Matrix3:     rank = 2; ncomps = 9; shp = "shape 3 3"; break;
    default: 
      fprintf(stderr, "unexpected field sub-type\n");
      abort();
    };
  
    vector<float> vals(nvals*ncomps);
    for(vector<float>::iterator vit(vals.begin());vit!=vals.end();vit++) *vit = 0.;
    
    minval.resize(ncomps);
    maxval.resize(ncomps);
    for(int ic=0;ic<ncomps;ic++) {
      minval[ic] = FLT_MAX;
      maxval[ic] =-FLT_MAX;
    }

    int ipart(0);
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      
      ConsecutiveRangeSet matls = da->queryMaterials(fieldname, patch, time);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
	  matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	
	switch(subtype->getType()) {
	case Uintah::TypeDescription::float_type:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<float> value;
	      da->query(value, fieldname, matl, patch, time);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		float val = REMOVE_SMALL(value[*iter]);
		vals[ipart] = val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<float> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		if(this->onedim && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		float val = REMOVE_SMALL(value[*iter]);
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else {
	      NCVariable<float> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(this->onedim && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		float val = REMOVE_SMALL(value[*iter]);
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    }
	  } break;
	case Uintah::TypeDescription::double_type:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<double> value;
	      da->query(value, fieldname, matl, patch, time);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		float val = REMOVE_SMALL(value[*iter]);
		vals[ipart++] = val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<double> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		IntVector ind(*iter-minind);
		cout << "index: " << ind << endl;
		if(this->onedim && (ind[1]!=midind[1] ||ind[2]!=midind[2])) continue;
		
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		double val = REMOVE_SMALL(value[*iter]);
		cout << "  val = " << val << endl;
		
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else {
	      NCVariable<double> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(this->onedim && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		float val = REMOVE_SMALL(value[*iter]);
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    }
	  } break;
	case Uintah::TypeDescription::Point:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<Point> value;
	      da->query(value, fieldname, matl, patch, time);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		for(int ic=0;ic<3;ic++) {
		  float val = REMOVE_SMALL(value[*iter](ic));
		  vals[ipart++] = val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<Point> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		if(this->onedim && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int ic=0;ic<3;ic++){
		  float val = REMOVE_SMALL(value[*iter](ic));
		  vals[3*ioff+ic] += val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    } else {
	      NCVariable<Point> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(this->onedim && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int ic=0;ic<3;ic++){
		  float val = REMOVE_SMALL(value[*iter](ic));
		  vals[3*ioff+ic] += val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    }
	  } break;
	case Uintah::TypeDescription::Vector:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<Vector> value;
	      da->query(value, fieldname, matl, patch, time);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		for(int ic=0;ic<3;ic++) {
		  float val = REMOVE_SMALL(value[*iter][ic]);
		  vals[ipart++] = val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<Vector> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		if(this->onedim && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int ic=0;ic<3;ic++){
		  float val = REMOVE_SMALL(value[*iter][ic]);
		  vals[3*ioff+ic] += val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    } else {
	      NCVariable<Vector> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(this->onedim && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int ic=0;ic<3;ic++) {
		  float val = REMOVE_SMALL(value[*iter][ic]);
		  vals[3*ioff+ic] += val;
		  if(val<minval[ic]) minval[ic] = val;
		  if(val>maxval[ic]) maxval[ic] = val;
		}
	      }
	    }
	  } break;
	case Uintah::TypeDescription::Matrix3:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<Matrix3> value;
	      da->query(value, fieldname, matl, patch, time);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		for(int jc=0;jc<3;jc++)
		  for(int ic=0;ic<3;ic++) {
		    float val = REMOVE_SMALL(value[*iter](ic,jc));
		    vals[ipart++] = val;
		    if(val<minval[ic+jc*3]) minval[ic+jc*3] = val;
		    if(val>maxval[ic+jc*3]) maxval[ic+jc*3] = val;
		  }
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<Matrix3> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(this->onedim && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int jc=0;jc<3;jc++)
		  for(int ic=0;ic<3;ic++) {
		    float val = REMOVE_SMALL(value[*iter](ic,jc));
		    vals[9*ioff+ic+jc*3] += val;
		    if(val<minval[ic+jc*3]) minval[ic+jc*3] = val;
		    if(val>maxval[ic+jc*3]) maxval[ic+jc*3] = val;
		  }
	      }
	    } else {
	      NCVariable<Matrix3> value;
	      da->query(value, fieldname, matl, patch, time);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(this->onedim && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		for(int jc=0;jc<3;jc++)
		  for(int ic=0;ic<3;ic++) {
		    float val = REMOVE_SMALL(value[*iter](ic,jc));
		    vals[9*ioff+ic+jc*3] += val;
		    if(val<minval[ic+jc*3]) minval[ic+jc*3] = val;
		    if(val>maxval[ic+jc*3]) maxval[ic+jc*3] = val;
		  }
	      }
	    }
	  } break;
	default:
	  ;
	  // fprintf(stderr, "unexpected subtype\n");
	  // abort();
	} // subtype switch
	
      } // materials
    } // patches
    
    os << "# step " << this->index << " values" << endl;
    os << "object " << ++fldwriter->dxobj << " class array rank " << rank << " " << shp << " items " << nparts;
    os << dmode << " data follows " << endl;;
    int ioff = 0;
    for(int iv=0;iv<nvals;iv++) { 
      for(int ic=0;ic<ncomps;ic++) 
	if(!bin)
	  os << vals[ioff++] << " ";
	else
	  os.write((char *)&vals[ioff++], sizeof(float));
      if(!bin) os << endl;
    }
    os << endl;
    if(iscell) 
      {
	os << "attribute \"dep\" string \"connections\"" << endl;
      }
    else
      {
	os << "attribute \"dep\" string \"positions\"" << endl;
      }
    
    dataobj = fldwriter->dxobj;
  }
  
  // build field object
  os << "# step " << this->index << " " << fieldname << " field" << endl;
  os << "object " << ++fldwriter->dxobj << " class field" << endl;
  if(posnobj!=-1) os << "  component \"positions\" value " << posnobj << endl;
  if(connobj!=-1) os << "  component \"connections\" value " << connobj << endl;
  if(dataobj!=-1) os << "  component \"data\" value " << dataobj << endl;
  os << endl;
  
  fldwriter->timesteps.push_back( pair<float,int>(this->time, fldwriter->dxobj));
  
  int istep = this->fileindex+1;
  dxstrm << "# step " << istep << " " << fieldname << " minimum " << endl;
  dxstrm << "object \"" << fieldname << " " << istep << " min\" "
	 << "class array type float rank " << rank << " " << shp << " items " << 1
	 << " data follows" << endl;
  for(int ic=0;ic<ncomps;ic++)
    dxstrm << minval[ic] << " ";
  dxstrm << endl << endl;
  
  dxstrm << "# step " << istep << " " << fieldname << " maximum " << endl;
  dxstrm << "object \"" << fieldname << " " << istep << " max\" "
	 << "class array type float rank " << rank << " " << shp << " items " << 1
	 << " data follows" << endl;
  for(int ic=0;ic<ncomps;ic++)
    dxstrm << maxval[ic] << " ";
  dxstrm << endl << endl;
  
  dxstrm << "object \"" << fieldname << " " << istep << " filename\" class string \"" << fieldname << ".dx\"" << endl;
  dxstrm << endl;
  
  dxstrm << "# step " << istep << " info " << endl;
  dxstrm << "object \"" << fieldname << " " << istep << " info\" class group" << endl;
  dxstrm << "  member \"minimum\" \"" << fieldname << " " << istep << " min\"" << endl;
  dxstrm << "  member \"maximum\" \"" << fieldname << " " << istep << " max\"" << endl;
  dxstrm << "  member \"filename\" \"" << fieldname << " " << istep << " filename\"" << endl;
  dxstrm << "  attribute \"source\" string \"" << source << "\"" << endl;
  dxstrm << endl;
  
  fldstrm << "  member \"" << fieldname <<  "\" " << "\"" << fieldname << " " << istep << " info\"" << endl;
}

// -----------------------------------------------------------------------------

class HistogramDumper : public Dumper 
{
public:
  HistogramDumper(DataArchive* da, string datadir, int nbins=256);
  
  void addField(string fieldname, const Uintah::TypeDescription * type) {}
  
  class Step : public Dumper::Step {
  public:
    Step(DataArchive * da, string datadir, int index, double time, int nbins);
    
    void storeGrid () {}
    void storeField(string fieldname, const Uintah::TypeDescription * type);
    
    string infostr() const { return stepdname; }
    
  private:
    string stepdname;

    template <class ElemT>
    void _binvals(LevelP level, Uintah::TypeDescription::Type type_, 
                  const string & fieldname, int imat, int idiag,
                  double & minval, double & maxval, vector<int> & counts, string & ext);
    
  private:
    DataArchive* da;
    string datadir;
    int nbins;
  };
  
  //
  Step * addStep(int index, double time);
  void   finishStep(Dumper::Step * s);
  
  static string time_string(double tval);
  static string mat_string (int    mval);

private:
  static string makeFileName(string raydatadir, string time_file="", string variable_file="", 
			     string materialType_file="", string ext="");
  
private:
  ofstream idxos;
  int      nbins;
  FILE*    filelist;
};

HistogramDumper::HistogramDumper(DataArchive* da_, string datadir_, int nbins_)
  : Dumper(da_, datadir_+"_hist"), nbins(nbins_)
{
  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  
  // Create a directory if it's not already there.
  // The exception occurs when the directory is already there
  // and the Dir.create fails.  This exception is ignored. 
  Dir dumpdir;
  try {
    dumpdir.create(datadir);
  } catch (Exception& e) {
    ;
  }
  
  // set up the file that contains a list of all the files
  string filelistname = datadir + string("/") + string("timelist");
  filelist = fopen(filelistname.c_str(),"w");
  if (!filelist) {
    cerr << "Can't open output file " << filelistname << endl;
    abort();
  }
}

string
HistogramDumper::time_string(double tval)
{
  ostringstream b;
  b.setf(ios::fixed,ios::floatfield);
  b.precision(8);
  b << setw(12) << setfill('0') << tval;
  string btxt(b.str());
  string r;
  for(string::const_iterator bit(btxt.begin());bit!=btxt.end();bit++) {
    char c = *bit;
    if(c=='.' || c==' ')
      r += '_';
    else
      r += c;
  }
  return r;
}

string
HistogramDumper::mat_string(int mval)
{
  ostringstream b;
  b << setw(4) << setfill('0') << mval;
  return b.str();
}

string
HistogramDumper::makeFileName(string datadir, string time_file, string variable_file, 
                              string materialType_file, string ext) 
{
  string datafile;
  if (datadir != "")
    datafile+= datadir + string("/");
  datafile+= string("TS_") + time_file + string("/");
  if (variable_file != "")
    datafile+= string("VAR_") + variable_file + string(".");
  if (materialType_file != "")
    datafile+= string("MT_") + materialType_file + string(".");
  if (ext!="")
    datafile+=ext;
  return datafile;
}

HistogramDumper::Step * 
HistogramDumper::addStep(int index, double time)
{
  return new Step(da, datadir, index, time, nbins);
}  

void
HistogramDumper::finishStep(Dumper::Step * s)
{
  fprintf(filelist, "%10d %16.8g  %20s\n", s->index, s->time, s->infostr().c_str());
}

HistogramDumper::Step::Step(DataArchive * da_, string datadir_, int index_, double time_,  int nbins_)
  : 
  Dumper::Step(index_, time_),
  da(da_), datadir(datadir_), nbins(nbins_)
{
  stepdname = HistogramDumper::makeFileName(datadir, TextDumper::time_string(time));
  Dir stepdir;
  try {
    stepdir.create(stepdname);
  } catch (...) {
    ; // 
  }
}

static inline double MIN(double a, double b) { if(a<b) return a; return b; }
static inline double MAX(double a, double b) { if(a>b) return a; return b; }

static void _binval(vector<int> & bins, double minval, double maxval, double val)
{
  if(val<minval || val>maxval) return;
  int nbins = bins.size();
  int ibin = (int)(nbins*(val-minval)/(maxval-minval+1.e-5));
  bins[ibin]++;
}

template <class ElemT>
void 
HistogramDumper::Step::
_binvals(LevelP level, Uintah::TypeDescription::Type type_, 
         const string & fieldname, int imat, int idiag,
         double & minval, double & maxval, vector<int> & counts, string & ext)
{
  ScalarDiagGen<ElemT> diaggen;
  
  ext = diaggen.name(idiag);
  if(ext!="") ext = "_"+ext;
  
  minval =  FLT_MAX;
  maxval = -FLT_MAX;
  
  IntVector minind, maxind;
  level->findNodeIndexRange(minind, maxind);
  
  for(int ipass=0;ipass<2;ipass++) {
    for(Level::const_patchIterator iter = level->patchesBegin();
        iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      ConsecutiveRangeSet matls = da->queryMaterials(fieldname, patch, time);
      
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
          matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
        if (matl!=imat) break;
        
        switch(type_){
        case Uintah::TypeDescription::NCVariable:
          {
            NCVariable<ElemT>  value;
            NCVariable<double> Mvalue;
            
            da->query(value, fieldname, matl, patch, time);
            da->query(Mvalue, "g.mass", matl, patch, time);
            
            for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
              if(_outside(*iter, minind, maxind)) continue;
              
              double mcell = Mvalue[*iter];
              if(mcell<1.e-10) continue;
              
              const double val = diaggen.gen(value[*iter], idiag);
              if(ipass==0) {
                minval = MIN(minval, val);
                maxval = MAX(maxval, val);
              } else {
                _binval(counts, minval, maxval, val);
              }
            }
          } break;
          
        case Uintah::TypeDescription::CCVariable:
          {
            CCVariable<ElemT>  value;
            da->query(value, fieldname, matl, patch, time);
            
            for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
              if(_outside(*iter, minind, maxind)) continue;
              double val = diaggen.gen(value[*iter], idiag);
              if(ipass==0) {
                minval = MIN(minval, val);
                maxval = MAX(maxval, val);
              } else {
                _binval(counts, minval, maxval, val);
              }
            }
          } break;
          
        case Uintah::TypeDescription::ParticleVariable:
          {
            ParticleVariable<ElemT> value;
            da->query(value, fieldname, matl, patch, time);
            ParticleSubset* pset = value.getParticleSubset();
            for(ParticleSubset::iterator iter = pset->begin();
                iter != pset->end(); iter++) {
              double mag = diaggen.gen(value[*iter], idiag);
              if(ipass==0) {
                minval = MIN(minval, mag);
                maxval = MAX(maxval, mag);
              } else {
                _binval(counts, minval, maxval, mag);
              }
            }
          } break;
          
        default:
          break;
        } // type switch
        
      } // materials
    } // patches

    if(minval>maxval) break;
    if(ipass==0) {
      for(size_t ibin=0;ibin<counts.size();ibin++) counts[ibin] = 0;
    }
    
  } // pass
  
}

void
HistogramDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
{
  cout << "   " << fieldname << endl;
  
  GridP grid = da->queryGrid(time);
  
  int nmats = 0;
  for(int l=0;l<=0;l++) {
    LevelP level = grid->getLevel(l);
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      ConsecutiveRangeSet matls= da->queryMaterials(fieldname, patch, time);
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
	int matl = *matlIter;
	if(matl>=nmats) nmats = matl+1;
      }
    }
  }

  // dump one material at a time
  for(int imat=0;imat<nmats;imat++) {
    
    int ndiags = 0;
    string ext = "";
    switch(td->getSubType()->getType()){
    case Uintah::TypeDescription::float_type:  ndiags = ScalarDiagGen<float  >::number; break;
    case Uintah::TypeDescription::double_type: ndiags = ScalarDiagGen<double >::number; break;
    case Uintah::TypeDescription::Point:       ndiags = ScalarDiagGen<Point  >::number; break;
    case Uintah::TypeDescription::Vector:      ndiags = ScalarDiagGen<Vector >::number; break;
    case Uintah::TypeDescription::Matrix3:     ndiags = ScalarDiagGen<Matrix3>::number; break;
    default: ndiags = 0;
    }
    
    for(int idiag=0;idiag<ndiags;idiag++) {
      double minval, maxval;
      vector<int> bins(nbins);
      
      // only support level 0 for now
      switch(td->getSubType()->getType()){
      case Uintah::TypeDescription::float_type:  
        _binvals<float>(grid->getLevel(0), td->getType(), fieldname, imat, idiag, minval, maxval, bins, ext);
        break;
      case Uintah::TypeDescription::double_type:
        _binvals<double>(grid->getLevel(0), td->getType(), fieldname, imat, idiag, minval, maxval, bins, ext);
        break;
      case Uintah::TypeDescription::Point:
        _binvals<Point>(grid->getLevel(0), td->getType(), fieldname, imat, idiag, minval, maxval, bins, ext);
        break;
      case Uintah::TypeDescription::Vector:
        _binvals<Vector>(grid->getLevel(0), td->getType(), fieldname, imat, idiag, minval, maxval, bins, ext);
        break;
      case Uintah::TypeDescription::Matrix3:
        _binvals<Matrix3>(grid->getLevel(0), td->getType(), fieldname, imat, idiag, minval, maxval, bins, ext);
        break;
      default:
        cout << "no match for sub type " << endl;
        ;
      }
      
      // if no points found, dont write empty histogram
      if(minval>maxval) continue;
      
      string matname = HistogramDumper::mat_string(imat);
      string fname   = HistogramDumper::makeFileName(datadir, time_string(time), fieldname+ext, matname, "hist");
      cout << "     " << fname << endl;
      cout << "       mat " << imat << ", "
           << "range = " << minval << "," << maxval
           << endl;
      
      ofstream os(fname.c_str());
      os << "# time = " << time << ", field = " 
         << fieldname << ", mat " << matname << endl;
      os << "# min = " << minval << endl;
      os << "# max = " << maxval << endl;
      
      for(int ibin=0;ibin<nbins;ibin++) {
        double xmid = minval+(ibin+0.5)*(maxval-minval)/nbins;
        os << xmid << " " << bins[ibin] << endl;
      }
    } // idiag
  } // imat
}

// -----------------------------------------------------------------------------

// store tuple of (variable, it's type)
typedef pair<string, const Uintah::TypeDescription*> typed_varname;		  

static 
void usage(const string& badarg, const string& progname)
{
  if(badarg != "")
    cerr << "Error parsing argument: " << badarg << endl;
  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -bin                dump in binary format\n";
  cerr << "  -onemesh            only store single mesh (ensight only)\n";
  cerr << "  -onedim             generate one dim plots (dx/text only)\n";
  cerr << "  -tseries            generate single time series file (text only)\n";
  cerr << "  -withpart           include particles (ensight only)\n";
  cerr << "  -nbins              number of particle bins\n";
  cerr << "  -timesteplow  [int] (only outputs timestep from int)\n";
  cerr << "  -timestephigh [int] (only outputs timesteps upto int)\n";
  cerr << "  -timestepinc  [int] (only outputs every int timesteps)\n";
  cerr << "  -basename     [bnm] basename to write to\n";
  cerr << "  -format       [fmt] output format, one of (text,ensight,dx,histogram)\n";
  cerr << "  -field        [fld] field to dump\n";
  exit(1);
}

int
main(int argc, char** argv)
{
  /*
   * Parse arguments
   */
  bool do_verbose = false;
  int time_step_lower = 0;
  int time_step_upper = INT_MAX;
  int time_step_inc   = 1;
  string datadir = "";
  string fieldnames = "";
  string fmt = "text";
  bool binary = false;
  bool onemesh = false;
  bool onedim  = false;
  bool tseries  = false;
  bool withpart = false;
  int nbins(256);
  
  for(int i=1;i<argc;i++){
    string s=argv[i];
    
    if (s == "-verbose") {
      do_verbose = true;
    } else if (s == "-timesteplow") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-timestephigh") {
      time_step_upper = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-timestepinc") {
      time_step_inc = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-field") {
      fieldnames = argv[++i];
    } else if (s == "-basename") {
      datadir = argv[++i];
    } else if (s == "-format") {
      fmt = argv[++i];
    } else if (s == "-bin") {
      binary = true;
    } else if (s == "-onemesh") {
      onemesh = true;
    } else if (s == "-onedim") {
      onedim = true;
    } else if (s == "-nbins") {
      nbins = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-tseries") {
      tseries = true;
    } else if (s == "-withpart") {
      withpart = true;
    } else if( (s == "-help") || (s == "-h") ) {
      usage( "", argv[0] );
    }
  }
  string filebase = argv[argc-1];
  if(filebase == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }
  
  if(!datadir.size())
    {
      datadir = filebase.substr(0, filebase.find('.'));
    }
  
  try {
    DataArchive* da = scinew DataArchive(filebase);
    
    // load list of possible variables from the data archive
    vector<string> allvars;
    vector<const Uintah::TypeDescription*> alltypes;
    da->queryVariables(allvars, alltypes);
    ASSERTEQ(allvars.size(), alltypes.size());
    cout << "There are " << allvars.size() << " variables:\n";
    
    // load list of possible indices and times
    vector<int>    index;
    vector<double> times;
    da->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    cout << "There are " << index.size() << " timesteps:\n";
    
    if(time_step_lower<0)                  time_step_lower = 0;
    if(time_step_upper>=(int)index.size()) time_step_upper = index.size()-1;
    if(time_step_inc<=0)                   time_step_inc   = 1;
    
    // build list of variables to dump
    list<typed_varname> dumpvars;
    int nvars = allvars.size();
    if(fieldnames.size()) {
      vector<string> requested_fields = split(fieldnames,',',false);
      int nreq = requested_fields.size();
      for(int j=0;j<nreq;j++) {
        bool foundvar = false;
        string fname = requested_fields[j];
        for(int i=0;i<nvars;i++) {
          if(allvars[i]==fname) {
            dumpvars.push_back( typed_varname(fname, alltypes[i]) );
            foundvar = true;
            break;
          }
        }
        if(!foundvar) {
          cerr << "Failed to find variable '" << fname << "'" << endl;
          cerr << "valid values are: " << endl;
          for(vector<string>::const_iterator vit(allvars.begin());vit!=allvars.end();vit++)
            cerr << "   " << *vit << endl;
          cerr << endl;
          abort();
        }
        
      }
    } else {
      for(int i=0;i<nvars;i++) {
        dumpvars.push_back( typed_varname(allvars[i], alltypes[i]) );
      }
    }
    
    if(!dumpvars.size()) {
      cerr << "Failed to find variable from '" << fieldnames << "'" << endl;
      cerr << "valid values are: " << endl;
      for(vector<string>::const_iterator vit(allvars.begin());vit!=allvars.end();vit++)
        cerr << "   " << *vit << endl;
      cerr << endl;
      abort();
    }
    
    Dumper * dumper = 0;
    if(fmt=="text") {
      dumper = new TextDumper(da, datadir, onedim, tseries);
    } else if(fmt=="ensight") {
      dumper = new EnsightDumper(da, datadir, binary, onemesh, withpart);
    } else if(fmt=="histogram") {
      dumper = new HistogramDumper(da, datadir, nbins);
    } else if(fmt=="dx") {
      dumper = new DxDumper(da, datadir, binary, onedim);
    }
    
    if(dumper) {
      
      for(list<typed_varname>::const_iterator vit(dumpvars.begin());vit!=dumpvars.end();vit++) {
	const string fieldname = vit->first;
	const Uintah::TypeDescription* td      = vit->second;
	dumper->addField(fieldname, td);
      }
      
      // loop over the times
      for(int i=time_step_lower;i<=time_step_upper;i+=time_step_inc) {
	cout << index[i] << ": " << times[i] << endl;
      
	Dumper::Step * step_dumper = dumper->addStep(index[i], times[i]);
        
	step_dumper->storeGrid();
        
	for(list<typed_varname>::const_iterator vit(dumpvars.begin());vit!=dumpvars.end();vit++) {
	  const string fieldname = vit->first;
	  const Uintah::TypeDescription* td      = vit->second;
	  
	  step_dumper->storeField(fieldname, td);
	}
	
	cout << endl;
	
	dumper->finishStep(step_dumper);
	delete step_dumper;
      }
      
      delete dumper;
      
    } else {
      cerr << "Failed to find match to format '" + fmt + "'" << endl;
    }
    
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
}

