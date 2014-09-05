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

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
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
#include <errno.h>
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

// split string at sep into a list
static
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
  const string name(int /*idiag*/) const { return ""; }
  double gen(double v, int /*idiag*/) const { return v; }
};

template <>
class ScalarDiagGen<double> {
public:
  static const int number = 1;
  const string name(int /*idiag*/) const { return ""; }
  double gen(double v, int /*idiag*/) const { return v; }
};

template <>
class ScalarDiagGen<Vector> {
public: 
  static const int number = 1;
  const string name(int /*idiag*/) const { return ""; }
  double gen(const Vector & v, int /*idiag*/) const { return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
};

template <>
class ScalarDiagGen<Point> {
public:
  static const int number = 1;
  const string name(int /*idiag*/) const { return "mag"; }
  double gen(const Point & v, int /*idiag*/) const { return sqrt(v(0)*v(0)+v(1)*v(1)+v(2)*v(2)); }
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
    default:
      return 0.;
    }
  }
};

// -----------------------------------------------------------------------------
// Base dumper class 
// one dumper for each output format
class Dumper {

protected:
  Dumper(DataArchive * da, string basedir);
  
  string createDirectory();
  
public:
  virtual ~Dumper();
  
  // extension for the output directory
  virtual string directoryExt() const = 0;
  
  // add a new field
  virtual void addField(string fieldname, const Uintah::TypeDescription * type) = 0;
  
  // dump a single step
  class Step {
  public:
    Step(string tsdir, int index, double time);
    
    virtual ~Step();
    
    virtual string infostr() const = 0;
    virtual void   storeGrid() = 0;
    virtual void   storeField(string fieldname, const Uintah::TypeDescription * type) = 0;
    
  public: // hmmm ....
    string tsdir_;
    int    index_;
    double time_;
    
  protected:
    string fileName(string variable_name, 
                    int    materialNum,
                    string extension="") const;
  };
  virtual Step * addStep(int index, double time) = 0;
  virtual void finishStep(Step *) = 0;

  DataArchive * archive() const { return this->da_; }
  
private:
  static string mat_string(int mval);
  static string time_string(double tval);

protected:
  string dirName(double tval) const;
  
private:
  DataArchive* da_;
  string       basedir_;
};

Dumper::Dumper(DataArchive * da, string basedir)
  : da_(da), basedir_(basedir)
{
}

string
Dumper::createDirectory()
{
  // Create a directory if it's not already there.
  // The exception occurs when the directory is already there
  // and the Dir.create fails.  This exception is ignored. 
  string dirname = this->basedir_+"_"+this->directoryExt();
  Dir dumpdir;
  try {
    dumpdir.create(dirname);
  } catch (ErrnoException & e) {
    // only allow exists as a reason for failure
    if(e.getErrno()!= EEXIST)
      throw;
  }
  return dirname;
}

string
Dumper::time_string(double tval)
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
Dumper::mat_string(int mval)
{
  ostringstream b;
  b << setw(4) << setfill('0') << mval;
  return b.str();
}

string 
Dumper::dirName(double time_val) const
{
  return this->basedir_+"_"+this->directoryExt()+"/"+time_string(time_val);
}

Dumper::~Dumper() {}

Dumper::Step::Step(string tsdir, int index, double time)
: tsdir_(tsdir), index_(index), time_(time) 
{
  // create directory to store output files 
  Dir stepdir;
  try {
    stepdir.create(tsdir_);
  } catch (ErrnoException & e) {
    // only allow exists as a reason for failure
    if(e.getErrno()!= EEXIST)
      throw;
  }
}

string
Dumper::Step::fileName(string variable_name, int imat, string ext)  const
{
  string datafile = tsdir_+"/";
  
  datafile += string("VAR_") + variable_name + string(".");
  
  if (imat>=0)
    datafile+= string("MT_") + mat_string(imat) + string(".");
  
  if (ext!="")
    datafile+=ext;
  
  return datafile;
}

Dumper::Step::~Step() {}

// -----------------------------------------------------------------------------

#define ONEDIM_DIM 2

class TextDumper : public Dumper 
{
public:
  TextDumper(DataArchive * da, string basedir, bool onedim=false, bool tseries=false);
  
  string directoryExt() const { return "text"; }
  void addField(string fieldname, const Uintah::TypeDescription * /*type*/);
  
  class Step : public Dumper::Step {
  public:
    Step(DataArchive * da, string basedir, int index, double time, bool onedim, bool tseries);
    
    string infostr() const { return tsdir_; }
    void   storeGrid () {}
    void   storeField(string fieldname, const Uintah::TypeDescription * type);
    
  private:
    DataArchive *       da_;
    string              outdir_;
    bool                onedim_;
    bool                tseries_;
  };
  
  //
  Step * addStep(int index, double time);
  void   finishStep(Dumper::Step * s);
  
private:
  ofstream idxos_;
  bool     onedim_, tseries_;
  FILE*    filelist_;
};

TextDumper::TextDumper(DataArchive * da, string basedir, bool onedim, bool tseries)
  : Dumper(da, basedir), onedim_(onedim), tseries_(tseries)
{
  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  
  // set up a file that contains a list of all the files
  string dirname = this->createDirectory();
  string filelistname = dirname+"/timelist";
  
  filelist_ = fopen(filelistname.c_str(),"w");
  if (!filelist_) {
    cerr << "Can't open output file " << filelistname << endl;
    abort();
  }
}

TextDumper::Step * 
TextDumper::addStep(int index, double time)
{
  return scinew Step(this->archive(), this->dirName(time), 
                     index, time, onedim_, tseries_);
}  

void
TextDumper::addField(string fieldname, const Uintah::TypeDescription * type)
{
}

void
TextDumper::finishStep(Dumper::Step * s)
{
  fprintf(filelist_, "%10d %16.8g  %20s\n", s->index_, s->time_, s->infostr().c_str());
}

TextDumper::Step::Step(DataArchive * da, string tsdir,
                       int index, double time,  bool onedim, bool tseries)
  : 
  Dumper::Step(tsdir, index, time),
  da_(da), onedim_(onedim), tseries_(tseries)
{
  GridP grid = da_->queryGrid(time);
}

static
bool
outside(IntVector p, IntVector mn, IntVector mx)
{
  return  ( p[0]<mn[0] || p[0]>=mx[0] ||
	    p[1]<mn[1] || p[1]>=mx[1] ||
	    p[2]<mn[2] || p[2]>=mx[2] );
}

void
TextDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
{
  GridP grid = da_->queryGrid(time_);
  
  cout << "   " << fieldname << endl;
  const Uintah::TypeDescription* subtype = td->getSubType();
  
  int nmats = 0;
  // count the materials
  for(int l=0;l<=0;l++) {
    LevelP level = grid->getLevel(l);
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      ConsecutiveRangeSet matls= da_->queryMaterials(fieldname, patch, time_);
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
	int matl = *matlIter;
	if(matl>=nmats) nmats = matl+1;
      }
    }
  }
  
  // only support level 0 for now
  for(int l=0;l<=0;l++) {
    LevelP level = grid->getLevel(l);
    
    IntVector minind, maxind;
    level->findNodeIndexRange(minind, maxind);
    if(onedim_) {
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
      
      ConsecutiveRangeSet matls = da_->queryMaterials(fieldname, patch, time_);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
	  matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
        
        string fname = fileName(fieldname, matl, "txt");
        ofstream outfile;
        if(tseries_ || index_==1) {
          outfile.open( fname.c_str() );
          outfile << "# time = " << time_ << ", field = " << fieldname << ", mat " << matl << endl;
        } else {
          outfile.open( fname.c_str(), ios::app);
          outfile << endl;
        }
	bool no_match = false;
	      
	ParticleVariable<Point> partposns;
	if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	  da_->query(partposns, "p.x", matl, patch, time_);
	}
	
	switch(subtype->getType()) {
	case Uintah::TypeDescription::float_type:
	  {
	    switch(td->getType()){
	    case Uintah::TypeDescription::NCVariable:
	      {
		NCVariable<float> value;
		da_->query(value, fieldname, matl, patch, time_);
		
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter] << " "
                          << endl;
		}
	      } break;
	    case Uintah::TypeDescription::CCVariable:
	      {
		CCVariable<float> value;
		da_->query(value, fieldname, matl, patch, time_);
		
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter] << " "
                          << endl;
		}
	      } break;
	    case Uintah::TypeDescription::ParticleVariable:
	      {
		ParticleVariable<float> value;
		da_->query(value, fieldname, matl, patch, time_);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << (value[*iter]) << " " 
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
		da_->query(value, fieldname, matl, patch, time_);
			
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter] << " "
                          << endl;
		}
	      } break;
	    case Uintah::TypeDescription::CCVariable:
	      {
		CCVariable<double> value;
		da_->query(value, fieldname, matl, patch, time_);
			
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter] << " "
                          << endl;
		}
	      } break;
	    case Uintah::TypeDescription::ParticleVariable:
	      {
		ParticleVariable<double> value;
		da_->query(value, fieldname, matl, patch, time_);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << (value[*iter]) << " " 
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
		da_->query(value, fieldname, matl, patch, time_);
		    
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter](0) << " "
                          << value[*iter](1) << " "
                          << value[*iter](2) << " "
                          << endl;
		}
	      } break;
	    case Uintah::TypeDescription::CCVariable:
	      {
		CCVariable<Point> value;
		da_->query(value, fieldname, matl, patch, time_);
		    
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter](0) << " "
                          << value[*iter](1) << " "
                          << value[*iter](2) << " "
                          << endl;
		}
	      } break;
	    case Uintah::TypeDescription::ParticleVariable:
	      {
		ParticleVariable<Point> value;
		da_->query(value, fieldname, matl, patch, time_);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter](0) << " "
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
		da_->query(value, fieldname, matl, patch, time_);
		      
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter][0] << " "
                          << value[*iter][1] << " "
                          << value[*iter][2] << " "
                          << endl;
		}
	      } break;
	    case Uintah::TypeDescription::CCVariable:
	      {
		CCVariable<Vector> value;
		da_->query(value, fieldname, matl, patch, time_);
		      
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter][0] << " "
                          << value[*iter][1] << " "
                          << value[*iter][2] << " "
                          << endl;
		}
	      } break;
	    case Uintah::TypeDescription::ParticleVariable:
	      {
		ParticleVariable<Vector> value;
		da_->query(value, fieldname, matl, patch, time_);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter][0] << " "
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
		da_->query(value, fieldname, matl, patch, time_);
		    
		for(NodeIterator iter = patch->getNodeIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter](0,0) << " "
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
		da_->query(value, fieldname, matl, patch, time_);
		    
		for(CellIterator iter = patch->getCellIterator();
		    !iter.done(); iter++){
		  if(outside(*iter, minind, maxind)) continue;
		  Point xpt = patch->nodePosition(*iter);
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter](0,0) << " "
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
		da_->query(value, fieldname, matl, patch, time_);
		ParticleSubset* pset = value.getParticleSubset();
		for(ParticleSubset::iterator iter = pset->begin();
		    iter != pset->end(); iter++) {
		  Point xpt = partposns[*iter];
                  if(tseries_) outfile << time_ << " ";
                  outfile << xpt(0) << " " 
                          << xpt(1) << " " 
                          << xpt(2) << " ";
		  outfile << value[*iter](0,0) << " "
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
          
	case Uintah::TypeDescription::bool_type:
	case Uintah::TypeDescription::short_int_type:
	case Uintah::TypeDescription::long_type:
	case Uintah::TypeDescription::long64_type:
          {
            ; // silently ignore integer variables
          } break;
	default:
	  no_match = true;
	} // type switch
        
	if (no_match)
	  cerr << "WARNING: Unexpected type for " << td->getName() << " of " << subtype->getName() << endl;
        
      } // materials
    } // patches
  } // levels 
}    

// -----------------------------------------------------------------------------

class EnsightDumper : public Dumper 
{
private:
  // dump field as text or binary
  struct FldDumper { 
    // ensight is very fussy about the format of the text, so it's nice
    // to have this all in one place
    FldDumper(bool bin) : bin_(bin), os_(0) {}
    
    void setstrm(ostream * os) { os_ = os; }
    void unsetstrm() { os_ = 0; }
    
    void textfld(string v, int width=80) {
      *os_ << setw(width) << setiosflags(ios::left) << v;
    }
    
    void textfld(string v, int width, int binwidth) {
      if(bin_)
        *os_ << setw(binwidth)  << setiosflags(ios::left) << v;
      else
        *os_ << setw(width)  << setiosflags(ios::left) << v;
    }
    
    void textfld(int v, int width=10) {
      *os_ << setiosflags(ios::left) << setw(width) << v;
    }
    
    void textfld(float v) { 
      char b[13];
      snprintf(b, 13, "%12.5e", v);
      *os_ << b;
    }
    
    void textfld(double v) { 
      char b[13];
      snprintf(b, 13, "%12.5e", v);
      *os_ << b;
    }
    
    void numfld(int v, int wid=10) { 
      if(!bin_) {
        textfld(v, wid);
      } else {
        os_->write((char*)&v, sizeof(int));
      }
    }
    
    void endl() { 
      if(!bin_) {
        *os_ << std::endl;
      }
    }
    
    void numfld(float v) { 
      v = (fabs(v)<FLT_MIN)?0.:v;
      if(!bin_) {
        textfld(v);
      } else {
        os_->write((char*)&v, sizeof(float));
      }
    }
    
    void numfld(double v) { 
      numfld((float)v);
    }
    
    bool      bin_;
    ostream * os_;
  };

  struct Data {
    Data(DataArchive * da, FldDumper * dumper, bool onemesh, bool withpart)
      : da_(da), dumper_(dumper), onemesh_(onemesh), withpart_(withpart) {}
    
    DataArchive * da_;
    string        dir_;
    FldDumper   * dumper_;
    bool          onemesh_, withpart_;
  };
  
public:
  EnsightDumper(DataArchive* da, string basedir, bool binmode=false, bool onemesh=false, bool withpart=false);
  ~EnsightDumper();
  
  string directoryExt() const { return "ensight"; }
  void addField(string fieldname, const Uintah::TypeDescription * type);
  
  class Step : public Dumper::Step {
    friend class EnsightDumper;
  private:
    Step(Data * data, string tsdir, int index, double time, int fileindex);
    
  public:
    string infostr() const { return stepdesc_; }
    void   storeGrid ();
    void   storeField(string filename, const Uintah::TypeDescription * type);
    
  private:
    void storePartField(string filename, const Uintah::TypeDescription * type);
    void storeGridField(string filename, const Uintah::TypeDescription * type);
    
  private:
    int    fileindex_;
    string stepdname_;
    string stepdesc_;
    
  private:
    Data    * data_;
    bool      needmesh_;
    IntVector vshape_, minind_;
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
  int           nsteps_;
  ofstream      casestrm_;
  int           tscol_;
  ostringstream tsstrm_;
  FldDumper     flddumper_;
  Data          data_;
};

EnsightDumper::EnsightDumper(DataArchive* da_, string basedir_, bool bin_, bool onemesh_, bool withpart_)
  : 
  Dumper(da_, basedir_), nsteps_(0), flddumper_(bin_), 
  data_(da_,&flddumper_,onemesh_,withpart_)
{
  data_.dir_ = this->createDirectory();
  
  // set up the file that contains a list of all the files
  string casefilename = data_.dir_ + string("/") + string("ensight.case");
  casestrm_.open(casefilename.c_str());
  if (!casestrm_) {
    cerr << "Can't open output file " << casefilename << endl;
    abort();
  }
  cout << "     " << casefilename << endl;
  
  // header
  casestrm_ << "FORMAT" << endl;
  casestrm_ << "type: ensight gold" << endl;
  casestrm_ << endl;
  casestrm_ << "GEOMETRY" << endl;
  if(data_.onemesh_)
    casestrm_ << "model:       gridgeo" << endl;
  else
    casestrm_ << "model:   1   gridgeo****" << endl;
  casestrm_ << endl;
  casestrm_ << "VARIABLE" << endl;
  
  // time step data
  tscol_  = 0;
  tsstrm_ << "time values: ";;
}

EnsightDumper::~EnsightDumper()
{
  casestrm_ << endl;
  casestrm_ << "TIME" << endl;
  casestrm_ << "time set: 1 " << endl;
  casestrm_ << "number of steps: " << nsteps_ << endl;
  casestrm_ << "filename start number: 0" << endl;
  casestrm_ << "filename increment: 1" << endl;
  casestrm_ << tsstrm_.str() << endl;
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
	casestrm_ << "scalar per " << typestr << ": 1 " 
                 << fieldname << tensor_diagname(idiag) 
                 << " " << nodots(fieldname) << tensor_diagname(idiag) << "****" << endl;
    }
  else
    {
      casestrm_ << subtypestr << " per " << typestr << ": 1 " << fieldname << " " << nodots(fieldname) << "****" << endl;
    }
}

EnsightDumper::Step * 
EnsightDumper::addStep(int index, double time)
{
  Step * res = scinew Step(&data_, this->dirName(time), index, time, nsteps_);
  nsteps_++;
  return res;
}

void
EnsightDumper::finishStep(Dumper::Step * step)
{
  tsstrm_ << step->time_ << " ";
  if(++tscol_==10) {
    tsstrm_ << endl << "  ";
    tscol_ = 0;
  }
}

EnsightDumper::Step::Step(Data * data, string tsdir, int index, double time, int fileindex)
  :
  Dumper::Step(tsdir, index, time),
  fileindex_(fileindex),
  data_(data),
  needmesh_(!data->onemesh_||(fileindex==0))
{
}

void
EnsightDumper::Step::storeGrid()
{
  GridP grid = data_->da_->queryGrid(time_);
  FldDumper * fd = data_->dumper_;
  
  // only support level 0 for now
  int lnum = 0;
  LevelP level = grid->getLevel(lnum);
  
  // store to basename/grid.geo****
  char goutname[1024];
  char poutname[1024];
  if(data_->onemesh_) {
    snprintf(goutname, 1024, "%s/gridgeo", data_->dir_.c_str());
  } else {
    snprintf(goutname, 1024, "%s/gridgeo%04d", data_->dir_.c_str(), fileindex_);
  }
  snprintf(poutname, 1024, "%s/partgeo%04d", data_->dir_.c_str(), fileindex_);
  
  if(needmesh_) {
    
    // find ranges
    // dont bother dumping ghost stuff to ensight
    IntVector minind, maxind;
    level->findNodeIndexRange(minind, maxind);
    minind_ = minind;
    vshape_ = (maxind-minind);
  
    cout << "  " << goutname << endl;
    cout << "   minind = " << minind_ << endl;
    cout << "   maxind = " << maxind << endl;
    cout << "   vshape = " << vshape_ << endl;
    cout << endl;
  
    ofstream gstrm(goutname);
    fd->setstrm(&gstrm);
  
    if(fd->bin_) fd->textfld("C Binary",79,80);
    fd->textfld("grid description",79,80) ; fd->endl();
    fd->textfld("grid description",79,80) ; fd->endl();
    fd->textfld("node id off",     79,80) ; fd->endl();
    fd->textfld("element id off",  79,80) ; fd->endl();
  
    fd->textfld("part",            79,80) ; fd->endl();
    fd->numfld(1); fd->endl();
    fd->textfld("3d regular block",79,80); fd->endl();
    fd->textfld("block",79,80); fd->endl();
    fd->numfld(vshape_(0),10);
    fd->numfld(vshape_(1),10);
    fd->numfld(vshape_(2),10);
    fd->endl();
    
    for(int id=0;id<3;id++) {
      for(int k=minind_[2];k<maxind[2];k++) {
        for(int j=minind_[1];j<maxind[1];j++) {
          for(int i=minind_[0];i<maxind[0];i++) {
            fd->numfld(level->getNodePosition(IntVector(i,j,k))(id)) ; fd->endl();
          }
        }
      }
    }
    
    fd->unsetstrm();
  }
  
  if(data_->withpart_) {
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
    
      ConsecutiveRangeSet matls = data_->da_->queryMaterials("p.x", patch, time_);
    
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
          matlIter != matls.end(); matlIter++) {
        ParticleVariable<Matrix3> value;
        
        const int matl = *matlIter;
      
        ParticleVariable<Point> partposns;
        data_->da_->query(partposns, "p.x", matl, patch, time_);
      
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
    storePartField(fieldname, td);
  else 
    storeGridField(fieldname, td);
}

void
EnsightDumper::Step::storeGridField(string fieldname, const Uintah::TypeDescription * td)
{
  GridP grid = data_->da_->queryGrid(time_);
  FldDumper * fd = data_->dumper_;
  
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
		 data_->dir_.c_str(), nodots(fieldname).c_str(), 
		 tensor_diagname(idiag).c_str(), fileindex_);
      else
	snprintf(outname, 1024, "%s/%s%04d", data_->dir_.c_str(), nodots(fieldname).c_str(), fileindex_);
      
      int icol(0);
      
      cout << "  " << outname;
      cout.flush();
      
      ofstream vstrm(outname);
      fd->setstrm(&vstrm);
      
      ostringstream descb;
      descb << "data field " << nodots(fieldname) << " at " << time_;
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
        
	Uintah::Array3<double> vals(vshape_[0],vshape_[1],vshape_[2]);
	for(Uintah::Array3<double>::iterator vit(vals.begin());vit!=vals.end();vit++)
	  {
	    *vit = 0.;
	  }
        
        cout << ", patch ";
        for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
	  const Patch* patch = *iter;
          cout << patch->getID() << " ";
          cout.flush();
          
          IntVector ilow, ihigh;
          patch->computeVariableExtents(subtype->getType(), IntVector(0,0,0), Ghost::None, 0, ilow, ihigh);
          
          ConsecutiveRangeSet matls = data_->da_->queryMaterials(fieldname, patch, time_);
          
          // loop over materials
          for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
            int i,j,k;
            const int matl = *matlIter;
            
            switch(subtype->getType()) {
            case Uintah::TypeDescription::float_type:
              {
                NCVariable<float> value;
                data_->da_->query(value, fieldname, matl, patch, time_);
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  vals[ijk-minind_] += value[ijk];
                }
	      } break;
	    case Uintah::TypeDescription::double_type:
	      {
                NCVariable<double> value;
                data_->da_->query(value, fieldname, matl, patch, time_);
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  vals[ijk-minind_] += value[ijk];
                }
	      } break;
	    case Uintah::TypeDescription::Point:
	      {
                NCVariable<Point> value;
                data_->da_->query(value, fieldname, matl, patch, time_);
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  vals[ijk-minind_] += value[ijk](icomp);
                }
	      } break;
	    case Uintah::TypeDescription::Vector:
	      {
                NCVariable<Vector> value;
                data_->da_->query(value, fieldname, matl, patch, time_);
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  vals[ijk-minind_] += value[ijk][icomp];
                }
	      } break;
	    case Uintah::TypeDescription::Matrix3:
	      {
		NCVariable<Matrix3> value;
		data_->da_->query(value, fieldname, matl, patch, time_);
                
                for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
                  IntVector ijk(i,j,k);
                  const Matrix3 t = value[ijk];
                  switch(idiag) {
                  case 0: // mag
                    vals[ijk-minind_] = t.Norm();
                    break;
                  case 1: // equiv
                    {
                      const double  p  = t.Trace()/3.;
                      const Matrix3 ti = t - Matrix3(p,0,0, 0,p,0, 0,0,p);
                      vals[ijk-minind_] = ti.Norm();
                    }
                    break;
                  case 2: // trace
                    vals[ijk-minind_] = t.Trace();
                    break;
                  case 3: // max absolute element
                    vals[ijk-minind_] = t.MaxAbsElem();
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
	for(int k=0;k<vshape_[2];k++) 
	  for(int j=0;j<vshape_[1];j++) 
	    for(int i=0;i<vshape_[0];i++) {
	      fd->numfld(vals[IntVector(i,j,k)]);
	      if(++icol==1) 
		{
		  fd->endl();
		  icol = 0;
		}
	      if(icol!=0) fd->endl();
	    }
        cout << endl;
        
      } // components
      
      fd->unsetstrm();
      
    } // diags  
}

void
EnsightDumper::Step::storePartField(string /*fieldname*/, const Uintah::TypeDescription * /*td*/)
{
  cout << "no particles for you - i spit in your general direction" << endl;
}

// -----------------------------------------------------------------------------

class DXDumper : public Dumper 
{
public:
  DXDumper(DataArchive* da, string basedir, bool binmode=false, bool onedim=false);
  ~DXDumper();
  
  string directoryExt() const { return "dx"; }
  void addField(string fieldname, const Uintah::TypeDescription * type);
  
  struct FldWriter { 
    FldWriter(string outdir, string fieldname);
    ~FldWriter();
    
    int      dxobj_;
    ofstream strm_;
    list< pair<float,int> > timesteps_;
  };
  
  class Step : public Dumper::Step {
  public:
    Step(DataArchive * da, string outdir, int index, double time, int fileindex, 
	 const map<string,FldWriter*> & fldwriters, bool bin, bool onedim);
    
    void storeGrid ();
    void storeField(string filename, const Uintah::TypeDescription * type);
    
    string infostr() const { return dxstrm_.str()+"\n"+fldstrm_.str()+"\n"; }
    
  private:
    DataArchive*  da_;
    int           fileindex_;
    string        outdir_;
    ostringstream dxstrm_, fldstrm_;
    const map<string,FldWriter*> & fldwriters_;
    bool          bin_, onedim_;
  };
  friend class Step;
  
  //
  Step * addStep(int index, double time);
  void finishStep(Dumper::Step * step);
  
private:  
  int           nsteps_;
  int           dxobj_;
  ostringstream timestrm_;
  ofstream      dxstrm_;
  map<string,FldWriter*> fldwriters_;
  bool          bin_;
  bool          onedim_;
  string        dirname_;
};

DXDumper::DXDumper(DataArchive* da, string basedir, bool bin, bool onedim)
  : Dumper(da, basedir), nsteps_(0), dxobj_(0), bin_(bin), onedim_(onedim)
{
  // set up the file that contains a list of all the files
  this->dirname_ = this->createDirectory();
  string indexfilename = dirname_ + string("/") + string("index.dx");
  dxstrm_.open(indexfilename.c_str());
  if (!dxstrm_) {
    cerr << "Can't open output file " << indexfilename << endl;
    abort();
  }
  cout << "     " << indexfilename << endl;
}

DXDumper::~DXDumper()
{
  dxstrm_ << "object \"udadata\" class series" << endl;
  dxstrm_ << timestrm_.str();
  dxstrm_ << endl;
  dxstrm_ << "default \"udadata\"" << endl;
  dxstrm_ << "end" << endl;
  
  for(map<string,FldWriter*>::iterator fit(fldwriters_.begin());fit!=fldwriters_.end();fit++) {
    delete fit->second;
  }
}

void
DXDumper::addField(string fieldname, const Uintah::TypeDescription * /*td*/)
{
  if(fieldname=="p.particleID") return;
  fldwriters_[fieldname] = scinew FldWriter(this->dirname_, fieldname);
}

DXDumper::Step * 
DXDumper::addStep(int index, double time)
{
  DXDumper::Step * r = scinew Step(archive(), dirname_, index, time, nsteps_++, fldwriters_, bin_, onedim_);
  return r;
}
  
void
DXDumper::finishStep(Dumper::Step * step)
{
  dxstrm_ << step->infostr() << endl;
  timestrm_ << "  member " << nsteps_-1 << " " << step->time_ << " \"stepf " << nsteps_ << "\" " << endl;
}

DXDumper::FldWriter::FldWriter(string outdir, string fieldname)
  : dxobj_(0)
{
  string outname = outdir+"/"+fieldname+".dx";
  strm_.open(outname.c_str());
  if(!strm_) {
    cerr << "Can't open output file " << outname << endl;
    abort();
  }
  cout << "     " << outname << endl;
}

DXDumper::FldWriter::~FldWriter()
{
  strm_ << "# time series " << endl;
  strm_ << "object " << ++dxobj_ << " series" << endl;
  int istep(0);
  for(list< pair<float,int> >::iterator tit(timesteps_.begin());tit!=timesteps_.end();tit++)
    {
      strm_ << "  member " << istep++ << " " << tit->first << " " << tit->second << endl;
    }
  strm_ << endl;
  
  strm_ << "default " << dxobj_ << endl;
  strm_ << "end" << endl;
}

DXDumper::Step::Step(DataArchive * da, string tsdir, int index, double time, int fileindex, 
		     const map<string,DXDumper::FldWriter*> & fldwriters, bool bin, bool onedim)
  :
  Dumper::Step(tsdir, index, time),
  da_(da), 
  fileindex_(fileindex),
  fldwriters_(fldwriters),
  bin_(bin), onedim_(onedim)
{
  fldstrm_ << "object \"step " << fileindex_+1 << "\" class group" << endl;
}

static
double
REMOVE_SMALL(double v)
{
  if(fabs(v)<FLT_MIN) return 0;
  else return v;
}

void
DXDumper::Step::storeGrid()
{
  // FIXME: why arent I doing this (lazy ...)
}

void
DXDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
{
  if(fieldname=="p.particleID") return;
  
  FldWriter * fldwriter = fldwriters_.find(fieldname)->second;
  ostream & os = fldwriter->strm_;
  
  GridP grid = da_->queryGrid(time_);
  
  const Uintah::TypeDescription* subtype = td->getSubType();
  
  // only support level 0 for now
  int lnum = 0;
  LevelP level = grid->getLevel(lnum);
  
  string dmode;
  if(bin_) {
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
    
    os << "# step " << index_ << " positions" << endl;
    if(onedim_)
      {
	os << "object " << ++fldwriter->dxobj_ << " class array type float items " 
	   << nnodes[0] << " data follows " << endl;
	for(int i=0;i<nnodes[0];i++)
	  {
	    os << x0(0)+dx[0]*(i-minind[0]) << endl;
	  }
      }
    else
      {
	os << "object " << ++fldwriter->dxobj_ << " class gridpositions counts " 
	   << nnodes[0] << " " << nnodes[1] << " " << nnodes[2] << endl;
	os << "origin " << x0(0)-minind[0]*dx[0] << " " << x0(1)-minind[1]*dx[1] << " " << x0(2)-minind[2]*dx[2] << endl;
	os << "delta " << dx[0] << " " << 0. << " " << 0. << endl;
	os << "delta " << 0. << " " << dx[1] << " " << 0. << endl;
	os << "delta " << 0. << " " << 0. << " " << dx[2] << endl;
	os << endl;
      }
    posnobj = fldwriter->dxobj_;
    
    os << "# step " << index_ << " connections" << endl;
    if(onedim_)
      {
	os << "object " << ++fldwriter->dxobj_ << " class gridconnections counts " 
	   << nnodes[0] << endl; // dx wants node counts here !
	os << "attribute \"element type\" string \"lines\"" << endl;
      } 
    else
      {
	os << "object " << ++fldwriter->dxobj_ << " class gridconnections counts " 
	   << nnodes[0] << " " << nnodes[1] << " " << nnodes[2] << endl; // dx wants node counts here !
	os << "attribute \"element type\" string \"cubes\"" << endl;
      }
    os << "attribute \"ref\" string \"positions\"" << endl;
    os << endl;
    connobj = fldwriter->dxobj_;
    
    if(onedim_)
      nparts = strides(0);
    else
      nparts = strides(0)*strides(1)*strides(2);
    
  } else {
    nparts = 0;
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      
      ConsecutiveRangeSet matls = da_->queryMaterials("p.x", patch, time_);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	
	ParticleVariable<Point> partposns;
	da_->query(partposns, "p.x", matl, patch, time_);
	ParticleSubset* pset = partposns.getParticleSubset();
	nparts += pset->numParticles();
      }
    }
    
    os << "# step " << index_ << " positions" << endl;
    os << "object " << ++fldwriter->dxobj_ << " class array rank 1 shape 3 items " << nparts;
    os << dmode << " data follows " << endl;;
    
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      
      ConsecutiveRangeSet matls = da_->queryMaterials("p.x", patch, time_);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	
	ParticleVariable<Point> partposns;
	da_->query(partposns, "p.x", matl, patch, time_);
	ParticleSubset* pset = partposns.getParticleSubset();
	for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
	  Point xpt = partposns[*iter];
	  if(!bin_)
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
    posnobj = fldwriter->dxobj_;
  }
  
  int ncomps, rank;
  string shp, source;
  vector<float> minval, maxval;
  
  /*if(1)*/ { // FIXME: skip p.x
    int nvals;
    switch (td->getType()) { 
    case Uintah::TypeDescription::NCVariable:
      if(onedim_)
	nvals = strides(0);
      else
	nvals = strides(0)*strides(1)*strides(2);
      source = "nodes";
      break;
    case Uintah::TypeDescription::CCVariable:
      if(onedim_)
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
      
      ConsecutiveRangeSet matls = da_->queryMaterials(fieldname, patch, time_);
      
      // loop over materials
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
	  matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
	
	switch(subtype->getType()) {
	case Uintah::TypeDescription::float_type:
	  {
	    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
	      ParticleVariable<float> value;
	      da_->query(value, fieldname, matl, patch, time_);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		float val = REMOVE_SMALL(value[*iter]);
		vals[ipart] = val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<float> value;
	      da_->query(value, fieldname, matl, patch, time_);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
		IntVector ind(*iter-minind);
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		float val = REMOVE_SMALL(value[*iter]);
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else {
	      NCVariable<float> value;
	      da_->query(value, fieldname, matl, patch, time_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
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
	      da_->query(value, fieldname, matl, patch, time_);
	      ParticleSubset* pset = value.getParticleSubset();
	      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
		float val = REMOVE_SMALL(value[*iter]);
		vals[ipart++] = val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else if(td->getType()==Uintah::TypeDescription::CCVariable) {
	      CCVariable<double> value;
	      da_->query(value, fieldname, matl, patch, time_);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		IntVector ind(*iter-minind);
		cout << "index: " << ind << endl;
		if(onedim_ && (ind[1]!=midind[1] ||ind[2]!=midind[2])) continue;
		
		int ioff = ind[2]+strides[2]*(ind[1]+strides[1]*ind[0]);
		double val = REMOVE_SMALL(value[*iter]);
		cout << "  val = " << val << endl;
		
		vals[ioff] += val;
		if(val<minval[0]) minval[0] = val;
		if(val>maxval[0]) maxval[0] = val;
	      }
	    } else {
	      NCVariable<double> value;
	      da_->query(value, fieldname, matl, patch, time_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
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
	      da_->query(value, fieldname, matl, patch, time_);
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
	      da_->query(value, fieldname, matl, patch, time_);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
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
	      da_->query(value, fieldname, matl, patch, time_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
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
	      da_->query(value, fieldname, matl, patch, time_);
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
	      da_->query(value, fieldname, matl, patch, time_);
	      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
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
	      da_->query(value, fieldname, matl, patch, time_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
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
	      da_->query(value, fieldname, matl, patch, time_);
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
	      da_->query(value, fieldname, matl, patch, time_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
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
	      da_->query(value, fieldname, matl, patch, time_);
	      for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
		if(onedim_ && ((*iter)[1]!=midind[1] ||(*iter)[2]!=midind[2])) continue;
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
    
    os << "# step " << index_ << " values" << endl;
    os << "object " << ++fldwriter->dxobj_ << " class array rank " << rank << " " << shp << " items " << nparts;
    os << dmode << " data follows " << endl;;
    int ioff = 0;
    for(int iv=0;iv<nvals;iv++) { 
      for(int ic=0;ic<ncomps;ic++) 
	if(!bin_)
	  os << vals[ioff++] << " ";
	else
	  os.write((char *)&vals[ioff++], sizeof(float));
      if(!bin_) os << endl;
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
    
    dataobj = fldwriter->dxobj_;
  }
  
  // build field object
  os << "# step " << index_ << " " << fieldname << " field" << endl;
  os << "object " << ++fldwriter->dxobj_ << " class field" << endl;
  if(posnobj!=-1) os << "  component \"positions\" value " << posnobj << endl;
  if(connobj!=-1) os << "  component \"connections\" value " << connobj << endl;
  if(dataobj!=-1) os << "  component \"data\" value " << dataobj << endl;
  os << endl;
  
  fldwriter->timesteps_.push_back( pair<float,int>(time_, fldwriter->dxobj_));
  
  int istep = fileindex_+1;
  dxstrm_ << "# step " << istep << " " << fieldname << " minimum " << endl;
  dxstrm_ << "object \"" << fieldname << " " << istep << " min\" "
	 << "class array type float rank " << rank << " " << shp << " items " << 1
	 << " data follows" << endl;
  for(int ic=0;ic<ncomps;ic++)
    dxstrm_ << minval[ic] << " ";
  dxstrm_ << endl << endl;
  
  dxstrm_ << "# step " << istep << " " << fieldname << " maximum " << endl;
  dxstrm_ << "object \"" << fieldname << " " << istep << " max\" "
	 << "class array type float rank " << rank << " " << shp << " items " << 1
	 << " data follows" << endl;
  for(int ic=0;ic<ncomps;ic++)
    dxstrm_ << maxval[ic] << " ";
  dxstrm_ << endl << endl;
  
  dxstrm_ << "object \"" << fieldname << " " << istep << " filename\" class string \"" << fieldname << ".dx\"" << endl;
  dxstrm_ << endl;
  
  dxstrm_ << "# step " << istep << " info " << endl;
  dxstrm_ << "object \"" << fieldname << " " << istep << " info\" class group" << endl;
  dxstrm_ << "  member \"minimum\" \"" << fieldname << " " << istep << " min\"" << endl;
  dxstrm_ << "  member \"maximum\" \"" << fieldname << " " << istep << " max\"" << endl;
  dxstrm_ << "  member \"filename\" \"" << fieldname << " " << istep << " filename\"" << endl;
  dxstrm_ << "  attribute \"source\" string \"" << source << "\"" << endl;
  dxstrm_ << endl;
  
  fldstrm_ << "  member \"" << fieldname <<  "\" " << "\"" << fieldname << " " << istep << " info\"" << endl;
}

// -----------------------------------------------------------------------------

class HistogramDumper : public Dumper 
{
public:
  HistogramDumper(DataArchive* da, string basedir, int nbins=256);
  
  string directoryExt() const { return "hist"; }
  void addField(string /*fieldname*/, const Uintah::TypeDescription * /*theType*/) {}
  
  class Step : public Dumper::Step {
  public:
    Step(DataArchive * da, string outdir, int index, double time, int nbins);
    
    void storeGrid () {}
    void storeField(string fieldname, const Uintah::TypeDescription * type);
    
    string infostr() const { return stepdname_; }
    
  private:
    string stepdname_;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#  pragma set woff 1424 // Template parameter not used in declaring arguments.
#endif                  // This turns of SGI compiler warning.
    template <class ElemT>
    void _binvals(LevelP level, Uintah::TypeDescription::Type type, 
                  const string & fieldname, int imat, int idiag,
                  double & minval, double & maxval, vector<int> & counts, string & ext);
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#  pragma reset woff 1424
#endif  

    DataArchive* da_;
    string       basedir_;
    int          nbins_;
  };
  
  //
  Step * addStep(int index, double time);
  void   finishStep(Dumper::Step * s);
  
private:
  ofstream idxos_;
  int      nbins_;
  FILE*    filelist_;
};

HistogramDumper::HistogramDumper(DataArchive* da, string basedir, int nbins)
  : Dumper(da, basedir), nbins_(nbins)
{
  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  
  string outdir = this->createDirectory();
  
  // set up the file that contains a list of all the files
  string filelistname = outdir + string("/") + string("timelist");
  filelist_ = fopen(filelistname.c_str(),"w");
  if (!filelist_) {
    cerr << "Can't open output file " << filelistname << endl;
    abort();
  }
}

HistogramDumper::Step * 
HistogramDumper::addStep(int index, double time)
{
  return scinew Step(this->archive(), this->dirName(time), index, time, nbins_);
}

void
HistogramDumper::finishStep(Dumper::Step * s)
{
  fprintf(filelist_, "%10d %16.8g  %20s\n", s->index_, s->time_, s->infostr().c_str());
}

HistogramDumper::Step::Step(DataArchive * da, string tsdir, int index, double time,  int nbins)
  : 
  Dumper::Step(tsdir, index, time), da_(da), nbins_(nbins)
{
}

static inline double MIN(double a, double b) { if(a<b) return a; return b; }
static inline double MAX(double a, double b) { if(a>b) return a; return b; }

static void _binval(vector<int> & bins, double minval, double maxval, double val)
{
  if(val<minval || val>maxval) return;
  int nbins = (int)bins.size();
  int ibin = (int)(nbins*(val-minval)/(maxval-minval+1.e-5));
  bins[ibin]++;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#  pragma set woff 1424 // Template parameter not used in declaring arguments.
#endif                  // This turns of SGI compiler warning.
template <class ElemT>
void 
HistogramDumper::Step::
_binvals(LevelP level, Uintah::TypeDescription::Type theType, 
         const string & fieldname, int imat, int idiag,
         double & minval, double & maxval, vector<int> & counts, string & ext)
{
  ScalarDiagGen<ElemT> diaggen;
  
  ext = diaggen.name(idiag);
  
  minval =  FLT_MAX;
  maxval = -FLT_MAX;
  
  IntVector minind, maxind;
  level->findNodeIndexRange(minind, maxind);
  
  for(int ipass=0;ipass<2;ipass++) {
    for(Level::const_patchIterator iter = level->patchesBegin();
        iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      ConsecutiveRangeSet matls = da_->queryMaterials(fieldname, patch, time_);
      
      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
          matlIter != matls.end(); matlIter++) {
	const int matl = *matlIter;
        if (matl!=imat) break;
        
        switch(theType){
        case Uintah::TypeDescription::NCVariable:
          {
            NCVariable<ElemT>  value;
            NCVariable<double> Mvalue;
            
            da_->query(value, fieldname, matl, patch, time_);
            da_->query(Mvalue, "g.mass", matl, patch, time_);
            
            for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
              if(outside(*iter, minind, maxind)) continue;
              
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
            da_->query(value, fieldname, matl, patch, time_);
            
            for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
              if(outside(*iter, minind, maxind)) continue;
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
            da_->query(value, fieldname, matl, patch, time_);
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
        } // theType switch
        
      } // materials
    } // patches

    if(minval>maxval) break;
    if(ipass==0) {
      for(size_t ibin=0;ibin<counts.size();ibin++) counts[ibin] = 0;
    }
    
  } // pass
  
}
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#  pragma reset woff 1424
#endif

void
HistogramDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
{
  cout << "   " << fieldname << endl;
  
  GridP grid = da_->queryGrid(time_);
  
  int nmats = 0;
  for(int l=0;l<=0;l++) {
    LevelP level = grid->getLevel(l);
    for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      ConsecutiveRangeSet matls= da_->queryMaterials(fieldname, patch, time_);
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
      vector<int> bins(nbins_);
      
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
      
      string fname   = this->fileName(fieldname+ext, imat, "hist");
      cout << "     " << fname << endl;
      cout << "       mat " << imat << ", "
           << "range = " << minval << "," << maxval
           << endl;
      
      ofstream os(fname.c_str());
      os << "# time = " << time_ << ", field = " 
         << fieldname << ", mat " << imat << endl;
      os << "# min = " << minval << endl;
      os << "# max = " << maxval << endl;
      
      for(int ibin=0;ibin<nbins_;ibin++) {
        double xmid = minval+(ibin+0.5)*(maxval-minval)/nbins_;
        os << xmid << " " << bins[ibin] << endl;
      }
      
      if(!os)
        throw InternalError("Failed to write histogram file '"+fname+"'");
      
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
  cerr << "  -basename     [bnm]    output basename\n";
  cerr << "  -field        [fld]    field to dump\n";
  cerr << "  -timesteplow  [int]    (only outputs timestep from int)\n";
  cerr << "  -timestephigh [int]    (only outputs timesteps upto int)\n";
  cerr << "  -timestepinc  [int]    (only outputs every int timesteps)\n";
  cerr << "  -format       [fmt]    output format, one of (text,ensight,opendx,histogram)\n";
  cerr << "  text options:" << endl;
  cerr << "      -onedim            generate one dim plots\n";
  cerr << "      -tseries           generate single time series file\n";
  cerr << "  histogram options:" << endl;
  cerr << "      -nbins             number of particle bins\n";
  cerr << "  ensight options:" << endl;
  cerr << "      -withpart          include particles\n";
  cerr << "      -onemesh           only store a single mesh\n";
  cerr << "      -bin               dump in binary format\n";
  cerr << "  opendx options:" << endl;
  cerr << "      -onedim            generate one dim plots\n";
  cerr << "      -bin               dump in binary format\n";
  exit(1);
}

int
main(int argc, char** argv)
{
  /*
   * Parse arguments
   */
  //bool do_verbose = false;
  int time_step_lower = 0;
  int time_step_upper = INT_MAX;
  int time_step_inc   = 1;
  string basedir      = "";
  string fieldnames   = "";
  string fmt    = "text";
  bool binary   = false;
  bool onemesh  = false;
  bool onedim   = false;
  bool tseries  = false;
  bool withpart = false;
  int nbins(256);
  
  int args = argc-1;
  for(int i=1;i<argc;i++){
    string s=argv[i];
    
    if (s == "-verbose") {
      //do_verbose = true;
    } else if (s == "-timesteplow") {
      time_step_lower = (int)strtoul(argv[++i],(char**)NULL,10);
      args -= 2;
    } else if (s == "-timestephigh") {
      time_step_upper = (int)strtoul(argv[++i],(char**)NULL,10);
      args -= 2;
    } else if (s == "-timestepinc") {
      time_step_inc = (int)strtoul(argv[++i],(char**)NULL,10);
      args -= 2;
    } else if (s == "-field") {
      fieldnames = argv[++i];
      args -= 2;
    } else if (s == "-basename") {
      basedir = argv[++i];
      args -= 2;
    } else if (s == "-format") {
      fmt = argv[++i];
      args -= 2;
    } else if (s == "-bin") {
      binary = true;
      args -= 1;
    } else if (s == "-onemesh") {
      onemesh = true;
      args -= 1;
    } else if (s == "-onedim") {
      onedim = true;
      args -= 1;
    } else if (s == "-nbins") {
      nbins = (int)strtoul(argv[++i],(char**)NULL,10);
      args -= 2;
    } else if (s == "-tseries") {
      tseries = true;
      args -= 1;
    } else if (s == "-withpart") {
      withpart = true;
      args -= 1;
    } else if( (s == "-help") || (s == "-h") ) {
      usage( "", argv[0] );
      args -= 1;
    }
  }
  if(args!=1){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }
  string filebase = argv[argc-1];
  
  if(!basedir.size())
    {
      basedir = filebase.substr(0, filebase.find('.'));
    }
  
  try {
    cout << "filebase: " << filebase << endl;
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
    if(time_step_upper>=(int)index.size()) time_step_upper = (int)index.size()-1;
    if(time_step_inc<=0)                   time_step_inc   = 1;
    
    // build list of variables to dump
    list<typed_varname> dumpvars;
    int nvars = (int)allvars.size();
    if(fieldnames.size()) {
      vector<string> requested_fields = split(fieldnames,',',false);
      int nreq = (int)requested_fields.size();
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
      dumper = scinew TextDumper(da, basedir, onedim, tseries);
    } else if(fmt=="ensight") {
      dumper = scinew EnsightDumper(da, basedir, binary, onemesh, withpart);
    } else if(fmt=="dx" || fmt=="opendx") {
      dumper = scinew DXDumper(da, basedir, binary, onedim);
    } else if(fmt=="histogram" || fmt=="hist") {
      dumper = new HistogramDumper(da, basedir, nbins);
    } else {
      cerr << "Failed to find match to format '" + fmt + "'" << endl;
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
    }
    
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
}

