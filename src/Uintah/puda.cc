
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

#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/NodeIterator.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <iostream>
#include <string>
#include <vector>

using SCICore::Exceptions::Exception;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace std;
using namespace Uintah;
using namespace PSECore::XMLUtil;

void usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
	cerr << "Error parsing argument: " << badarg << '\n';
    cerr << "Usage: " << progname << " [options] <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h[elp]\n";
    cerr << "  -timesteps\n";
    cerr << "  -gridstats\n";
    cerr << "  -listvariables\n";
    cerr << "  -varsummary\n\n";

    cerr << "USAGE IS NOT FINISHED\n\n";
    exit(1);
}

int main(int argc, char** argv)
{
   /*
    * Default values
    */
   bool do_timesteps=false;
   bool do_gridstats=false;
   bool do_listvars=false;
   bool do_varsummary=false;
   string filebase;
   
   /*
    * Parse arguments
    */
   for(int i=1;i<argc;i++){
      string s=argv[i];
      if(s == "-timesteps"){
	 do_timesteps=true;
      } else if(s == "-gridstats"){
	 do_gridstats=true;
      } else if(s == "-listvariables"){
	 do_listvars=true;
      } else if(s == "-varsummary"){
	 do_varsummary=true;
      } else if( (s == "-help") || (s == "-h") ) {
	usage( "", argv[0] );
      } else {
	 if(filebase!="")
	    usage(s, argv[0]);
	 else
	    filebase = argv[i];
      }
   }

   if(filebase == ""){
      cerr << "No archive file specified\n";
      usage("", argv[0]);
   }
   
   try {
      XMLPlatformUtils::Initialize();
   } catch(const XMLException& toCatch) {
      cerr << "Caught XML exception: " << toString(toCatch.getMessage()) 
	   << '\n';
      exit( 1 );
   }

   try {
      DataArchive* da = new DataArchive(filebase);
      
      if(do_timesteps){
	 vector<int> index;
	 vector<double> times;
	 da->queryTimesteps(index, times);
	 ASSERTEQ(index.size(), times.size());
	 cout << "There are " << index.size() << " timesteps:\n";
	 for(int i=0;i<index.size();i++)
	    cout << index[i] << ": " << times[i] << '\n';
      }
      if(do_gridstats){
	 vector<int> index;
	 vector<double> times;
	 da->queryTimesteps(index, times);
	 ASSERTEQ(index.size(), times.size());
	 cout << "There are " << index.size() << " timesteps:\n";
	 for(int i=0;i<index.size();i++){
	    cout << index[i] << ": " << times[i] << '\n';
	    GridP grid = da->queryGrid(times[i]);
	    grid->performConsistencyCheck();
	    grid->printStatistics();
	 }
      }
      if(do_listvars){
	 vector<string> vars;
	 vector<const TypeDescription*> types;
	 da->queryVariables(vars, types);
	 cout << "There are " << vars.size() << " variables:\n";
	 for(int i=0;i<vars.size();i++){
	    cout << vars[i] << ": " << types[i]->getName() << '\n';
	 }
      }
      if(do_varsummary){
	 vector<string> vars;
	 vector<const TypeDescription*> types;
	 da->queryVariables(vars, types);
	 ASSERTEQ(vars.size(), types.size());
	 cout << "There are " << vars.size() << " variables:\n";

	 vector<int> index;
	 vector<double> times;
	 da->queryTimesteps(index, times);
	 ASSERTEQ(index.size(), times.size());
	 cout << "There are " << index.size() << " timesteps:\n";

	 for(int t=0;t<times.size();t++){
	    double time = times[t];
	    cerr << "time = " << time << "\n";
	    GridP grid = da->queryGrid(time);
	    for(int v=0;v<vars.size();v++){
	       std::string var = vars[v];
	       const TypeDescription* td = types[v];
	       const TypeDescription* subtype = td->getSubType();
	       cerr << "\tVariable: " << var << ", type " << td->getName() << "\n";
	       for(int l=0;l<grid->numLevels();l++){
		  LevelP level = grid->getLevel(l);
		  for(Level::const_regionIterator iter = level->regionsBegin();
		      iter != level->regionsEnd(); iter++){
		     const Region* region = *iter;
		     cerr << "\t\tRegion: " << region->getID() << "\n";
		     int numMatls = da->queryNumMaterials(var, region, time);
		     for(int matl=0;matl<numMatls;matl++){
			cerr << "\t\t\tMaterial: " << matl << "\n";
			switch(td->getType()){
			case TypeDescription::ParticleVariable:
			   switch(subtype->getType()){
			   case TypeDescription::double_type:
			      {
				 ParticleVariable<double> value;
				 da->query(value, var, matl, region, time);
				 ParticleSubset* pset = value.getParticleSubset();
				 cerr << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
				 if(pset->numParticles() > 0){
				    double min, max;
				    ParticleSubset::iterator iter = pset->begin();
				    min=max=value[*iter++];
				    for(;iter != pset->end(); iter++){
				       min=Min(min, value[*iter]);
				       max=Max(max, value[*iter]);
				    }
				    cerr << "\t\t\t\tmin value: " << min << '\n';
				    cerr << "\t\t\t\tmax value: " << max << '\n';
				 }
			      }
			      break;
			   case TypeDescription::Point:
			      {
				 ParticleVariable<Point> value;
				 da->query(value, var, matl, region, time);
				 ParticleSubset* pset = value.getParticleSubset();
				 cerr << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
				 if(pset->numParticles() > 0){
				    Point min, max;
				    ParticleSubset::iterator iter = pset->begin();
				    min=max=value[*iter++];
				    for(;iter != pset->end(); iter++){
				       min=Min(min, value[*iter]);
				       max=Max(max, value[*iter]);
				    }
				    cerr << "\t\t\t\tmin value: " << min << '\n';
				    cerr << "\t\t\t\tmax value: " << max << '\n';
				 }
			      }
			      break;
			   case TypeDescription::Vector:
			      {
				 ParticleVariable<Vector> value;
				 da->query(value, var, matl, region, time);
				 ParticleSubset* pset = value.getParticleSubset();
				 cerr << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
				 if(pset->numParticles() > 0){
				    double min, max;
				    ParticleSubset::iterator iter = pset->begin();
				    min=max=value[*iter++].length2();
				    for(;iter != pset->end(); iter++){
				       min=Min(min, value[*iter].length2());
				       max=Max(max, value[*iter].length2());
				    }
				    cerr << "\t\t\t\tmin magnitude: " << sqrt(min) << '\n';
				    cerr << "\t\t\t\tmax magnitude: " << sqrt(max) << '\n';
				 }
			      }
			      break;
			   default:
			      cerr << "Particle Variable of unknown type: " << subtype->getType() << '\n';
			      break;
			   }
			   break;
			case TypeDescription::NCVariable:
			   switch(subtype->getType()){
			   case TypeDescription::double_type:
			      {
				 NCVariable<double> value;
				 da->query(value, var, matl, region, time);
				 cerr << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
				 IntVector dx(value.getHighIndex()-value.getLowIndex());
				 if(dx.x() && dx.y() && dx.z()){
				    double min, max;
				    NodeIterator iter = region->getNodeIterator();
				    min=max=value[*iter];
				    for(;!iter.done(); iter++){
				       min=Min(min, value[*iter]);
				       max=Max(max, value[*iter]);
				    }
				    cerr << "\t\t\t\tmin value: " << min << '\n';
				    cerr << "\t\t\t\tmax value: " << max << '\n';
				 }
			      }
			      break;
			   case TypeDescription::Point:
			      {
				 NCVariable<Point> value;
				 da->query(value, var, matl, region, time);
				 cerr << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
				 IntVector dx(value.getHighIndex()-value.getLowIndex());
				 if(dx.x() && dx.y() && dx.z()){
				    Point min, max;
				    NodeIterator iter = region->getNodeIterator();
				    min=max=value[*iter];
				    for(;!iter.done(); iter++){
				       min=Min(min, value[*iter]);
				       max=Max(max, value[*iter]);
				    }
				    cerr << "\t\t\t\tmin value: " << min << '\n';
				    cerr << "\t\t\t\tmax value: " << max << '\n';
				 }
			      }
			      break;
			   case TypeDescription::Vector:
			      {
				 NCVariable<Vector> value;
				 da->query(value, var, matl, region, time);
				 cerr << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
				 IntVector dx(value.getHighIndex()-value.getLowIndex());
				 if(dx.x() && dx.y() && dx.z()){
				    double min, max;
				    NodeIterator iter = region->getNodeIterator();
				    min=max=value[*iter].length2();
				    for(;!iter.done(); iter++){
				       min=Min(min, value[*iter].length2());
				       max=Max(max, value[*iter].length2());
				    }
				    cerr << "\t\t\t\tmin magnitude: " << sqrt(min) << '\n';
				    cerr << "\t\t\t\tmax magnitude: " << sqrt(max) << '\n';
				 }
			      }
			      break;
			   default:
			      cerr << "Particle Variable of unknown type: " << subtype->getType() << '\n';
			      break;
			   }
			   break;
			default:
			   cerr << "Variable of unknown type: " << td->getType() << '\n';
			   break;
			}
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
}

//
// $Log$
// Revision 1.3  2000/05/21 08:19:04  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.2  2000/05/20 19:54:52  dav
// browsing puda, added a couple of things to usage, etc.
//
// Revision 1.1  2000/05/20 08:09:01  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
//
