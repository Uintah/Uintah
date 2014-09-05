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

/*
 *  Support for printing out Tecplot data added by Patric Hu
 *  Department of Mechanical Enginerring, U of U, 2003.
 *
 *  Currently it only supports CCVariables.
 * 
 * Usage of converting Uintah data archive to a tecplot data file
 * puda -tecplot <i_xd> <uda directory> :
 *       print all CCVariables into different tecplot data files 
 * puda -tecplot <i_xd> <CCVariable's Name> <uda directory>:
 *       print one CCVariable into a tecplot data file
 * puda -tecplot <i_xd> <tskip> <uda directory>:
 *       print all CCVariables into different tecplot data files
 *       by every tskip time steps
 * puda -tecplot <i_xd> <CCVariable's Name> <tskip> <uda directory>:
 *       print one CCVariable into a tecplot data file
 *       by every tskip time steps
 * i_xd may be i_1d, i_2d, i_3d for 1D, 2D and 3D problem
 *
 */

#include <Packages/Uintah/StandAlone/tools/puda/puda.h>

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>

#include <Packages/Uintah/StandAlone/tools/puda/asci.h>
#include <Packages/Uintah/StandAlone/tools/puda/jim1.h>
#include <Packages/Uintah/StandAlone/tools/puda/rtdata.h>
#include <Packages/Uintah/StandAlone/tools/puda/tecplot.h>
#include <Packages/Uintah/StandAlone/tools/puda/util.h>
#include <Packages/Uintah/StandAlone/tools/puda/varsummary.h>

#include <Core/Containers/Array3.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <stdio.h>
#include <math.h>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

/////////////////////////////////////////////////////////////////
// Pre-declarations:
//
void printParticleVariable(DataArchive* da, 
			   string particleVariable,
			   unsigned long time_step_lower,
			   unsigned long time_step_upper);

/////////////////////////////////////////////////////////////////

void
usage( const std::string& badarg, const std::string& progname )
{
  if(badarg != "")
    cerr << "Error parsing argument: " << badarg << endl;
  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -h[elp]\n";
  cerr << "  -timesteps\n";
  cerr << "  -gridstats\n";
  cerr << "  -listvariables\n";
  cerr << "  -varsummary\n";
  cerr << "  -jim1\n";
  cerr << "  -partvar <variable name>\n";
  cerr << "  -asci\n";
  cerr << "  -tecplot <variable name>\n";
  cerr << "  -no_extra_cells     (Excludes extra cells when iterating over cells.\n";
  cerr << "                       Default is to include extra cells.)\n";
  cerr << "  -cell_stresses\n";
  cerr << "  -rtdata <output directory>\n";
  cerr << "  -PTvar\n";
  cerr << "  -ptonly             (prints out only the point location\n";
  cerr << "  -patch              (outputs patch id with data)\n";
  cerr << "  -material           (outputs material number with data)\n";
  cerr << "  -NCvar <double | float | point | vector>\n";
  cerr << "  -CCvar <double | float | point | vector>\n";
  cerr << "  -verbose            (prints status of output)\n";
  cerr << "  -timesteplow <int>  (only outputs timestep from int)\n";
  cerr << "  -timestephigh <int> (only outputs timesteps upto int)\n";
  cerr << "  -matl <int>         (only outputs data for matl (for -jim1 only))\n";
  cerr << "*NOTE* to use -PTvar or -NVvar -rtdata must be used\n";
  cerr << "*NOTE* ptonly, patch, material, timesteplow, timestephigh "
       << "are used in conjuntion with -PTvar.\n\n";
    
  cerr << "USAGE IS NOT FINISHED\n\n";
  exit(1);
}

void
gridstats( DataArchive* da,
           const bool tslow_set, 
           const bool tsup_set,
           unsigned long & time_step_lower,
           unsigned long & time_step_upper )
{
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
      
  findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, time_step_upper);
           
  for( unsigned long t = time_step_lower; t <= time_step_upper; t++ ) {
    double time = times[t];
    cout << "__________________________________"<<endl;
    cout << "Timestep " << t << ": " << time << endl;
    GridP grid = da->queryGrid(time);
    grid->performConsistencyCheck();
    grid->printStatistics();

    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);
      cout << "Level: index " << level->getIndex() << ", id " << level->getID() << endl;

      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        cout << *patch << endl;
        cout << "\t   BC types: x- " << patch->getBCType(Patch::xminus) << ", x+ "<<patch->getBCType(Patch::xplus)
             << ", y- "<< patch->getBCType(Patch::yminus) << ", y+ "<< patch->getBCType(Patch::yplus)
             << ", z- "<< patch->getBCType(Patch::zminus) << ", z+ "<< patch->getBCType(Patch::zplus)<< endl;
      }
    }
  }
} // end gridstats()


int
main(int argc, char** argv)
{
  if (argc <= 1) {
    // Print out the usage and die
    usage("", argv[0]);
  }

  CommandLineFlags clf;

  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-timesteps"){
      clf.do_timesteps=true;}
    else if( s == "-no_extra_cells" ) {
      clf.use_extra_cells = false;
    }
    else if(s == "-tecplot"){ 
      clf.do_tecplot = true;
      if(argc == 4) {
        clf.do_all_ccvars = true;
        clf.i_xd = argv[i+1];
        clf.tskip = 1;
      } else if(argc == 5){
        clf.do_all_ccvars = true;
        clf.i_xd = argv[i+1];
        clf.tskip = atoi(argv[i+2]);
      } else if(argc == 6 ) {
        clf.i_xd = argv[i+1];
        clf.ccVarInput = argv[i+2];
        clf.tskip = atoi(argv[i+3]);
        if (clf.ccVarInput[0] == '-')
          usage("-tecplot <i_xd> <ccVariable name> <tskip> ", argv[0]);
      }
    } else if(s == "-gridstats"){
      clf.do_gridstats=true;
    } else if(s == "-listvariables"){
      clf.do_listvars=true;
    } else if(s == "-varsummary"){
      clf.do_varsummary=true;
    } else if(s == "-jim1"){
      clf.do_jim1=true;
    } else if(s == "-partvar"){
      clf.do_partvar=true;
      clf.particleVariable = argv[++i]; 
      if (clf.particleVariable[0] == '-') {
        usage("-partvar <particle variable name>", argv[0]);
      }
    } else if(s == "-asci"){
      clf.do_asci=true;
    } else if(s == "-cell_stresses"){
      clf.do_cell_stresses=true;
    } else if(s == "-rtdata") {
      clf.do_rtdata = true;
      if (++i < argc) {
        s = argv[i];
        if (s[0] == '-') {
          usage("-rtdata", argv[0]);
        }
        clf.raydatadir = s;
      }
    } else if(s == "-NCvar") {
      if (++i < argc) {
        s = argv[i];
        if (s == "double")
          clf.do_NCvar_double = true;
        else if (s == "float")
          clf.do_NCvar_float = true;
        else if (s == "point")
          clf.do_NCvar_point = true;
        else if (s == "vector")
          clf.do_NCvar_vector = true;
        else if (s == "matrix3")
          clf.do_NCvar_matrix3 = true;
        else
          usage("-NCvar", argv[0]);
      }
      else
        usage("-NCvar", argv[0]);
    } else if(s == "-CCvar") {
      if (++i < argc) {
        s = argv[i];
        if (s == "double")
          clf.do_CCvar_double = true;
        else if (s == "float")
          clf.do_CCvar_float = true;
        else if (s == "point")
          clf.do_CCvar_point = true;
        else if (s == "vector")
          clf.do_CCvar_vector = true;
        else if (s == "matrix3")
          clf.do_CCvar_matrix3 = true;
        else
          usage("-CCvar", argv[0]);
      }
      else
        usage("-CCvar", argv[0]);
    } else if(s == "-PTvar") {
      clf.do_PTvar = true;
    } else if (s == "-ptonly") {
      clf.do_PTvar_all = false;
    } else if (s == "-patch") {
      clf.do_patch = true;
    } else if (s == "-material") {
      clf.do_material = true;
    } else if (s == "-verbose") {
      clf.do_verbose = true;
    } else if (s == "-timesteplow") {
      clf.time_step_lower = strtoul(argv[++i],(char**)NULL,10);
      clf.tslow_set = true;
    } else if (s == "-timestephigh") {
      clf.time_step_upper = strtoul(argv[++i],(char**)NULL,10);
      clf.tsup_set = true;
    } else if (s == "-timestepinc") {
      clf.time_step_inc = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-matl") {
      clf.matl_jim1 = strtoul(argv[++i],(char**)NULL,10);
    } else if( (s == "-help") || (s == "-h") ) {
      usage( "", argv[0] );
    } else if( clf.filebase == "") {

      if( argv[i][0] == '-' ) { // File name can't start with a dash.
        usage( s, argv[0]);
      }
      clf.filebase = argv[i];
    } else {
      usage( s, argv[0]);
    }
  }

  if( clf.filebase == "" ) {
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    DataArchive* da = scinew DataArchive( clf.filebase );
    
    if(clf.do_timesteps){
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      cout << "There are " << index.size() << " timesteps:\n";
      for(int i=0;i<(int)index.size();i++)
	cout << index[i] << ": " << times[i] << endl;
    }
    //______________________________________________________________________
    //    DO GRIDSTATS
    if(clf.do_gridstats){
      gridstats( da, clf.tslow_set, clf.tsup_set, clf.time_step_lower, clf.time_step_upper );
    }
    if(clf.do_listvars){
      vector<string> vars;
      vector<const Uintah::TypeDescription*> types;
      da->queryVariables(vars, types);
      cout << "There are " << vars.size() << " variables:\n";
      for(int i=0;i<(int)vars.size();i++){
	cout << vars[i] << ": " << types[i]->getName() << endl;
      }
    }

    // Print a particular particle variable
    if (clf.do_partvar) {
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      if( !clf.tslow_set ) {
	clf.time_step_lower =0;
      }
      else if (clf.time_step_lower >= times.size()) {
	cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
	abort();
      }
      if( !clf.tsup_set ) {
	clf.time_step_upper = times.size() - 1;
      }
      else if( clf.time_step_upper >= times.size() ) {
	cerr << "timestephigh must be between 0 and " << times.size()-1 << endl;
	abort();
      }
      printParticleVariable( da, clf.particleVariable,
                             clf.time_step_lower, clf.time_step_upper );
    }
#if 0
    tecplot();
#endif
    //______________________________________________________________________
    //              V A R S U M M A R Y   O P T I O N
    if(clf.do_varsummary){
      varsummary( da, clf );
    }

    if( clf.do_jim1 ){
      jim1( da, clf );
    }

    if (clf.do_asci){
      asci( da, clf.tslow_set, clf.tsup_set, clf.time_step_lower, clf.time_step_upper );
    }

    //______________________________________________________________________
    //	       DO CELL STRESSES	
    if (clf.do_cell_stresses){
      vector<string> vars;
      vector<const Uintah::TypeDescription*> types;
      da->queryVariables(vars, types);
      ASSERTEQ(vars.size(), types.size());
      
      cout << "There are " << vars.size() << " variables:\n";
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      
      cout << "There are " << index.size() << " timesteps:\n";
      
      findTimestep_loopLimits( clf.tslow_set, clf.tsup_set,times, clf.time_step_lower, clf.time_step_upper );
      
      // obtain the desired timesteps
      unsigned long t = 0, start_time, stop_time;

      cout << "Time Step       Value\n";
      
      for(t = clf.time_step_lower; t <= clf.time_step_upper; t++){
	double time = times[t];
	cout << "    " << t + 1 << "        "  << time << endl;
      }
      cout << endl;
      if (t != (clf.time_step_lower +1)){
	cout << "Enter start time-step (1 - " << t << "): ";
	cin >> start_time;
	start_time--;
	cout << "Enter stop  time-step (1 - " << t << "): ";
	cin >> stop_time;
	stop_time--;
      }
      else 
      	if(t == (clf.time_step_lower + 1)){
	  start_time = t-1;
	  stop_time  = t-1;
	}
      // end of timestep acquisition
      
      for(t=start_time;t<=stop_time;t++){
	
	double time = times[t];
	cout << "time = " << time << endl;
	GridP grid = da->queryGrid(time);
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  
	  // only dumps out data if it is variable g.stressFS
	  if (var == "g.stressFS"){
	    const Uintah::TypeDescription* td = types[v];
	    const Uintah::TypeDescription* subtype = td->getSubType();
	    cout << "\tVariable: " << var << ", type " << td->getName() << endl;
	    for(int l=0;l<grid->numLevels();l++){
	      LevelP level = grid->getLevel(l);
	      for(Level::const_patchIterator iter = level->patchesBegin();
		  iter != level->patchesEnd(); iter++){
		const Patch* patch = *iter;
		cout << "\t\tPatch: " << patch->getID() << endl;
                ConsecutiveRangeSet matls =
		  da->queryMaterials(var, patch, time);
	        // loop over materials
	        for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		    matlIter != matls.end(); matlIter++){
		  int matl = *matlIter;
		  
		  // dumps header and variable info to file
		  ostringstream fnum, pnum, matnum; 
		  string filename;
		  unsigned long timestepnum=t+1;
		  fnum << setw(4) << setfill('0') << timestepnum;
                  pnum << setw(4) << setfill('0') << patch->getID();
		  matnum << setw(4) << setfill('0') << matl;
		  string partroot("stress.t");
                  string partextp(".p"); 
		  string partextm(".m");
		  filename = partroot+fnum.str()+partextp+pnum.str()+partextm+matnum.str();
		  ofstream partfile(filename.c_str());
		  partfile << "# x, y, z, st11, st12, st13, st21, st22, st23, st31, st32, st33" << endl;
		  
		  cout << "\t\t\tMaterial: " << matl << endl;
		  switch(td->getType()){
		  case Uintah::TypeDescription::NCVariable:
		    switch(subtype->getType()){
		    case Uintah::TypeDescription::Matrix3:{
		      NCVariable<Matrix3> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex()
			   << " to " << value.getHighIndex() << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			NodeIterator iter = patch->getNodeIterator();
			for(;!iter.done(); iter++){
			  partfile << (*iter).x() << " " << (*iter).y() << " " << (*iter).z()
				   << " " << (value[*iter])(0,0) << " " << (value[*iter])(0,1) << " " 
				   << (value[*iter])(0,2) << " " << (value[*iter])(1,0) << " "
				   << (value[*iter])(1,1) << " " << (value[*iter])(1,2) << " "
				   << (value[*iter])(2,0) << " " << (value[*iter])(2,1) << " "
                                   << (value[*iter])(2,2) << endl;
			}
		      }
		    }
		      break;
		    default:
		      cerr << "No Matrix3 Subclass avaliable." << subtype->getType() << endl;
		      break;
		    }
		    break;
		  default:
		    cerr << "No NC Variables avaliable." << td->getType() << endl;
		    break;
		  }
		}
	      }
	    }
	  }
	  else
	    cout << "No g.stressFS variables avaliable at time " << t << "." << endl;
	}
	if (start_time == stop_time)
	  t++;   
      }
    } // end do_cell_stresses
    
    //______________________________________________________________________
    //              DO RTDATA
    if (clf.do_rtdata) {
      rtdata( da, clf );
    }
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
} // end main()

////////////////////////////////////////////////////////////////////////////
//
// Print ParticleVariable
//
void
printParticleVariable( DataArchive* da, 
                       string particleVariable,
                       unsigned long time_step_lower,
                       unsigned long time_step_upper )
{
  // Check if the particle variable is available
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  bool variableFound = false;
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == particleVariable) variableFound = true;
  }
  if (!variableFound) {
    cerr << "Variable " << particleVariable << " not found\n"; 
    exit(1);
  }

  // Now that the variable has been found, get the data for all 
  // available time steps // from the data archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  //cout << "There are " << index.size() << " timesteps:\n";
      
  // Loop thru all time steps and store the volume and variable (stress/strain)
  for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
    double time = times[t];
    //cout << "Time = " << time << endl;
    GridP grid = da->queryGrid(time);

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patchIterator iter = level->patchesBegin(); 
      int patchIndex = 0;
      for(; iter != level->patchesEnd(); iter++){
	const Patch* patch = *iter;
        ++patchIndex; 

	// Loop thru all the variables 
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  const Uintah::TypeDescription* td = types[v];
	  const Uintah::TypeDescription* subtype = td->getSubType();

	  // Check if the variable is a ParticleVariable
	  if(td->getType() == Uintah::TypeDescription::ParticleVariable) { 

	    // loop thru all the materials
	    ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	    ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
	    for(; matlIter != matls.end(); matlIter++){
	      int matl = *matlIter;

	      // Find the name of the variable
	      if (var == particleVariable) {
		//cout << "Material: " << matl << endl;
		switch(subtype->getType()){
		case Uintah::TypeDescription::double_type:
		  {
		    ParticleVariable<double> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl; 
			cout << " " << pid[*iter];
                        cout << " " << value[*iter] << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::float_type:
		  {
		    ParticleVariable<float> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << pid[*iter];
                        cout << " " << value[*iter] << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::int_type:
		  {
		    ParticleVariable<int> value;
		    da->query(value, var, matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl;
			cout << " " << pid[*iter];
                        cout << " " << value[*iter] << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::Point:
		  {
		    ParticleVariable<Point> value;
		    da->query(value, var, matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << pid[*iter];
                        cout << " " << value[*iter](0) 
                             << " " << value[*iter](1)
                             << " " << value[*iter](2) << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::Vector:
		  {
		    ParticleVariable<Vector> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << pid[*iter];
			cout << " " << value[*iter][0] 
                             << " " << value[*iter][1]
                             << " " << value[*iter][2] << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::Matrix3:
		  {
		    ParticleVariable<Matrix3> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << pid[*iter];
                        for (int ii = 0; ii < 3; ++ii) {
                          for (int jj = 0; jj < 3; ++jj) {
			    cout << " " << value[*iter](ii,jj) ;
                          }
                        }
			cout << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::long64_type:
		  {
		    ParticleVariable<long64> value;
		    da->query(value, var, matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << value[*iter] << endl;
		      }
		    }
		  }
		break;
		default:
		  cerr << "Particle Variable of unknown type: " 
		       << subtype->getType() << endl;
		  break;
		}
	      } // end of var compare if
	    } // end of material loop
	  } // end of ParticleVariable if
	} // end of variable loop
      } // end of patch loop
    } // end of level loop
  } // end of time step loop
} // end printParticleVariable()
