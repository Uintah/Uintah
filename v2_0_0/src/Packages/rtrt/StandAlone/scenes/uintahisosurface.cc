
#define USE_HVBRICK 1

// now for the SCI stuff
#include <Core/Thread/Thread.h>
#include <Core/Math/MinMax.h>
#include <Core/Exceptions/Exception.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Dataflow/XMLUtil/XMLUtil.h>

// rtrt stuff
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/Group.h>
#ifdef USE_HVBRICK
#  include <Packages/rtrt/Core/HVolumeBrick.h>
#else
#  include <Packages/rtrt/Core/HVolume.cc>
#endif
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/Scene.h>
//#include <Packages/rtrt/Core/Slice.h>
#include <Packages/rtrt/Core/TimeObj.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Array3.h>
// undef some things that conflict
#undef None

#ifdef Success
#undef Success
#endif
// standard c files
#include <fstream>
#include <iostream>
#include <math.h>
#include <string.h>

//using namespace rtrt;
using namespace std;
using namespace SCIRun;
//using SCIRun::Thread;
using rtrt::Scene;
using rtrt::Color;
using rtrt::Material;
#ifdef USE_HVBRICK
using rtrt::HVolumeBrick;
#else
using rtrt::HVolume;
using rtrt::BrickArray3;
using rtrt::VMCell;
#endif
using rtrt::Group;
using rtrt::Phong;
using rtrt::Camera;
using rtrt::Light;
using rtrt::LinearBackground;
using rtrt::VolumeDpy;
using rtrt::TimeObj;


extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  double rate=3;
  char* file=0;
  int depth=3;
  int time_step_lower = -1;
  int time_step_upper = -1;
  string var("");
  bool debug = false;
  bool do_verbose = false;
  int mat = -1;
  
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-rate") == 0){
      i++;
      rate=atof(argv[i]);
    } else if(strcmp(argv[i], "-depth") == 0){
      i++;
      depth=atoi(argv[i]);
    } else if (strcmp(argv[i], "-timesteplow") == 0) {
      time_step_lower = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-timestephigh") == 0) {
      time_step_upper = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-mat") == 0) {
      mat = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-var") == 0) {
      var = string(argv[++i]);
    } else if (strcmp(argv[i], "-debug") == 0) {
      debug = true;
    } else if (strcmp(argv[i], "-v") == 0) {
      do_verbose = true;
    } else {
      if(file){
	cerr << "Unknown option: " << argv[i] << '\n';
	cerr << "Valid options for scene: " << argv[0] << '\n';
	cerr << " -rate\n";
	cerr << " -depth\n";
	cerr << " -timesteplow [int]\n";
	cerr << " -timestephigh [int]\n";
	return 0;
      }
      file=argv[i];
    }
  }

  if(strcmp(file, "") == 0) {
    cerr << "No archive file specified\n";
    return(0);
  }
  
  try {
    XMLPlatformUtils::Initialize();
  } catch(const XMLException& toCatch) {
    cerr << "Caught XML exception: " << toCatch.getMessage() 
	 << '\n';
    exit( 1 );
  }

  Color surf(.50000, 0.0, 0.00);
  Material* matl0=new Phong( surf*0.6, surf*0.6, 100, .4);
  VolumeDpy* dpy=new VolumeDpy();

  TimeObj* timeobj1=new TimeObj(rate);
  Group* timeblock;
  
  try {
    DataArchive* da = new DataArchive(file);

    cerr << "Done parsing dataArchive\n";
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;
    da->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    bool var_found = false;
    int var_index = 0;
    for (;var_index < vars.size(); var_index++) {
      cout << "vars[" << var_index << "] = " << vars[var_index] << endl;
      if (vars[var_index] == var) {
	var_found = true;
	break;
      }
    }
    if (!var_found) {
      cerr << "Variable \"" << var << "\" was not found.\n";
      cerr << "Aborting!!\n";
      exit(-1);
      //      var = vars[0];
    }
    
    vector<int> index;
    vector<double> times;
    da->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    // This will make sure that when we cast to an int, we don't burn
    // ourselves.
    ASSERTL3(times.size() < INT_MAX);
    cout << "There are " << index.size() << " timesteps:\n";
    
    //------------------------------
    // figure out the lower and upper bounds on the timesteps
    if (time_step_lower <= -1)
      time_step_lower = 0;
    else if (time_step_lower >= times.size()) {
      cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
      abort();
    }
    if (time_step_upper <= -1)
      time_step_upper = (int)(times.size()-1);
    else if (time_step_upper >= times.size()) {
      cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
      abort();
    }

    //------------------------------
    // start the data extraction
    
    // for all timesteps
    for(int t=time_step_lower;t<=time_step_upper;t++){
      if (debug) cerr << "Started timestep\n";
      
      double time = times[t];
      GridP grid = da->queryGrid(time);
      if(do_verbose)
	cout << "time = " << time << "\n";
      cerr << "Creating new timeblock.\n";
      timeblock = new Group();
      
      // for each level in the grid
      for(int l=0;l<grid->numLevels();l++){
	if (debug) cerr << "Started level\n";
	LevelP level = grid->getLevel(l);
	
	// for each patch in the level
	for(Level::const_patchIterator iter = level->patchesBegin();
	    iter != level->patchesEnd(); iter++){
	  const Patch* patch = *iter;
	  
	  //	  std::string var = vars[v];
	  ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	  
	  int mat_num;
	  if (mat == -1) {
	    mat_num = *(matls.begin());
	  } else {
	    int mat_index = 0;
	    for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
		 matlIter != matls.end(); matlIter++){
	      int matl = *matlIter;
	      if (matl == mat) {
		mat_num = matl;
		break;
	      }
	      mat_index++;
	    }
	    if (mat_index == matls.size()) {
	      // then we didn't find the right material
	      cerr << "Didn't find material " << mat << " in the data.\n";
	      mat_num = *(matls.begin());
	      cerr << "Using the material " << mat_num << ".\n";
	    }
	  }
	  const Uintah::TypeDescription* td = types[var_index];
	  const Uintah::TypeDescription* subtype = td->getSubType();
	  IntVector dim;
#ifdef USE_HVBRICK
	  float* data;
#else
	  rtrt::Array3<float> data;
#endif
	  double data_min, data_max;
	  switch(td->getType()){
	  case Uintah::TypeDescription::CCVariable:
	    switch(subtype->getType()){
	    case Uintah::TypeDescription::double_type:
	      {
		// get the data
		CCVariable<double> value;
		da->query(value, var, mat_num, patch, time);
		dim = IntVector(value.getHighIndex()-value.getLowIndex());
		if(dim.x() && dim.y() && dim.z()){
#ifdef USE_HVBRICK
		  data = new float[dim.x()*dim.y()*dim.z()];
#else
		  data.resize(dim.x(), dim.y(), dim.z());
		  float* data_ptr = data.get_dataptr();
		  int data_size = dim.z() * dim.y() * dim.x();
		  cerr << "dim=" << dim << '\n';
		  cerr << "data_size = " << data_size << endl;
#endif
		  CellIterator iter = patch->getCellIterator();
		  data_min = data_max = value[*iter];
		  int data_index = 0;
		  for(;!iter.done(); iter++){
		    //IntVector idx=*iter;
		    //cerr << "idx = " << idx << endl;
		    //		    int data_index=idx.z()+idx.y()*dim.z()+idx.x()*dim.z()*dim.y();
		    //int data_index=idx.x()+idx.y()*dim.x()+idx.z()*dim.x()*dim.y();
		    data_min = SCIRun::Min(data_min, value[*iter]);
		    data_max = SCIRun::Max(data_max, value[*iter]);
#ifdef USE_HVBRICK
		    data[data_index] = (float)value[*iter];
#else
		    data_ptr[data_index] = (float)value[*iter];
		    if (data_index >= data_size)
		      cerr << "Went over. data_index = " << data_index << endl;
#endif
		    data_index++;
		  }
		  // reorder the data
		  {
		    int size = dim.x()*dim.y()*dim.z();
		    float* temp_data = new float[size];
#ifdef USE_HVBRICK
		    float* data_ptr = data;
#else
		    float* data_ptr = data.get_dataptr();
#endif
		    data_index = 0;
		    int new_index;
		    for (int z = 0; z < dim.z(); z++)
		      for (int y = 0; y < dim.y(); y++)
			for (int x = 0; x < dim.x(); x++)
			  {
			    new_index = z + y*dim.z() + x * (dim.z()*dim.y());
			    temp_data[new_index] = data_ptr[data_index++];
			  }
		    // copy the data back
		    for (data_index = 0; data_index < size; data_index++)
		      data_ptr[data_index] = temp_data[data_index];
		    // clean up memory
		    delete[] temp_data;
#if 1
		    for (int x = 0; x < dim.x(); x++)
		      for (int y = 0; y < dim.y(); y++)
			for (int z = 0; z < dim.z(); z++)
			  {
			    //float val = (float) x + y + z;
			    float val = (float) x * y * z;
#ifdef USE_HVBRICK
			    data[z + y*dim.z() + x * (dim.z()*dim.y())] = val;
#else
			    data(x,y,z) = ;
#endif
			  }	
		    data_max = (dim.x()-1) * (dim.y()-1) * (dim.z()-1);
		    //data_max = dim.x() + dim.y() + dim.z() - 3;
		    data_min = 0;
#endif
		  }
		}
	      }
	      break;
	    case Uintah::TypeDescription::Point:
	      {
		// not implemented at this time
	      }
	      break;
	    case Uintah::TypeDescription::Vector:
	      {
		// not implemented at this time
	      }
	      break;
	    case Uintah::TypeDescription::Matrix3:
	      {
		// not implemented at this time
	      }
	      break;
	    default:
	      cerr << "NC variable of unknown type: " << subtype->getType() << '\n';
	      break;
	    } // end subtyp
	    break;
	  case Uintah::TypeDescription::NCVariable:
	    break;
	  case Uintah::TypeDescription::Matrix3:
	    break;
	  case Uintah::TypeDescription::ParticleVariable:
	  case Uintah::TypeDescription::ReductionVariable:
	  case Uintah::TypeDescription::Unknown:
	  case Uintah::TypeDescription::Other:
	    // currently not implemented
	    break;
	  default:
	    cerr << "Variable (" << var << ") is of unknown type: " << td->getType() << '\n';
	    break;
	  } // end switch(td->getType())
	  if (dim.x() && dim.y() && dim.z()) {
	    SCIRun::Point b_min = patch->getBox().lower();
	    SCIRun::Point b_max = patch->getBox().upper();
	    
	    rtrt::Point rtrt_b_min(b_min.x(),b_min.y(),b_min.z());
	    rtrt::Point rtrt_b_max(b_max.x(),b_max.y(),b_max.z());
	    cerr << "dim=" << dim << '\n';
	    cerr << "min=" << rtrt_b_min << ", max=" << rtrt_b_max << '\n';
#ifdef USE_HVBRICK
	    HVolumeBrick* hvol = new HVolumeBrick(matl0, dpy, depth,
						  nworkers, 
						  dim.x(), dim.y(), dim.z(),
						  rtrt_b_min, rtrt_b_max,
						  (float)data_min,
						  (float)data_max,
						  data);
#else
	    HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > >*
	      hvol =
	      new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > >
	      //HVolume<float, rtrt::Array3<float>, rtrt::Array3<VMCell<float> > >*
	      //hvol =
	      //new HVolume<float, rtrt::Array3<float>, rtrt::Array3<VMCell<float> > >
	      (matl0, dpy, depth,
	       nworkers, 
	       dim.x(), dim.y(), dim.z(),
	       rtrt_b_min, rtrt_b_max,
	       (float)data_min,
	       (float)data_max,
	       data);
#endif
	    timeblock->add(hvol);
	  } // end if (dim.x() && dim.y() && dim.z())
	  if (debug) cerr << "Finished processdata\n";
	} // end patch
	if (debug) cerr << "Finished patch\n";
      } // end level
      if (debug) cerr << "Finished level\n";
      cout << "Adding timestep.\n";
      timeobj1->add(timeblock);
    } // end timestep
    if (debug) cerr << "Finished timestep\n";
  } catch (SCIRun::Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }

#if 0
  // while there is stuff in the file
  while(in){
    char file[1000];
    // stick the next line in file
    in >> file;
    if (in) {
      if (strcmp(file,"<TIMESTEP>") == 0) {
	cerr << "-------------Starting timestep----------\n";
	timeblock = new Group();
      }
      else if (strcmp(file,"</TIMESTEP>") == 0) {
	cerr << "=============Ending timestep============\n";
	timeobj1->add(timeblock);
      }
      else if (strcmp(file,"<PATCH>") == 0) {
	//
      }
      else if (strcmp(file,"</PATCH>") == 0) {
	//
      }
      else {
	cerr << "Reading " << file << "\n";
	
	HVolumeBrick* hvol=new HVolumeBrick(matl0, dpy, file,
					    depth, nworkers);
	timeblock->add(hvol);
      }
    }
  }
#endif

  Group* group=new Group();
  group->add(timeobj1);
  (new Thread(dpy, "Volume GUI thread"))->detach();
  
  double bgscale=0.3;
  Color groundcolor(0,0,0);
  Color averagelight(1,1,1);
  double ambient_scale=.5;
  
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  
  rtrt::Plane groundplane ( rtrt::Point(0, 0, 0), rtrt::Vector(1, 0, 0) );

  Camera cam(rtrt::Point(5,0,0), rtrt::Point(0,0,0),
	     rtrt::Vector(0,1,0), 60);
  
  Scene* scene=new Scene(group, cam,
			 bgcolor, groundcolor*averagelight, bgcolor,
			 groundplane, ambient_scale);
  scene->add_light(new Light(rtrt::Point(5,-3,3), Color(1,1,.8)*2, 0));
  scene->set_background_ptr( new LinearBackground(Color(0.2, 0.4, 0.9),
						  Color(0.0,0.0,0.0),
						  rtrt::Vector(1, 0, 0)) );
  //scene->shadow_mode=0;
  scene->maxdepth=0;
  scene->attach_display(dpy);
  return scene;
}







