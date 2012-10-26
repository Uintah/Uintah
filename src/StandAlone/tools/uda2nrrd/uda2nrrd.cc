/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
 *  uda2nrrd.cc: Converts a Uintah Data Archive (UDA) to a nrrd.
 *
 *  Written by:
 *   Many people...?
 *   Department of Computer Science
 *   University of Utah
 *   April 2003-2007
 *
 */

#include <StandAlone/tools/uda2nrrd/wrap_nrrd.h>

#include <StandAlone/tools/uda2nrrd/Args.h>
#include <StandAlone/tools/uda2nrrd/bc.h>
#include <StandAlone/tools/uda2nrrd/handleVariable.h>
#include <StandAlone/tools/uda2nrrd/particles.h>
#include <StandAlone/tools/uda2nrrd/QueryInfo.h>

#include <Core/Math/Matrix3.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Math/MinMax.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>

#include <Core/OS/Dir.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Persistent/Pstreams.h>


#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Grid/Variables/LocallyComputedPatchVarMap.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/DataArchive/DataArchive.h>

#include <sci_hash_map.h>
#include <teem/nrrd.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

Args args;

void
usage( const string& badarg, const string& progname )
{
  if(badarg != "") {
    cerr << "Error parsing argument: " << badarg << "\n";
  }
  cerr << "Usage: " << progname << " [options] " << "-uda <archive file>\n";
  cerr << "\n";
  cerr << "  This program reads in an UDA data directory and produces a NRRD containing\n";
  cerr << "  the specified variable for each timestep in the UDA (unless -tlow, -thigh, \n";
  cerr << "  -tinc, or -tstep are used).\n";
  cerr << "\n";
  cerr << "Valid options are:\n";
  cerr << "  -h,--help  Prints this message out\n";
  cerr << "  -uda <archive file>\n";
  cerr << "\n";
  cerr << "Field Specifier Options\n";
  cerr << "  -v,--variable <variable name> - may not be used with -p\n";
  cerr << "  -p,--particledata - Pull out all the particle data into a single NRRD.  May not be used with -v\n";
  cerr << "             Particles only exist on a single level (but it can be any of the levels).  When -p\n";
  cerr << "             is specified, the code will automatically determine which level the particles live on,\n";
  cerr << "             therefore you should not specify -l or -a when you use -p.\n";
  cerr << "  -m,--material <material number> [defaults to first material found]\n";
  cerr << "  -l,--level <level index> [defaults to 0]\n";
  cerr << "  -a,--all - Use all levels.  Overrides -l.  Uses the resolution\n";
  cerr << "             of the finest level. Fills the entire domain by \n";
  cerr << "             interpolating data from lower resolution levels\n";
  cerr << "             when necessary.  May not be used with -p.\n";
  cerr << "             (-p attempts to find the one level that particles exist on and uses just that level.)\n";
  cerr << "  -mo <operator> type of operator to apply to matricies.\n";
  cerr << "                 Options are none, det, norm, and trace\n";
  cerr << "                 [defaults to none]\n";
  cerr << "  -nbc,--noboundarycells - remove boundary cells from output\n";
  cerr << "\n";
  cerr << "Output Options\n";
  cerr << "  -o,--out <outputfilename> [defaults to 'particles_t#######' or '<varName>_t######']\n";
  cerr << "  -oi <index> [default to 0] - Output index to use in naming file.\n";
  cerr << "  -dh,--detatched-header - writes the data with detached headers.  The default is to not do this.\n";
  cerr << "  -ow,--overwrite - overwrite existing output files without prompting.\n";
  //    cerr << "  -binary (prints out the data in binary)\n";
  
  cerr << "\nTimestep Specifier Optoins\n";
  cerr << "  -tlow,--timesteplow [int] (only outputs timestep from int) [defaults to 0]\n";
  cerr << "  -thigh,--timestephigh [int] (only outputs timesteps up to int) [defaults to last timestep]\n";
  cerr << "  -tinc [int] (output every n timesteps) [defaults to 1]\n";
  cerr << "  -tstep,--timestep [int] (only outputs timestep int)\n";
  cerr << "\n";
  cerr << "Chatty Options\n";
  cerr << "  -vv,--verbose (prints status of output)\n";
  cerr << "  -q,--quiet (very little output)\n";
  Thread::exitAll( 1 );
}

/////////////////////////////////////////////////////////////////////

int
main(int argc, char** argv)
{
  /*
   * Default values
   */
  bool do_binary=false;

  unsigned long time_step_lower = 0;
  // default to be last timestep, but can be set to 0
  unsigned long time_step_upper = (unsigned long)-1;
  unsigned long tinc = 1;

  string input_uda_name;
  string output_file_name("");
  int    output_file_index = 0; // Beginning index for modifying output file name.
  bool use_default_file_name = true;
  IntVector var_id(0,0,0);
  string variable_name("");
  // It will use the first material found unless other indicated.
  int material = -1;
  int level_index = 0;

  bool do_particles = false;

  bool minus_l_specified = false;
  
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-v" || s == "--variable") {
      if( do_particles ) {
        cout << "\n";
        cout << "Error: you may only use -v or -p, not both!\n";
        cout << "\n";
        usage( "", argv[0] );
      }
      variable_name = string(argv[++i]);
    } else if (s == "-p" || s == "--particledata") {
      if( args.use_all_levels || minus_l_specified ) {
        cout << "\n";
        cout << "Error: you may not use " << (args.use_all_levels ? "-a" : "-l") << " and -p at the same time!\n";
        cout << "\n";
        usage( "", argv[0] );
      }
      if( variable_name != "" ) {
        cout << "\n";
        cout << "Error: you may only use -v or -p, not both!\n";
        cout << "\n";
        usage( "", argv[0] );
      }
      do_particles = true;
    } else if (s == "-m" || s == "--material") {
      material = atoi(argv[++i]);
    } else if (s == "-l" || s == "--level") {

      if( args.use_all_levels || do_particles ) {
        cout << "\n";
        cout << "Error: you may not use both '";
        cout << (args.use_all_levels ? "-a" : "-p");
        cout << "' and '-l'!\n";
        cout << "\n";
        usage( "", argv[0] );
      }
      minus_l_specified = true;
      level_index = atoi(argv[++i]);
    } else if (s == "-a" || s == "--all"){
      args.use_all_levels = true;
      if( do_particles ) {
        cout << "\n";
        cout << "Error: you may not use -a and -p at the same time!\n";
        cout << "\n";
        usage( "", argv[0] );
      }
    } else if (s == "-vv" || s == "--verbose") {
      args.verbose = true;
    } else if (s == "-q" || s == "--quiet") {
      args.quiet = true;
    } else if (s == "-ow" || s == "--overwrite") {
      args.force_overwrite = true;
    } else if (s == "-tlow" || s == "--timesteplow") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-thigh" || s == "--timestephigh") {
      time_step_upper = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-tstep" || s == "--timestep") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
      time_step_upper = time_step_lower;
    } else if (s == "-tinc") {
      tinc = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-i" || s == "--index") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_id = IntVector(x,y,z);
    } else if( s ==  "-dh" || s == "--detatched-header") {
      args.attached_header = false;
    } else if( (s == "-h") || (s == "--help") ) {
      usage( "", argv[0] );
    } else if (s == "-uda") {
      input_uda_name = string(argv[++i]);
    } else if (s == "-oi" ) {
      output_file_index = atoi(argv[++i]);
    } else if (s == "-o" || s == "--out") {
      output_file_name = string(argv[++i]);
      use_default_file_name = false;
    } else if(s == "-mo") {
      s = argv[++i];
      if (s == "det")
        args.matrix_op = Det;
      else if (s == "norm")
        args.matrix_op = Norm;
      else if (s == "trace")
        args.matrix_op = Trace;
      else if (s == "none")
        args.matrix_op = None;
      else
        usage(s, argv[0]);
    } else if(s == "-binary") {
      do_binary=true;
    } else if(s == "-nbc" || s == "--noboundarycells") {
      args.remove_boundary = true;
    } else {
      usage(s, argv[0]);
    }
  }
  
  if(input_uda_name == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  // Verify that we can create a file in this directory

  string tmp_filename = output_file_name;
  if( tmp_filename == "" ) {
    tmp_filename = "temporary_uda2nrrd_testfile";
  }
  FILE * fp = fopen( tmp_filename.c_str(), "w" );
  if( fp == NULL ) {
    cout << "\n\n";
    cout << "Error: Couldn't create output file ('" << tmp_filename << "')... please check permissions.\n";
    cout << "\n\n";
    Thread::exitAll( 1 );
  }
  remove( tmp_filename.c_str() );
  fclose( fp );

  try {
    DataArchive* archive = scinew DataArchive(input_uda_name);

    ////////////////////////////////////////////////////////
    // Get the times and indices.

    vector<int> index;
    vector<double> times;
    
    // query time info from dataarchive
    archive->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    if( !args.quiet ) cout << "There are " << index.size() << " timesteps:\n";
    
    //////////////////////////////////////////////////////////
    // Get the variables and types
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;

    archive->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    if( args.verbose ) cout << "There are " << vars.size() << " variables:\n";
    bool var_found = false;
    vector<unsigned int> var_indices;

    if( do_particles ) {
      unsigned int vi = 0;
      for( ; vi < vars.size(); vi++ ) {
        if( vars[vi][0] == 'p' && vars[vi][1] == '.' ) { // starts with "p."
          // It is a particle variable
          var_indices.push_back( vi );
        }
      }
      if( var_indices.size() == 0 ) {
        cout << "\n";
        cout << "Error: No particle variables found (\"p.something\")...\n";
        cout << "\n";
        cout << "Variables known are:\n";
        vi = 0;
        for( ; vi < vars.size(); vi++) {
          cout << "vars[" << vi << "] = " << vars[vi] << "\n";
        }
        cout << "\nGoodbye!!\n\n";
        Thread::exitAll( 1 );
      }
    } 
    else { // Not particles...

      if( variable_name == "" ) {
        cerr << "\n";
        cerr << "A variable name must be specified!  Use '-v [name]'.  (Or -p for particles.)\n";
        cerr << "\n";
        Thread::exitAll( 1 );
      }

      unsigned int vi = 0;
      for( ; vi < vars.size(); vi++ ) {
        if( variable_name == vars[vi] ) {
          var_found = true;
          break;
        }
      }
      if (!var_found) {
        cerr << "Variable '" << variable_name << "' was not found.\n";
        cerr << "\n";
        cerr << "Possible variable names are:\n";
        cerr << "\n";
        vi = 0;
        for( ; vi < vars.size(); vi++) {
          cout << "vars[" << vi << "] = " << vars[vi] << " (" << types[vi]->getName() << ")\n";
        }
        cerr << "\nExiting!!\n\n";
        Thread::exitAll( 1 );
      }
      var_indices.push_back( vi );
    }

    if( use_default_file_name ) { // Then use the variable name for the output name

      if( do_particles ) {
        output_file_name = "particles";
      } else {
        output_file_name = variable_name;
      }
      if( !args.quiet ) {
        cout << "Using variable name (" << output_file_name
             << ") as output file base name.\n";
      }
    }
    
    /////////////////////////////////////////////////////
    // figure out the lower and upper bounds on the timesteps
    if (time_step_lower >= times.size()) {
      cerr << "ERROR: timesteplow must be between 0 and " << times.size()-1 << ". Goodbye.\n";
      Thread::exitAll( 1 );
    }
      
    // set default max time value
    if (time_step_upper == (unsigned long)-1) {
      if( args.verbose )
        cout <<"Initializing time_step_upper to "<<times.size()-1<<"\n";
      time_step_upper = times.size() - 1;
    }
    
    if (time_step_upper >= times.size() || time_step_upper < time_step_lower) {
      cerr << "ERRIR: timestephigh("<<time_step_lower<<") must be greater than " 
           << time_step_lower << " and less than " << times.size()-1 << ". Goodbye.\n";
      Thread::exitAll( 1 );
    }
      
    if( !args.quiet ) { 
      if( time_step_lower != time_step_upper ) {
        cout << "Extracting data from timesteps " << time_step_lower << " to " << time_step_upper << ".  "
             << "Times: " << times[time_step_lower] << " to " << times[time_step_upper]
             << "\n";
      } else {
        cout << "Extracting data from timestep " << time_step_lower << " (time: " << times[time_step_lower] << ").\n";
      }
    }

    ////////////////////////////////////////////////////////
    // Loop over each timestep
    for( unsigned long time = time_step_lower; time <= time_step_upper; time += tinc ) {
        
      /////////////////////////////
      // Figure out the filename

      char filename_num[200];
      sprintf( filename_num, "_t%06d", index[time] );

      string filename( output_file_name + filename_num );
      
      // Check the level index
      double current_time = times[time];
      GridP grid = archive->queryGrid(time);
      if (level_index >= grid->numLevels() || level_index < 0) {
        cerr << "level index (" << level_index << ") is bad.  Should be from 0 to " 
	     << grid->numLevels()-1 << ".\n";
        cerr << "Trying next timestep.\n";
        continue;
      }
    
      vector<ParticleDataContainer> particleDataArray;

      LevelP level;

      // Loop over the specified variable(s)...
      //
      // ... Currently you can only specify one grid var, or all particles vars.
      // ... This loop is used to run over the one grid var, or over all the particle vars...
      // ... However, it should be easy to allow the user to create multiple grid var
      // ... NRRDs at the same time using this loop...
      
      for( unsigned int cnt = 0; cnt < var_indices.size(); cnt++ ) {

        unsigned int var_index = var_indices[cnt];
        variable_name = vars[var_index];
        
        if( !args.quiet ) {
          cout << "Extracting data for " << vars[var_index] << ": " << types[var_index]->getName() << "\n";
        }

        //////////////////////////////////////////////////
        // Set the level pointer

        if( level.get_rep() == NULL ) {  // Only need to get the level for the first timestep... as
                                         // the data will be on the same level(s) for all timesteps.
          if( do_particles ) { // Determine which level the particles are on...
            bool found_particle_level = false;
            for( int lev = 0; lev < grid->numLevels(); lev++ ) {
              LevelP particleLevel = grid->getLevel( lev );
              const Patch* patch = *(particleLevel->patchesBegin());
              ConsecutiveRangeSet matls = archive->queryMaterials(variable_name, patch, time);
              if( matls.size() > 0 ) {
                if( found_particle_level ) {
                  // Ut oh... found particles on more than one level... don't know how 
                  // to handle this yet...
                  cout << "\n";
                  cout << "Error: uda2nrrd currently can only handle particles on only a single level.  Goodbye.\n";
                  cout << "\n";
                  Thread::exitAll( 1 );
                }
                // The particles are on this level...
                found_particle_level = true;
                level = particleLevel;
                cout << "Found the PARTICLES on level " << lev << ".\n";
              }
            }
          }
          else {
            if( args.use_all_levels ){ // set to level zero
              level = grid->getLevel( 0 );
              if( grid->numLevels() == 1 ){ // only one level to use
                args.use_all_levels = false;
              }
            } else {  // set to requested level
              level = grid->getLevel(level_index);
            }
          }
        }
          
        ///////////////////////////////////////////////////
        // Check the material number.
        
        const Patch* patch = *(level->patchesBegin());
        ConsecutiveRangeSet matls = archive->queryMaterials(variable_name, patch, time);
        
        if( args.verbose ) {
          // Print out all the material indicies valid for this timestep
          cout << "Valid materials for " << variable_name << " at time[" << time << "](" << current_time << ") are:  ";
          for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
               matlIter != matls.end(); matlIter++) {
            cout << *matlIter << ", ";
          }
          cout << "\n";
        }
      
        ConsecutiveRangeSet  materialsOfInterest;

        if( do_particles ) {
          materialsOfInterest = matls;
        } else {
          if (material == -1) {
            materialsOfInterest.addInOrder( *(matls.begin()) ); // Default: only interested in first material.
          } else {
            unsigned int mat_index = 0;

            ConsecutiveRangeSet::iterator matlIter = matls.begin();

            for( ; matlIter != matls.end(); matlIter++ ){
              int matl = *matlIter;
              if (matl == material) {
                materialsOfInterest.addInOrder( matl );
                break;
              }
              mat_index++;
            }
            if( mat_index == matls.size() ) { // We didn't find the right material...
              cerr << "Didn't find material " << material << " in the data.\n";
              cerr << "Trying next timestep.\n";
              continue;
            }
          }
        }

        // get type and subtype of data
        const Uintah::TypeDescription* td = types[var_index];
        const Uintah::TypeDescription* subtype = td->getSubType();
          
        QueryInfo qinfo( archive, grid, level, variable_name, materialsOfInterest,
                         time, current_time, args.use_all_levels, td );

        IntVector hi, low, range;
        BBox box;

        // Remove the edges if no boundary cells
        if( args.remove_boundary ){
          level->findInteriorIndexRange(low, hi);
          level->getInteriorSpatialRange(box);
        } else {
          level->findIndexRange(low, hi);
          level->getSpatialRange(box);
        }
        range = hi - low;

        if (qinfo.type->getType() == Uintah::TypeDescription::CCVariable) {
          IntVector cellLo, cellHi;
          if( args.remove_boundary ) {
            level->findInteriorCellIndexRange(cellLo, cellHi);
          } else {
            level->findCellIndexRange(cellLo, cellHi);
          }
          if (is_periodic_bcs(cellHi, hi)) {
            IntVector newrange(0,0,0);
            get_periodic_bcs_range( cellHi, hi, range, newrange);
            range = newrange;
          }
        }
          
        // Adjust the range for using all levels
        if( args.use_all_levels && grid->numLevels() > 0 ){
          double exponent = grid->numLevels() - 1;
          range.x( range.x() * int(pow(2, exponent)));
          range.y( range.y() * int(pow(2, exponent)));
          range.z( range.z() * int(pow(2, exponent)));
          low.x( low.x() * int(pow(2, exponent)));
          low.y( low.y() * int(pow(2, exponent)));
          low.z( low.z() * int(pow(2, exponent)));
          hi.x( hi.x() * int(pow(2, exponent)));
          hi.y( hi.y() * int(pow(2, exponent)));
          hi.z( hi.z() * int(pow(2, exponent)));
            
          if( args.verbose ){
            cout << "The entire domain for all levels will have an index range of "
                 << low << " to " << hi
                 << " and a spatial range from " << box.min() << " to "
                 << box.max() << ".\n";
          }
        }

        ///////////////////
        // Get the data...
    
        if( td->getType() == Uintah::TypeDescription::ParticleVariable ) {  // Handle Particles

          if( !do_particles ) {
            cout << "\n\n";
            cout << "ERROR: extracting particle information, but you didn't specify particles\n"
                 << "       with '-p' on the command line... please start over and use -p.\n";
            cout << "\n\n";
            Thread::exitAll( 1 );
          }

          ParticleDataContainer data;

          switch (subtype->getType()) {
          case Uintah::TypeDescription::double_type:
            data = handleParticleData<double>( qinfo );
            break;
          case Uintah::TypeDescription::float_type:
            data = handleParticleData<float>( qinfo );
            break;
          case Uintah::TypeDescription::int_type:
            data = handleParticleData<int>( qinfo );
            break;
          case Uintah::TypeDescription::long64_type:
            data = handleParticleData<long64>( qinfo );
            break;
          case Uintah::TypeDescription::Point:
            data = handleParticleData<Point>( qinfo );
            break;
          case Uintah::TypeDescription::Vector:
            data = handleParticleData<Vector>( qinfo );
            break;
          case Uintah::TypeDescription::Matrix3:
            data = handleParticleData<Matrix3>( qinfo );
            break;
          default:
            cerr << "Unknown subtype for particle data: " << subtype->getName() << "\n";
            Thread::exitAll( 1 );
          } // end switch( subtype )
            
          particleDataArray.push_back( data );
            
        } else { // Handle Grid Variables
            
          switch (subtype->getType()) {
          case Uintah::TypeDescription::double_type:
            handleVariable<double>( qinfo, low, hi, range, box, filename, args );
            break;
          case Uintah::TypeDescription::float_type:
            handleVariable<float>(qinfo, low, hi, range, box, filename, args );
            break;
          case Uintah::TypeDescription::int_type:
            handleVariable<int>(qinfo, low, hi, range, box, filename, args );
            break;
          case Uintah::TypeDescription::Vector:
            handleVariable<Vector>(qinfo, low, hi, range, box, filename, args );
            break;
          case Uintah::TypeDescription::Matrix3:
            handleVariable<Matrix3>(qinfo, low, hi, range, box, filename, args );
            break;
          case Uintah::TypeDescription::bool_type:
          case Uintah::TypeDescription::short_int_type:
          case Uintah::TypeDescription::long_type:
          case Uintah::TypeDescription::long64_type:
            cerr << "Subtype " << subtype->getName() << " is not implemented...\n";
            Thread::exitAll( 1 );
            break;
          default:
            cerr << "Unknown subtype\n";
            Thread::exitAll( 1 );
          }
        }
      } // end variables loop

      if( do_particles ) {
        saveParticleData( particleDataArray, filename, current_time );
      }

    } // end time step loop
    
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << "\n";
    Thread::exitAll( 1 );
  } catch(...){
    cerr << "Caught unknown exception\n";
    Thread::exitAll( 1 );
  }
}
