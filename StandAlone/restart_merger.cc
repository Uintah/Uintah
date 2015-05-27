/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
 *  restart_merger.cc
 *
 *  Merges uda directories where one has been restarted from another
 *  with the "-nocopy" option (which is default).  As a result, it will
 *  create a uda that would be just as if the "-copy" option were used
 *  (however, there are file system reasons why it is better to use this
 *  restart_merger afterwards than to use "-copy" in the first place).
 *
 *  Written by:
 *   WayneWitzel
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 */

#include <Core/OS/Dir.h>
#include <Core/Util/Environment.h>

#include <CCA/Components/DataArchiver/DataArchiver.h>
#include <CCA/Components/Parent/ComponentFactory.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Components/SimulationController/AMRSimulationController.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>


void
usage( const std::string & badarg, const std::string & progname )
{
  std::cerr << "\n";
  if(badarg != "") {
    std::cerr << "Error parsing argument: " << badarg << '\n';
  }
  std::cerr << "Usage: " << progname << " [options] <uda dir 1> <uda dir 2> [<uda dir 3> ...]\n";
  std::cerr << "    There can be any number of udas on the command line.\n";
  std::cerr << "\n";
  std::cerr << "    The " << progname << " program is used to merge N UDAs together into a single UDA.\n";
  std::cerr << "    The N merged UDAs must all be restarts of a common UDA.  This tool is most commonly\n";
  std::cerr << "    used in order to create a single UDA for visualization purposes.\n";
  std::cerr << "\n";
  std::cerr << "Options:\n";
  std::cerr << "\t-copy\t(Default) Copies timestep directories into the new uda directory\n"
       << "\t\twithout affecting the source uda directories.\n";
  std::cerr << "\t-move\tMoves timestep directories from the source udas directories into\n"
       << "\t\tthe new uda directory and removes the source udas.\n"
       << "\t\tThis option can be faster if the source and destination are\n"
       << "\t\ton the same file system, but there may be data loss where the\n"
       << "\t\tudas overlap.\n";
  std::cerr << "\n";
  std::cerr << "\t\t\t-move or -copy must be the first argument if specified.\n";
  std::cerr << "\n";
  std::cerr << "Assuming <uda dir n> was created by restarting from <uda dir n-1>\n";
  std::cerr << "with the -nocopy option (which is the default), this will create a\n";
  std::cerr << "new uda directory that is a continuous version of these uda directories.\n"; 
  std::cerr << "\n";
  exit(1);
}

int
main( int argc, char *argv[], char *env[] )
{
  // Pass the env into the sci env so it can be used there...
  SCIRun::create_sci_environment( env, 0, true );

  bool move = false;
  int i = 1;
  for (i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "-copy") {
      move = false; // default anyway
    }
    else if (s == "-move") {
      move = true;
    }
    else
      break;
  }

  int n_udafiles = argc - i;
  if (n_udafiles < 2) {
    usage("", argv[0]);
  }
  
  char** udafile = &argv[i];
  std::string ups_filename = std::string(udafile[0]) + "/input.xml";

  bool thrownException = false;
  
  Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );

  std::string new_uda_dir;
  try {
    Uintah::ProblemSpecP ups = Uintah::ProblemSpecReader().readInputFile( ups_filename );
    Uintah::Parallel::initializeManager(argc, argv);
    const Uintah::ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();

    Uintah::UintahParallelComponent * comp = Uintah::ComponentFactory::create( ups, world, false, udafile[0] );
    Uintah::SimulationInterface     * sim  = dynamic_cast<Uintah::SimulationInterface*>( comp );

    Uintah::DataArchiver out_uda(world);
    out_uda.attachPort("sim", sim);
    out_uda.problemSetup(ups, NULL);
    out_uda.initializeOutput(ups);
    new_uda_dir = out_uda.getOutputLocation();

    int timestep = 0;
    int prevTimestep = 0;
    int i;
    for (i = 0; i < n_udafiles-1; i++) { // each file except last
      Uintah::DataArchive in_uda(udafile[i+1]);
      int old_timestep = timestep;
      if (!in_uda.queryRestartTimestep(timestep)) {
	std::cerr << std::endl << udafile[i+1] << " is not a restarted uda -- no \"restart\" tag found.\n";
	exit(1);
      }

      if (old_timestep > timestep) {
	std::cerr << std::endl << udafile[i] << " does not preceed " << udafile[i+1] << " with respect to restart timestep order.\n\n";
	usage("", argv[0]);
      }
      
      Uintah::Dir restartFromDir(udafile[i]);
      // the time argument doesn't matter.  Pass in 0 also to signify to not copy checkpoints
      out_uda.restartSetup(restartFromDir, prevTimestep, timestep, 0 /* this time doesn't matter for our purpose here */, false, move);
      prevTimestep = timestep;
    }
   
    // copy all of the last uda timesteps 
    SCIRun::Dir restartFromDir(udafile[i]);
    out_uda.copySection(restartFromDir, "globals");
    out_uda.copySection(restartFromDir, "variables");
    // pass in an arbitrary '1' for time, so it can copy the checkpoints.
    out_uda.restartSetup(restartFromDir, prevTimestep, -1, 1, false, move);

    //ups->releaseDocument();
  } catch (SCIRun::Exception& e) {
    std::cerr << "Caught exception: " << e.message() << '\n';
    if(e.stackTrace())
      std::cerr << "Stack trace: " << e.stackTrace() << '\n';
    // Dd: I believe that these cause error messages
    // to be lost when the program dies...
    //Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
    //abort();
    thrownException = true;
  } catch (std::exception e){
    std::cerr << "Caught std exception: " << e.what() << '\n';
    //Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
    //abort();
    thrownException = true;
  }
  /*
  catch(...){
    std::cerr << "Caught unknown exception\n";
    //Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
    //abort();
    thrownException = true;
  }
  */
  
  if( thrownException ) {
    usage( "", argv[0] );
  }

  std::cout << "Successfully created " << new_uda_dir << "\n";
  return 0;
}
