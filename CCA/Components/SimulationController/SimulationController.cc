#include <sci_defs/malloc_defs.h>

#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Time.h>

#include <sys/param.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#define SECONDS_PER_MINUTE 60.0
#define SECONDS_PER_HOUR   3600.0
#define SECONDS_PER_DAY    86400.0
#define SECONDS_PER_WEEK   604800.0
#define SECONDS_PER_YEAR   31536000.0

using namespace SCIRun;
using namespace std;

static DebugStream dbg("SimulationStats", true);
static DebugStream dbgTime("SimulationTimeStats", false);

namespace Uintah {

  // for calculating memory usage when sci-malloc is disabled.
  char* SimulationController::start_addr = NULL;

  double stdDeviation(double sum_of_x, double sum_of_x_squares, int n)
  {
    return sqrt((n*sum_of_x_squares - sum_of_x*sum_of_x)/(n*n));
  }

  SimulationController::SimulationController(const ProcessorGroup* myworld)
    : UintahParallelComponent(myworld)
  {
    d_n = 0;
    d_wallTime = 0;
    d_startTime = 0;
    d_prevWallTime = 0;
    d_sumOfWallTimes = 0;
    d_sumOfWallTimeSquares = 0;
  }

  SimulationController::~SimulationController()
  {
  }

  void SimulationController::doCombinePatches(std::string /*fromDir*/)
  {
    throw InternalError("Patch combining not implement for this simulation controller");
  }

  double SimulationController::getWallTime  ( void )
  {
    return d_wallTime;
  }

  void SimulationController::calcWallTime ( void )
  {
    d_wallTime = Time::currentSeconds() - d_startTime;
  }

  double SimulationController::getStartTime ( void )
  {
    return d_startTime;
  }

  void SimulationController::calcStartTime ( void )
  {
    d_startTime = Time::currentSeconds();
  }

  void SimulationController::initSimulationStatsVars ( void )
  {
    // vars used to calculate standard deviation
    d_n = 0;
    d_wallTime = 0;
    d_prevWallTime = Time::currentSeconds();
    d_sumOfWallTimes = 0; // sum of all walltimes
    d_sumOfWallTimeSquares = 0; // sum all squares of walltimes
  }

  void SimulationController::printSimulationStats ( SimulationStateP sharedState, double delt, double time )
  {
    // get memory stats for output
#ifndef DISABLE_SCI_MALLOC
    size_t nalloc,  sizealloc, nfree,  sizefree, nfillbin,
      nmmap, sizemmap, nmunmap, sizemunmap, highwater_alloc,  
      highwater_mmap;
      
    GetGlobalStats(DefaultAllocator(),
		   nalloc, sizealloc, nfree, sizefree,
		   nfillbin, nmmap, sizemmap, nmunmap,
		   sizemunmap, highwater_alloc, highwater_mmap);
    unsigned long memuse = sizealloc - sizefree;
    unsigned long highwater = highwater_mmap;
#else
    unsigned long memuse = 0;
    if ( ProcessInfo::IsSupported( ProcessInfo::MEM_RSS ) ) {
      memuse = ProcessInfo::GetMemoryResident();
    } else {
      memuse = (char*)sbrk(0)-start_addr;
    }
    unsigned long highwater = 0;
#endif
    
    // get memory stats for each proc if MALLOC_PERPROC is in the environent
    if ( getenv( "MALLOC_PERPROC" ) ) {
      ostream* mallocPerProcStream = NULL;
      char* filenamePrefix = getenv( "MALLOC_PERPROC" );
      if ( !filenamePrefix || strlen( filenamePrefix ) == 0 ) {
	mallocPerProcStream = &dbg;
      } else {
	char filename[MAXPATHLEN];
	sprintf( filename, "%s.%d" ,filenamePrefix, d_myworld->myrank() );
	if ( sharedState->getCurrentTopLevelTimeStep() == 0 ) {
	  mallocPerProcStream = new ofstream( filename, ios::out | ios::trunc );
	} else {
	  mallocPerProcStream = new ofstream( filename, ios::out | ios::app );
	}
	if ( !mallocPerProcStream ) {
	  delete mallocPerProcStream;
	  mallocPerProcStream = &dbg;
	}
      }
      *mallocPerProcStream << "Proc "     << d_myworld->myrank() << "   ";
      *mallocPerProcStream << "Timestep " << sharedState->getCurrentTopLevelTimeStep() << "   ";
      *mallocPerProcStream << "Size "     << ProcessInfo::GetMemoryUsed() << "   ";
      *mallocPerProcStream << "RSS "      << ProcessInfo::GetMemoryResident() << "   ";
      *mallocPerProcStream << "Sbrk "     << (char*)sbrk(0) - start_addr << "   ";
#ifndef DISABLE_SCI_MALLOC
      *mallocPerProcStream << "Sci_Malloc_Memuse "    << memuse << "   ";
      *mallocPerProcStream << "Sci_Malloc_Highwater " << highwater;
#endif
      *mallocPerProcStream << endl;
      if ( mallocPerProcStream != &dbg ) {
	delete mallocPerProcStream;
      }
    }
    
    unsigned long avg_memuse = memuse;
    unsigned long max_memuse = memuse;
    unsigned long avg_highwater = highwater;
    unsigned long max_highwater = highwater;
    if (d_myworld->size() > 1) {
      MPI_Reduce(&memuse, &avg_memuse, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
		 d_myworld->getComm());
      if(highwater){
	MPI_Reduce(&highwater, &avg_highwater, 1, MPI_UNSIGNED_LONG,
		   MPI_SUM, 0, d_myworld->getComm());
      }
      avg_memuse /= d_myworld->size(); // only to be used by processor 0
      avg_highwater /= d_myworld->size();
      MPI_Reduce(&memuse, &max_memuse, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0,
		 d_myworld->getComm());
      if(highwater){
	MPI_Reduce(&highwater, &max_highwater, 1, MPI_UNSIGNED_LONG,
		   MPI_MAX, 0, d_myworld->getComm());
      }
    }
    
    // calculate mean/std dev
    double stdDev = 0;
    double mean = 0;
    if (d_n > 2) { // ignore times 0,1,2
      //walltimes.push_back(d_wallTime - d_prevWallTime);
      d_sumOfWallTimes += (d_wallTime - d_prevWallTime);
      d_sumOfWallTimeSquares += pow(d_wallTime - d_prevWallTime,2);
    }
    if (d_n > 3) {
      // divide by n-2 and not n, because we wait till n>2 to keep track
      // of our stats
      stdDev = stdDeviation(d_sumOfWallTimes, d_sumOfWallTimeSquares, d_n-2);
      mean = d_sumOfWallTimes / (d_n-2);
      //         ofstream timefile("avg_elapsed_wallTime.txt");
      //         timefile << mean << " +- " << stdDev << endl;
    }

    // output timestep statistics
    if(d_myworld->myrank() == 0){
      dbg << "Time="         << time
	  << " (timestep "  << sharedState->getCurrentTopLevelTimeStep() 
	   << "), delT="     << delt
	   << ", elap T = "  << d_wallTime;
      
      if (d_n > 3) {
	dbg << ", mean: "   << mean
	    << " +- "       << stdDev;
      }
      dbg << ", Mem Use = ";
      if (avg_memuse == max_memuse && avg_highwater == max_highwater) {
	dbg << avg_memuse;
	if(avg_highwater) {
	  dbg << "/" << avg_highwater;
	}
      } else {
	dbg << avg_memuse;
	if(avg_highwater) {
	  dbg << "/" << avg_highwater;
	}
	dbg << " (avg), " << max_memuse;
	if(max_highwater) {
	  dbg << "/" << max_highwater;
	}
	dbg << " (max)";
      }
      dbg << endl;

      if ( d_n > 0 ) {
	double realSecondsNow = (d_wallTime - d_prevWallTime)/delt;
	double realSecondsAvg = d_wallTime/time;

	dbgTime << "1 sim second takes ";

	dbgTime << left << showpoint << setprecision(3) << setw(4);

	if (realSecondsNow < SECONDS_PER_MINUTE) {
	  dbgTime << realSecondsNow << " seconds (now), ";
	} else if ( realSecondsNow < SECONDS_PER_HOUR ) {
	  dbgTime << realSecondsNow/SECONDS_PER_MINUTE << " minutes (now), ";
	} else if ( realSecondsNow < SECONDS_PER_DAY  ) {
	  dbgTime << realSecondsNow/SECONDS_PER_HOUR << " hours (now), ";
	} else if ( realSecondsNow < SECONDS_PER_WEEK ) {
	  dbgTime << realSecondsNow/SECONDS_PER_DAY << " days (now), ";
	} else if ( realSecondsNow < SECONDS_PER_YEAR ) {
	  dbgTime << realSecondsNow/SECONDS_PER_WEEK << " weeks (now), ";
	} else {
	  dbgTime << realSecondsNow/SECONDS_PER_YEAR << " years (now), ";
	}

	dbgTime << setw(4);

	if (realSecondsAvg < SECONDS_PER_MINUTE) {
	  dbgTime << realSecondsAvg << " seconds (avg) ";
	} else if ( realSecondsAvg < SECONDS_PER_HOUR ) {
	  dbgTime << realSecondsAvg/SECONDS_PER_MINUTE << " minutes (avg) ";
	} else if ( realSecondsAvg < SECONDS_PER_DAY  ) {
	  dbgTime << realSecondsAvg/SECONDS_PER_HOUR << " hours (avg) ";
	} else if ( realSecondsAvg < SECONDS_PER_WEEK ) {
	  dbgTime << realSecondsAvg/SECONDS_PER_DAY << " days (avg) ";
	} else if ( realSecondsAvg < SECONDS_PER_YEAR ) {
	  dbgTime << realSecondsAvg/SECONDS_PER_WEEK << " weeks (avg) ";
	} else {
	  dbgTime << realSecondsAvg/SECONDS_PER_YEAR << " years (avg) ";
	}

	dbgTime << "to calculate." << endl;
      }
 
      d_prevWallTime = d_wallTime;
      d_n++;
    }
  }
  
} // namespace Uintah {
