#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/FileGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Math/Primes.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>

#include <xercesc/dom/DOMException.hpp> 


#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <sgi_stl_warnings_on.h>

// TODO:
// Either clean this up a bit, or merge the functionality into slb

using namespace Uintah;
using namespace std;

void usage( char *prog_name );

void
parseArgs( int argc, char *argv[], string & infile)
{
  if( argc < 2 || argc > 2 ) {
    usage( argv[0] );
  }

  infile = argv[1];
}

void
usage( char *prog_name )
{
  cout << "Usage: " << prog_name << " infile \n";
  exit( 1 );
}

int
main(int argc, char *argv[])
{
  try {
    Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );
    Uintah::Parallel::initializeManager( argc, argv, "" );

    string infile;
    
    parseArgs( argc, argv, infile);
    
    ProblemSpecInterface* reader = scinew ProblemSpecReader(infile);
    
    // Get the problem specification
    ProblemSpecP ups = reader->readInputFile();
    ups->writeMessages(true);
    if(!ups)
      throw ProblemSetupException("Cannot read problem specification");
    
    if(ups->getNodeName() != "Uintah_specification")
      throw ProblemSetupException("Input file is not a Uintah specification");
    
    const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
    
    // Setup the initial grid
    GridP grid=scinew Grid();
    IntVector extraCells(0,0,0);

    // save and remove the extra cells before the problem setup
    ProblemSpecP g = ups->findBlock("Grid");
    for( ProblemSpecP levelspec = g->findBlock("Level"); levelspec != 0;
         levelspec = levelspec->findNextBlock("Level")) {
      for (ProblemSpecP box = levelspec->findBlock("Box"); box != 0 ; 
           box = box->findNextBlock("Box")) {
        
        ProblemSpecP cells = box->findBlock("extraCells");
        if (cells != 0) {
          box->get("extraCells", extraCells);
          box->removeChild(cells);
        }
      }
    }

    grid->problemSetup(ups, world);  
    
    for (int l = 0; l < grid->numLevels(); l++) {
      LevelP level = grid->getLevel(l);
      
      IntVector low, high;
      level->findCellIndexRange(low, high);
      IntVector diff = high-low;
      long cells = diff.x()*diff.y()*diff.z();
      if(cells != level->totalCells())
        throw ProblemSetupException("pfs can only handle square grids");

      // Parse the geometry from the UPS
      ProblemSpecP mp = ups->findBlock("MaterialProperties");
      ProblemSpecP mpm = mp->findBlock("MPM");
      for (ProblemSpecP child = mpm->findBlock("material"); child != 0;
           child = child->findNextBlock("material")) {
        for (ProblemSpecP geom_obj_ps = child->findBlock("geom_object");
             geom_obj_ps != 0;
             geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

             for(ProblemSpecP child = geom_obj_ps->findBlock(); child != 0;
                              child = child->findNextBlock()){
               std::string go_type = child->getNodeName();
               if (go_type == "file"){
                 string f_name,of_name;
                 child->require("name",f_name);
                 ifstream source(f_name.c_str());
                 vector<vector<Point> > points(level->numPatches());
                 double x,y,z;
                 Point min(1e30,1e30,1e30),max(-1e30,-1e30,-1e30);
                 while (source >> x >> y >> z) {
                   Patch* currentpatch = level->getPatchFromPoint(Point(x,y,z));
                   int pid = currentpatch->getID();
                   Point pp(x,y,z);
                   min = Min(pp,min);
                   max = Max(pp,max);
                   points[pid].push_back(pp);
                 }
                 source.close();
                 for(Level::const_patchIterator iter = level->patchesBegin();
                                     iter != level->patchesEnd(); iter++){
                    const Patch* patch = *iter;
                    int pid = patch->getID();

                    char fnum[5];
                    sprintf(fnum,".%d",pid);
                    of_name = f_name+fnum;
                    ofstream dest(of_name.c_str());
                    dest << min.x() << " " << min.y() << " " << min.z() << " " 
                         << max.x() << " " << max.y() << " " << max.z() << endl;
                    for (int I = 0; I < (int) points[pid].size(); I++) {
                      dest << points[pid][I].x() << " " <<
                              points[pid][I].y() << " " <<
                              points[pid][I].z() << endl;
                    }
                    dest.close();
                 }
               }
             }
        }
      }

      
      // remove the 'Box' entry from the ups - note this should try to
      // remove *all* boxes from the level node
      ProblemSpecP lev = g->findBlock("Level");
      ProblemSpecP box = lev->findBlock("Box");
      
      lev->removeChild(box);
    }
    
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
    if(e.stackTrace())
      cerr << "Stack trace: " << e.stackTrace() << '\n';
  } catch (DOMException& e){
    cerr << "Caught Xerces DOM exception, code: " << e.code << '\n';
  } catch(...){
    cerr << "Caught unknown exception\n";
  }
}
