#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/FileGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
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

    grid->problemSetup(ups, world, false);  
    
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
            
            // Read in points from a file
            if (go_type == "file"){
              string f_name,of_name;
              child->require("name",f_name);
              ifstream source(f_name.c_str());
              int ncols = 0;
              // count number of vars, and their sizes
              for(ProblemSpecP varblock = child->findBlock("var");
                  varblock;varblock=varblock->findNextBlock("var")) {
                string next_var_name("EMPTY!");
                varblock->get(next_var_name);
                if      (next_var_name=="p.volume")        ncols += 1;
                else if (next_var_name=="p.temperature")   ncols += 1;
                else if (next_var_name=="p.externalforce") ncols += 3;
                else if (next_var_name=="p.fiberdir")      ncols += 3;
                else 
                  throw ProblemSetupException("Unexpected field variable of '"+next_var_name+"'");
              }
              
              vector<vector<pair<Point,vector<double> > > > points(level->numPatches());
              Point min(1e30,1e30,1e30),max(-1e30,-1e30,-1e30);
              
              double x,y,z;
              while (source >> x >> y >> z) {
                vector<double> cols;
                for(int ic=0;ic<ncols;ic++)
                  {
                    double v;
                    source >> v;
                    cols.push_back(v);
                  }
                
                Point pp(x,y,z);
                if(level->containsPoint(pp)){
                  const Patch* currentpatch =
                    level->selectPatchForCellIndex(level->getCellIndex(pp));
                  int pid = currentpatch->getID();
                  min = Min(pp,min);
                  max = Max(pp,max);
                  points[pid].push_back(pair<Point,vector<double> >(pp,cols));
                }
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
                  Point  p = points[pid][I].first;
                  vector<double> r = points[pid][I].second;
                  dest << p.x() << " " << p.y() << " " << p.z();
                  for(vector<double>::const_iterator rit(r.begin());rit!=r.end();rit++)
                    dest << " " << *rit;
                  dest << endl;
                }
                dest.close();
              }
            }
          }
        }
      }
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
