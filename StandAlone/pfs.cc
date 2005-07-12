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
parseArgs( int argc, char *argv[], string & infile, bool & binmode)
{
  binmode = false;
  infile = argv[argc-1];
  
  if( argc < 2) usage( argv[0] );
  
  if(string(argv[1])=="-b") {
    binmode = true;
    argc--;
  }
  if( argc > 2) usage( argv[0] );
}

void
usage( char *prog_name )
{
  cout << "Usage: " << prog_name << " [-b] [-B] infile \n";
  exit( 1 );
}

int
main(int argc, char *argv[])
{
  try {
    Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );
    Uintah::Parallel::initializeManager( argc, argv, "" );

    string infile;
    bool binmode;
    
    parseArgs( argc, argv, infile, binmode);
    
    ProblemSpecInterface* reader = scinew ProblemSpecReader(infile);
    
    // Get the problem specification
    ProblemSpecP ups = reader->readInputFile();
    ups->writeMessages(true);
    if(!ups)
      throw ProblemSetupException("Cannot read problem specification", __FILE__, __LINE__);
    
    if(ups->getNodeName() != "Uintah_specification")
      throw ProblemSetupException("Input file is not a Uintah specification", __FILE__, __LINE__);
    
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
        throw ProblemSetupException("pfs can only handle square grids", __FILE__, __LINE__);

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
                  throw ProblemSetupException("Unexpected field variable of '"+next_var_name+"'", __FILE__, __LINE__);
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
                
                // ADB: change this to always be 128 bytes, so that we can 
                // cleanly read the header off a binary file
                ofstream dest(of_name.c_str());
                if(binmode) {
                  double x0 = min.x(), y0 = min.y(), z0 = min.z();
                  double x1 = max.x(), y1 = max.y(), z1 = max.z();
                  dest.write((char*)&x0, sizeof(double));
                  dest.write((char*)&y0, sizeof(double));
                  dest.write((char*)&z0, sizeof(double));
                  dest.write((char*)&x1, sizeof(double));
                  dest.write((char*)&y1, sizeof(double));
                  dest.write((char*)&z1, sizeof(double));
                } else {
                  dest << min.x() << " " << min.y() << " " << min.z() << " " 
                       << max.x() << " " << max.y() << " " << max.z();
                }
                for (int I = 0; I < (int) points[pid].size(); I++) {
                  Point  p = points[pid][I].first;
                  vector<double> r = points[pid][I].second;
                  
                  // FIXME: should have way of specifying endiness
                  if(binmode) {
                    double x = p.x();
                    double y = p.y();
                    double z = p.z();
                    dest.write((char*)&x, sizeof(double));
                    dest.write((char*)&y, sizeof(double));
                    dest.write((char*)&z, sizeof(double));
                    for(vector<double>::const_iterator rit(r.begin());rit!=r.end();rit++) {
                      double v = *rit;
                      dest.write((char*)&v, sizeof(double));
                    }
                  } else {
                    dest << p.x() << " " << p.y() << " " << p.z();
                    for(vector<double>::const_iterator rit(r.begin());rit!=r.end();rit++)
                      dest << " " << *rit;
                    dest << endl;
                  }
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
