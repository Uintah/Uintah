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
            bool done = false;

            // Read in points from a file
            if (go_type == "file"){
              string f_name,of_name;
              string var_name1="0",var_name2="0",var_name3="0";
              child->require("name",f_name);
              child->get("var1",var_name1);
              child->get("var2",var_name2);
              child->get("var3",var_name3);
              ifstream source(f_name.c_str());
              vector<vector<Point> > points(level->numPatches());
              double x,y,z;
              Point min(1e30,1e30,1e30),max(-1e30,-1e30,-1e30);
              // Points only
              if(var_name1=="0" && var_name2 == "0" && var_name3 == "0"){
                done = true;
                while (source >> x >> y >> z) {
                  Point pp(x,y,z);
                  if(level->containsPoint(pp)){
                    const Patch* currentpatch =
                       level->selectPatchForCellIndex(level->getCellIndex(pp));
                    int pid = currentpatch->getID();
                    min = Min(pp,min);
                    max = Max(pp,max);
                    points[pid].push_back(pp);
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
                     dest << points[pid][I].x() << " " <<
                             points[pid][I].y() << " " <<
                             points[pid][I].z() << endl;
                   }
                   dest.close();
                }
              }  // if vn1==0 && vn2==0

              if(var_name1=="p.volume" && var_name2 == "0" && var_name3 == "0"){
                vector<vector<Vector> > normals(level->numPatches());
                done = true;
                double vol;
                vector<vector<double> > volumes(level->numPatches());
                while (source >> x >> y >> z >> vol) {
                  Point pp(x,y,z);
                  if(level->containsPoint(pp)){
                    const Patch* currentpatch =
                       level->selectPatchForCellIndex(level->getCellIndex(pp));
                    int pid = currentpatch->getID();
                    min = Min(pp,min);
                    max = Max(pp,max);
                    points[pid].push_back(pp);
                    volumes[pid].push_back(vol);
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
                     dest << points[pid][I].x() << " " <<
                             points[pid][I].y() << " " <<
                             points[pid][I].z() << " " << vol << endl;
                   }
                   dest.close();
                }
              }  // if vn1==pv && vn2==0 

              if(var_name1=="0" && var_name2=="p.externalforce"
                                &&  var_name3=="0"){
                vector<vector<Vector> > normals(level->numPatches());
                done = true;
                double nx,ny,nz;
                while (source >> x >> y >> z >> nx >> ny >> nz) {
                  Point pp(x,y,z);
                  if(level->containsPoint(pp)){
                    Vector norm(nx,ny,nz);
                    const Patch* currentpatch =
                       level->selectPatchForCellIndex(level->getCellIndex(pp));
                    int pid = currentpatch->getID();
                    min = Min(pp,min);
                    max = Max(pp,max);
                    points[pid].push_back(pp);
                    normals[pid].push_back(norm);
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
                     dest << points[pid][I].x() << " " <<
                             points[pid][I].y() << " " <<
                             points[pid][I].z() << " ";
                     dest << normals[pid][I].x() << " " <<
                             normals[pid][I].y() << " " <<
                             normals[pid][I].z() << endl;
                   }
                   dest.close();
                }
              }  // if vn1==0 && vn2==pef

              if(var_name1=="0" && var_name2=="p.externalforce"
                                && var_name3=="p.fiberdir"){
                vector<vector<Vector> > normals(level->numPatches());
                vector<vector<Vector> > fiberdirs(level->numPatches());
                done = true;
                double nx,ny,nz,fibx,fiby,fibz;
                while (source >> x >> y >> z >> nx >> ny >> nz >> fibx >> fiby >> fibz){
                  Point pp(x,y,z);
                  if(level->containsPoint(pp)){
                    Vector norm(nx,ny,nz);
                    Vector fibdir(fibx,fiby,fibz);
                    const Patch* currentpatch =
                       level->selectPatchForCellIndex(level->getCellIndex(pp));
                    int pid = currentpatch->getID();
                    min = Min(pp,min);
                    max = Max(pp,max);
                    points[pid].push_back(pp);
                    normals[pid].push_back(norm);
                    fiberdirs[pid].push_back(fibdir);
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
                     dest << points[pid][I].x() << " " <<
                             points[pid][I].y() << " " <<
                             points[pid][I].z() << " ";
                     dest << normals[pid][I].x() << " " <<
                             normals[pid][I].y() << " " <<
                             normals[pid][I].z() << " ";
                     dest << fiberdirs[pid][I].x() << " " <<
                             fiberdirs[pid][I].y() << " " <<
                             fiberdirs[pid][I].z() << endl;
                   }
                   dest.close();
                }
              }  // if vn1==0 && vn2==pef  && vn3==pfibdir

              if(!done){
                 cout << "This option not yet supported:" << endl;
                 cout << " var_name1 = " << var_name1
                      << " var_name2 = " << var_name2
                      << " var_name3 = " << var_name3 << endl;
                 cout << "Feel free to add it!" << endl;
                 exit(1);
              }  // if vn1==pv && vn2==pef
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
