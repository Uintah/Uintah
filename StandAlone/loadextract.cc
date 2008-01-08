/*
 *  loadextract.cc: extract loadsegments from uintah data archive for use in LS-Dyna
 *
 *  Written by:
 *   Martin Denison
 *   Reaction Engineering International
 *   by stealing lineextract from:
 *   Jim Guilkey
 *   Department of Mechancial Engineering 
 *   by stealing timeextract from:
 *   James L. Bigler
 *   Bryan J. Worthen
 *   Department of Computer Science
 *   University of Utah
 *   June 2004
 *
 *  Copyright (C) 2004 U of U
 */

#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>

#include <Core/Math/MinMax.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <Packages/Uintah/StandAlone/loadextract.h>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

bool verbose = false;
bool quiet = false;
bool d_printCell_coords = false;
  
void
usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
        cerr << "Error parsing argument: " << badarg << endl;
    cerr << "Usage: " << progname << " [options] "
         << "-uda <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h,--help\n";
    cerr << "  -v,--variable <variable name>\n";
    cerr << "  -m,--material <material number> [defaults to 0]\n";
    cerr << "  -tlow,--timesteplow [int] (sets start output timestep to int) [defaults to 0]\n";
    cerr << "  -thigh,--timestephigh [int] (sets end output timestep to int) [defaults to last timestep]\n";
    cerr << "  -timestep,--timestep [int] (only outputs from timestep int) [defaults to 0]\n";
    cerr << "  -istart,--indexs <x> <y> <z> (cell index) [defaults to 0,0,0]\n";
    cerr << "  -iend,--indexe <x> <y> <z> (cell index) [defaults to 0,0,0]\n";
    cerr << "  -l,--level [int] (level index to query range from) [defaults to 0]\n";
    cerr << "  -o,--out <outputfilename> [defaults to stdout]\n"; 
    cerr << "  -vv,--verbose (prints status of output)\n";
    cerr << "  -q,--quiet (only print data values)\n";
    cerr << "  -cellCoords (prints the cell centered coordinates on that level)\n";
    cerr << "  --cellIndexFile <filename> (file that contains a list of cell indices)\n";
    cerr << "                                   [int 100, 43, 0]\n";
    cerr << "                                   [int 101, 43, 0]\n";
    cerr << "                                   [int 102, 44, 0]\n";
    exit(1);
}

loadextract::loadextract(std::string kfilename, double pconv, 
                         double lconv1, double lconv2):
presconv(pconv), lengthconv1(lconv1), lengthconv2(lconv2), kreader()
{
   kreader.parse_inp(kfilename.c_str());

   int i, num = kreader.nodes.size(), k;
   for(i=0; i<num; i++){
     nodes_m[kreader.nodes[i]->nid] = kreader.nodes[i];
     for(k=0; k<3; k++) kreader.nodes[i]->x[k] *= lengthconv1;
   }

   // set solid element faces first
   /*
       7/-------/6
      /       / |
    /       /   |
  4---------5   |
   |       |    |
   |    3  |    / 2
   |       |  /         ^3
   |       |/           |/ 2
   ---------             ->1
   0       1
   */
   int fcver[6][4] = {1, 2, 6, 5,
                      0, 4, 7, 3,
                      3, 7, 6, 2,
                      0, 1, 5, 4,
                      4, 5, 6, 7,
                      0, 3, 2, 1};
   num = kreader.elemSolids.size();

   std::map<std::string, FACE>::iterator itf;
   std::map<unsigned int, Node*>::iterator itn;
   std::set<unsigned int> order;
   std::set<unsigned int>::iterator ito;
   Node* n0[8];
   int j1, j2;
   for(i=0; i<num; i++){
      for(j1=0; j1<8; j1++){
         itn = nodes_m.find(kreader.elemSolids[i]->n[j1]);
         n0[j1] = itn->second;
      }
      for(j1=0; j1<8; j1++){
         j2 = j1 + 1;
         if(j2==8) j2 = 0;
         if(kreader.elemSolids[i]->n[j1]!=kreader.elemSolids[i]->n[j2]){
            if(j1==3) std::cout<<"tetrahedron"<<std::endl;
            if(j1==4) std::cout<<"wedge"<<std::endl;
            break;
         }
      }
      for(j1=0; j1<6; j1++){
         order.clear();
         for(j2=0; j2<4; j2++) order.insert(kreader.elemSolids[i]->n[fcver[j1][j2]]);
         std::ostringstream str;
         for(ito=order.begin(); ito!=order.end(); ito++) str <<"a"<<*ito;
         std::string check = str.str();
         itf = faces_m.find(str.str());
         if(itf!=faces_m.end()){
            itf->second.elem.push_back(kreader.elemSolids[i]->eid);
            if(itf->second.elem.size()>2) std::cout<<" more than two elements per face "<<itf->second.elem.size()<<std::endl;
         }else{
            FACE fce;
	    for(k=0; k<3; k++) fce.x[0][k] = 0.0;
            for(j2=0; j2<4; j2++){
	      fce.node_id[j2] = n0[fcver[j1][j2]]->nid;
              for(k=0; k<3; k++) fce.x[0][k] += n0[fcver[j1][j2]]->x[k];
	    }
            for(k=0; k<3; k++){
	      fce.x[0][k] /= 4.0;
	      fce.x[1][k] = fce.x[0][k];
	    }
            fce.elem.push_back(kreader.elemSolids[i]->eid);
            faces_m[str.str()] = fce;
         }
      } // for(j1

   } // for(i

   for(itf=faces_m.begin(); itf!=faces_m.end(); itf++){
     if(itf->second.elem.size()==1) faces.push_back(&itf->second);
   }
   iface_shell = faces.size();

   cout<<"shell elements start at "<<iface_shell<<endl;

   // now do shell elements
   num = kreader.secShells.size();
   for(i=0; i<num; i++){
      for(j1=0; j1<4; j1++) kreader.secShells[i]->t[j1] *= lengthconv2;
      kreader.secShells_m[kreader.secShells[i]->secid] = kreader.secShells[i];
   }

   num = kreader.parts.size();
   for(i=0; i<num; i++){
      kreader.parts_m[kreader.parts[i]->pid] = kreader.parts[i];
   }
   std::map<unsigned int, Part*>::iterator itp;
   std::map<unsigned int,SecShell*>::iterator itssh;

   num = kreader.elemShells.size();
   faces_shell.resize(num);
   double xx[4][3], v1[3], v2[3], v3[3];
   for(i=0; i<num; i++){
      for(j1=0; j1<4; j1++) 
	{
	  kreader.elemShells[i]->t[j1] *= lengthconv2;
	  //The check here to either put section thickness into elemShells' thickness
	  
	  if (fabs(kreader.elemShells[i]->t[j1])<1E-12)
	    {
	      itp = kreader.parts_m.find(kreader.elemShells[i]->pid);
	      itssh = kreader.secShells_m.find(itp->second->secid);
	      kreader.elemShells[i]->t[j1] = itssh->second->t[j1];
	    }
	}
      for(j1=0; j1<4; j1++){
	itn = nodes_m.find(kreader.elemShells[i]->n[j1]);
	n0[j1] = itn->second;
	for(k=0; k<3; k++) xx[j1][k] = n0[j1]->x[k];
      } // j1
      if(n0[2]->nid==n0[3]->nid){
         for(k=0; k<3; k++){
            v1[k] = xx[1][k] - xx[0][k];
            v2[k] = xx[2][k] - xx[0][k];
         }
      }else{
         for(k=0; k<3; k++){
            v1[k] = xx[2][k] - xx[0][k];
            v2[k] = xx[3][k] - xx[1][k];
         }
      }
      kreader.cross_prod(v1, v2, v3);
      double denom = 0.0;
      for(k=0; k<3; k++) denom += v3[k]*v3[k];
      denom = sqrt(denom);
      if(denom) for(k=0; k<3; k++) v3[k] /= denom;

      FACE fce;
      for(k=0; k<3; k++) fce.x[0][k] = 0.0;
      int cnt = 0;
      double thickav = 0.0;
      for(j1=0; j1<4; j1++){
	fce.node_id[j1] = kreader.elemShells[i]->n[j1];
	if(j1<3||n0[2]->nid!=n0[3]->nid){
          cnt++;
	  for(k=0; k<3; k++) fce.x[0][k] += n0[j1]->x[k];
          thickav += kreader.elemShells[i]->t[j1];
	}
      }
      thickav /= double(cnt);
      for(k=0; k<3; k++){
	fce.x[0][k] /= double(cnt);
	fce.x[0][k] += 0.5*thickav*v3[k];
	fce.x[1][k] = fce.x[0][k] - thickav*v3[k];
      }
      fce.elem.push_back(kreader.elemShells[i]->eid);
      faces_shell[i] = fce;
   }

   for(i=0; i<num;  i++){
     faces.push_back(&faces_shell[i]);
   }

   num = faces.size();
   load_curve.resize(num);

}

// arguments are the dataarchive, the successive arguments are the same as 
// the arguments to archive->query for data values.  Then comes a type 
// dexcription of the variable being queried, and last is an output stream.
//______________________________________________________________________
//
template<class T>
void loadextract::printData(
               DataArchive* archive, string& variable_name, const Uintah::TypeDescription* variable_type,
               int material, const bool use_cellIndex_file, int levelIndex,
               IntVector& var_start, IntVector& var_end, vector<IntVector> cells,
               unsigned long time_start, unsigned long time_end, ostream& out) 

{
  // query time info from dataarchive
  vector<int> index;
  vector<double> times;

  archive->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  if (!quiet){
    cout << "There are " << index.size() << " timesteps\n";
  }
  
  // set default max time value
  if (time_end == (unsigned long)-1) {
    if (verbose) {
      cout <<"Initializing time_step_upper to "<<times.size()-1<<"\n";
    }
    time_end = times.size() - 1;
  }      

  //__________________________________
  // bullet proofing 
  if (time_end >= times.size() || time_end < time_start) {
    cerr << "timestephigh("<<time_end<<") must be greater than " << time_start 
         << " and less than " << times.size()-1 << endl;
    exit(1);
  }
  if (time_start >= times.size() || time_end > times.size()) {
    cerr << "timestep must be between 0 and " << times.size()-1 << endl;
    exit(1);
  }
  
  //__________________________________
  // make sure the user knows it could be really slow if he
  // tries to output a big range of data...
  IntVector var_range = var_end - var_start;
  if (var_range.x() && var_range.y() && var_range.z()) {
    cerr << "PERFORMANCE WARNING: Outputting over 3 dimensions!\n";
  }
  else if ((var_range.x() && var_range.y()) ||
           (var_range.x() && var_range.z()) ||
           (var_range.y() && var_range.z())){
    cerr << "PERFORMANCE WARNING: Outputting over 2 dimensions\n";
  }

  // set defaults for output stream
  out.setf(ios::scientific,ios::floatfield);
  out.precision(16);
  
  bool cellNotFound = false;
  //__________________________________
  // loop over timesteps
  for (unsigned long time_step = time_start; time_step <= time_end; time_step++) {
  
    cerr << "%outputting for times["<<time_step<<"] = " << times[time_step]<< endl;

    //__________________________________
    //  does the requested level exist
    bool levelExists = false;
    GridP grid = archive->queryGrid(times[time_step]); 
    int numLevels = grid->numLevels();
   
    for (int L = 0;L < numLevels; L++) {
      const LevelP level = grid->getLevel(L);
      if (level->getIndex() == levelIndex){
        levelExists = true;
      }
    }
    if (!levelExists){
      cerr<< " Level " << levelIndex << " does not exist at this timestep " << time_step << endl;
    }
    
    if(levelExists){   // only extract data if the level exists
      const LevelP level = grid->getLevel(levelIndex);
      //__________________________________
      // User input starting and ending indicies    
      if(!use_cellIndex_file) {
          
        // find the corresponding patches
        Level::selectType patches;
        level->selectPatches(var_start, var_end + IntVector(1,1,1), patches);
        if( patches.size() == 0){
          cerr << " Could not find any patches on Level " << level->getIndex()
               << " that contain cells along line: " << var_start << " and " << var_end 
               << " Double check the starting and ending indices "<< endl;
          exit(1);
        }

        // query all the data up front
        vector<Variable*> vars(patches.size());
        for (int p = 0; p < patches.size(); p++) {
          switch (variable_type->getType()) {
          case Uintah::TypeDescription::CCVariable:
            vars[p] = scinew CCVariable<T>;
            archive->query( *(CCVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          case Uintah::TypeDescription::NCVariable:
            vars[p] = scinew NCVariable<T>;
            archive->query( *(NCVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          case Uintah::TypeDescription::SFCXVariable:
            vars[p] = scinew SFCXVariable<T>;
            archive->query( *(SFCXVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          case Uintah::TypeDescription::SFCYVariable:
            vars[p] = scinew SFCYVariable<T>;
            archive->query( *(SFCYVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          case Uintah::TypeDescription::SFCZVariable:
            vars[p] = scinew SFCZVariable<T>;
            archive->query( *(SFCZVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          default:
            cerr << "Unknown variable type: " << variable_type->getName() << endl;
          }
          
        }

        IntVector c1(-1,-1,-1), c2(0,0,0);
        Point pt1 = level->getCellPosition(c1), pt2 = level->getCellPosition(c2);
        Vector x0 = 0.5*(pt2.asVector() + pt1.asVector());
        //cout <<x0.x()<<" "<<x0.y()<<" "<<x0.z()<<endl;
        int ifc, num = faces.size();
	for(ifc=0; ifc<num; ifc++){
	  int i, j, k;
          
	  //for (CellIterator ci(var_start, var_end + IntVector(1,1,1)); !ci.done(); ci++) {
          //IntVector c = *ci;

          // find out which patch it's on (to keep the printing in sorted order.
          // alternatively, we could just iterate through the patches)
          Vector dx;
          IntVector c, c3(-1,-1,-1), diff;
          double pres[2] = {101325.0, 101325.0};
	  double xx[2][3];
          int side, nside = 1, side1, dside, side1st;
          if(ifc>=iface_shell) nside = 2;

          //if(ifc==3199) cout<<"nside "<<nside<<" "<<iface_shell<<endl;

          int p;
          if(faces[ifc]->x[0][2]>faces[ifc]->x[1][2]||ifc<iface_shell){
            side = 0;
            dside = 1;
          }else{
            side = 1;
            dside = -1;
          }
          side1st = side;

          for(side1=0; side1<nside; side1++){

	    for(k=0; k<3; k++) xx[side][k] = faces[ifc]->x[side][k];
	    
            if(side1) c3 = c;

	    do{
	      p = 0;
          
	      for (; p < patches.size(); p++) {
		dx = patches[p]->dCell();
		i = int((xx[side][0] - x0.x())/dx.x());
		j = int((xx[side][1] - x0.y())/dx.y());
		k = int((xx[side][2] - x0.z())/dx.z());
		IntVector c4(i,j,k);
		c = c4;
                diff = c - c3;
		IntVector low = patches[p]->getLowIndex();
		IntVector high = patches[p]->getHighIndex();
		if (c.x() >= low.x() && c.y() >= low.y() && c.z() >= low.z() && 
		    c.x() < high.x() && c.y() < high.y() && c.z() < high.z()){
		  break;
		}
	      }
              if(p == patches.size()) break;
              for(k=0; k<3; k++) xx[side][k] = 2.0*xx[side][k] - xx[side1st][k];
	    }while(!diff.x()&&!diff.y()&&!diff.z()&&side1);

            if(p == patches.size()) break;
          
          T val;
          Vector shift(0,0,0);  // shift the cellPosition if it's a (X,Y,Z)FC variable
          switch (variable_type->getType()) {
          case Uintah::TypeDescription::CCVariable: 
            val = (*dynamic_cast<CCVariable<T>*>(vars[p]))[c]; 
          break;
          case Uintah::TypeDescription::NCVariable: 
            val = (*dynamic_cast<NCVariable<T>*>(vars[p]))[c]; 
          break;
          case Uintah::TypeDescription::SFCXVariable: 
            val = (*dynamic_cast<SFCXVariable<T>*>(vars[p]))[c];
            shift.x(-dx.x()/2.0); 
          break;
          case Uintah::TypeDescription::SFCYVariable: 
            val = (*dynamic_cast<SFCYVariable<T>*>(vars[p]))[c];
            shift.y(-dx.y()/2.0); 
          break;
          case Uintah::TypeDescription::SFCZVariable: 
            val = (*dynamic_cast<SFCZVariable<T>*>(vars[p]))[c];
            shift.z(-dx.z()/2.0); 
          break;
          default: break;
          }

	  pt1 = level->getCellPosition(c);
	  Vector x1 = pt1.asVector();
          //if(ifc==3199) cout<<side<<" "<<c.x()<<" "<<c.y()<<" "<<c.z()
	  //    <<" "<<x1.x()<<" "<<x1.y()<<" "<<x1.z()<<" "<<val<<endl;
          
          pres[side] = val;
          side += dside;
	  } // for(side1
	  if (p == patches.size()) {
	    cellNotFound = true;
	    continue;
	  }

	  //if(pres[1]<pres[0]) pres[1] = 101325.0;
	  //else pres[0] = 101325.0;

          TIME_PRES t_p;
          t_p.time = times[time_step];
	  //presconv = 10.0; //1.0e-6; // 10.0;
          t_p.pres = presconv*(pres[0] - pres[1]);
          load_curve[ifc].push_back(t_p);
	  /*if(d_printCell_coords){
            Point point = level->getCellPosition(c);
            Vector here = point.asVector() + shift;
            out << here.x() << " "<< here.y() << " " << here.z() << " "<<val << endl;;
          }else{
            out <<ifc<<" "<< c.x() << " "<< c.y() << " " << c.z() << " "<< val << endl;;
	    }*/
        }
        for (unsigned i = 0; i < vars.size(); i++)
          delete vars[i];
      }

      //__________________________________
      // If the cell indicies were read from a file. 
      if(use_cellIndex_file) {
        for (int i = 0; i<(int) cells.size(); i++) {
          IntVector c = cells[i];
          vector<T> values;
          try {
            archive->query(values, variable_name, material, c, 
                            times[time_step], times[time_step], levelIndex);
          } catch (const VariableNotFoundInGrid& exception) {
            cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
            exit(1);
          }
          if(d_printCell_coords){
            Point p = level->getCellPosition(c);
            out << p.x() << " "<< p.y() << " " << p.z() << " "<< values[0] << endl;
          }else{
            out << c.x() << " "<< c.y() << " " << c.z() << " "<< values[0] << endl;
          }
        }
      }
      out << endl;
    } // if level exists
    
  } // timestep loop
  output();
} 

void loadextract::output(){
	FILE *lcoutput;
	int i, j, k, lc_len;
        vector<bool> write_curve(faces.size(),true);
        std::map<unsigned int, Node*>::iterator itn;

	lcoutput=fopen("blast_loads.k", "w");
	if (lcoutput==NULL)
	{
		printf("Can't open %s for output!\n", "BLAST_LOADS.k");
		exit(-1);
	}

	fprintf(lcoutput,"*LOAD_SEGMENT\n");
	fprintf(lcoutput,"$-------------------------------------\n");
	for (i=0; i<faces.size(); i++)
	{
          double maxpres = 0.0;
          lc_len = load_curve[i].size();
	  for(j=0; j<lc_len; j++) if(fabs(load_curve[i][j].pres)>maxpres) maxpres = fabs(load_curve[i][j].pres);
          for(j=0; j<4; j++){
	    itn = nodes_m.find(faces[i]->node_id[j]);
	    //if(itn!=nodes_m.end()){
	      //write_curve[i] = write_curve[i]&&itn->second->x[1]>0.15;
	    //}
	  }
          if(!write_curve[i])//||maxpres<10000.0*presconv) //1.0e-6)
	    write_curve[i] = false;
          else
		fprintf(lcoutput,"%d, 1.0, 0.0, %d, %d, %d, %d\n",i+1, faces[i]->node_id[0],faces[i]->node_id[1],faces[i]->node_id[2],faces[i]->node_id[3]);
	}

	
	for (i=0;i<faces.size();i++)
	{
	  if(write_curve[i]){
		fprintf(lcoutput,"*DEFINE_CURVE\n");
		fprintf(lcoutput,"%d, 0, 1.0, 1.0\n", i+1);
		lc_len=load_curve[i].size();
		for (j=0; j<lc_len;j++ )
		{
			fprintf(lcoutput, "%lf,", load_curve[i][j].time);

			fprintf(lcoutput, " %lf\n", load_curve[i][j].pres);
			
		}
                //fprintf(lcoutput, "%lf,", 0.020);
                //fprintf(lcoutput, "%lf\n", load_curve[i][lc_len-1].pres);
	  }
	}
	fprintf(lcoutput,"*END\n");
	fclose(lcoutput);
}

/*_______________________________________________________________________
 Function:  readCellIndicies--
 Purpose: reads in a list of cell indicies
_______________________________________________________________________ */
/*void readCellIndicies(const string& filename, vector<IntVector>& cells)
{ 
  // open the file
  ifstream fp(filename.c_str());
  if (!fp){
    cerr << "Couldn't open the file that contains the cell indicies " << filename<< endl;
  }
  char c;
  int i,j,k;
  string text, comma;  
  
  while (fp >> c) {
    fp >> text>>i >> comma >> j >> comma >> k;
    IntVector indx(i,j,k);
    cells.push_back(indx);
    fp.get(c);
  }
  // We should do some bullet proofing here
  //for (int i = 0; i<(int) cells.size(); i++) {
  //  cout << cells[i] << endl;
  //}
}*/

//______________________________________________________________________
//    Notes:
// Now the material index is kind of a hard thing.  There is no way
// to reliably determine a default material.  Materials are defined
// on the patch for each varialbe, so this subset of materials could
// change over patches.  We can guess, that there will be a material
// 0.  This shouldn't cause the program to crash.  It will spit out
// an exception and exit gracefully.


void loadextract::extract(int argc, char** argv)
{

  //__________________________________
  //  Default Values
  bool use_cellIndex_file = false;

  unsigned long time_start = 0;
  unsigned long time_end = (unsigned long)-1;
  string input_uda_name;  
  string input_file_cellIndices;
  string output_file_name("-");
  IntVector var_start(0,0,0);
  IntVector var_end(1000000,1000000,1000000);
  int levelIndex = 0;
  vector<IntVector> cells;
  string variable_name("press_CC");

  int material = 0;
  
  //__________________________________
  // Parse arguments

  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-v" || s == "--variable") {
      variable_name = string(argv[++i]);
    } else if (s == "-m" || s == "--material") {
      material = atoi(argv[++i]);
    } else if (s == "-vv" || s == "--verbose") {
      verbose = true;
    } else if (s == "-q" || s == "--quiet") {
      quiet = true;
    } else if (s == "-tlow" || s == "--timesteplow") {
      time_start = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-thigh" || s == "--timestephigh") {
      time_end = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-timestep" || s == "--timestep") {
      int val = strtoul(argv[++i],(char**)NULL,10);
      time_start = val;
      time_end = val;
    } else if (s == "-istart" || s == "--indexs") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_start = IntVector(x,y,z);
    } else if (s == "-iend" || s == "--indexe") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_end = IntVector(x,y,z);
    } else if (s == "-l" || s == "--level") {
      levelIndex = atoi(argv[++i]);
    } else if( (s == "-h") || (s == "--help") ) {
      usage( "", argv[0] );
    } else if (s == "-uda") {
      input_uda_name = string(argv[++i]);
    } else if (s == "-o" || s == "--out") {
      output_file_name = string(argv[++i]);
    } else if (s == "--cellIndexFile") {
      use_cellIndex_file = true;
      input_file_cellIndices = string(argv[++i]);
    } else if (s == "--cellCoords" || s == "-cellCoords" ) {
      d_printCell_coords = true;
    }else {
      usage(s, argv[0]);
    }
  }
  
  if(input_uda_name == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    DataArchive* archive = scinew DataArchive(input_uda_name);
    
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;

    archive->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    if (verbose) cout << "There are " << vars.size() << " variables:\n";
    bool var_found = false;
    unsigned int var_index = 0;
    for (;var_index < vars.size(); var_index++) {
      if (variable_name == vars[var_index]) {
        var_found = true;
        break;
      }
    }
    //__________________________________
    // bulletproofing
    if (!var_found) {
      cerr << "Variable \"" << variable_name << "\" was not found.\n";
      cerr << "If a variable name was not specified try -var [name].\n";
      cerr << "Possible variable names are:\n";
      var_index = 0;
      for (;var_index < vars.size(); var_index++) {
        cout << "vars[" << var_index << "] = " << vars[var_index] << endl;
      }
      cerr << "Aborting!!\n";
      exit(-1);
    }

    if (!quiet) {
      cout << vars[var_index] << ": " << types[var_index]->getName() 
           << " being extracted for material "<<material
           <<" at index "<<var_start << " to " << var_end <<endl;
    }
    //__________________________________
    // get type and subtype of data
    const Uintah::TypeDescription* td = types[var_index];
    const Uintah::TypeDescription* subtype = td->getSubType();
     
    //__________________________________
    // Open output file, call printData with it's ofstream
    // if no output file, call with cout
    ostream *output_stream = &cout;
    if (output_file_name != "-") {
      if (verbose) cout << "Opening \""<<output_file_name<<"\" for writing.\n";
      ofstream *output = new ofstream();
      output->open(output_file_name.c_str());
      
      if (!(*output)) {   // bullet proofing
        cerr << "Could not open "<<output_file_name<<" for writing.\n";
        exit(1);
      }
      output_stream = output;
    } else {
      //output_stream = cout;
    }
    
    //__________________________________    
    // read in cell indices from a file
    //if ( use_cellIndex_file) {
    //readCellIndicies(input_file_cellIndices, cells);
    //}
    
    //__________________________________
    //  print data
    switch (subtype->getType()) {
    case Uintah::TypeDescription::double_type:
      printData<double>(archive, variable_name, td, material, use_cellIndex_file,
                        levelIndex, var_start, var_end, cells,
                        time_start, time_end, *output_stream);
      break;
    case Uintah::TypeDescription::float_type:
      printData<float>(archive, variable_name, td, material, use_cellIndex_file,
                        levelIndex, var_start, var_end, cells,
                        time_start, time_end, *output_stream);
      break;
    case Uintah::TypeDescription::int_type:
      printData<int>(archive, variable_name, td, material, use_cellIndex_file,
                     levelIndex, var_start, var_end, cells,
                     time_start, time_end, *output_stream);
      break;
      /*case Uintah::TypeDescription::Vector:
      printData<Vector>(archive, variable_name, td, material, use_cellIndex_file,
                        levelIndex, var_start, var_end, cells,
                        time_start, time_end, *output_stream);    
			break;*/
      /*case Uintah::TypeDescription::Other:
      if (subtype->getName() == "Stencil7") {
        printData<Stencil7>(archive, variable_name, td, material, use_cellIndex_file,
                          levelIndex, var_start, var_end, cells,
                          time_start, time_end, *output_stream);    
        break;
	}*/
      // don't break on else - flow to the error statement
    case Uintah::TypeDescription::Matrix3:
    case Uintah::TypeDescription::bool_type:
    case Uintah::TypeDescription::short_int_type:
    case Uintah::TypeDescription::long_type:
    case Uintah::TypeDescription::long64_type:
      cerr << "Subtype is not implemented\n";
      exit(1);
      break;
    default:
      cerr << "Unknown subtype\n";
      exit(1);
    }

    // Delete the output file if it was created.
    if (output_file_name != "-") {
      delete((ofstream*)output_stream);
    }

  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    exit(1);
  } catch(...){
    cerr << "Caught unknown exception\n";
    exit(1);
  }
}
