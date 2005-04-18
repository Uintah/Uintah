/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  ForwardIPM.cc:
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>
#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>
#include <Core/ImportExport/ExecConverter.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>

namespace SCIRun {

class ForwardIPM : public Module {
public:
  ForwardIPM(GuiContext* ctx);

  virtual void execute();

private:

  bool write_par_file(string filename);
};


DECLARE_MAKER(ForwardIPM)

ForwardIPM::ForwardIPM(GuiContext* ctx)
  : Module("ForwardIPM", ctx, Filter, "NeuroFEM", "BioPSE")
{
}



static bool
write_geo_file(ProgressReporter *pr, FieldHandle field, const char *filename)
{
  
  TetVolMesh *mesh = dynamic_cast<TetVolMesh *>(field->mesh().get_rep());
  if (mesh == 0)
  {
    pr->error("Field does not contain a TetVolMesh.");
    return false;
  }

 
  TetVolMesh::Node::iterator nitr, neitr;
  TetVolMesh::Node::size_type nsize; 
  mesh->begin(nitr);
  mesh->end(neitr);
  mesh->size(nsize);

  TetVolMesh::Cell::iterator citr, ceitr;
  TetVolMesh::Cell::size_type csize; 
  mesh->begin(citr);
  mesh->end(ceitr);
  mesh->size(csize); 

  FILE *f = fopen(filename, "w");
  if (f == NULL)
  {
    pr->error(string("Unable to open file '") + filename + "' for writing.");
    return false;
  }

  // Write header.
  fprintf(f, "BOI - GEOMETRIEFILE\n");
  fprintf(f, "===================================================================\n");
  fprintf(f, "===================================================================\n");
  fprintf(f, "BOI - STEUERKARTE\n");
  fprintf(f, "  ANZAHL DER KNOTEN             : %d\n",(unsigned)(nsize));
  fprintf(f, "  ANZAHL DER ELEMENTE           : %d\n",(unsigned)(csize));
  fprintf(f, "  GEOMETR. STRUKTUR - DIMENSION :      3\n");
  fprintf(f, "EOI - STEUERKARTE\n");
  fprintf(f, "===================================================================\n");
  fprintf(f, "===================================================================\n");    
  fprintf(f, "BOI - KOORDINATENKARTE\n");  

  // write it out
  int odd=1;
  while (nitr != neitr)
  {
    Point p;
    mesh->get_point(p, *nitr);

    if(odd==1) {
      fprintf(f, "   %11.5e %11.5e %11.5e ", p.x(), p.y(), p.z());
      odd=0;
    }
    else{
      fprintf(f, "   %11.5e %11.5e %11.5e\n", p.x(), p.y(), p.z());
      odd=1;
    }
    // Write p;

    ++nitr;
  }
  // write middle part
  fprintf(f, "\nEOI - KOORDINATENKARTE\n");
  fprintf(f, "===================================================================\n");
  fprintf(f, "===================================================================\n"); 
  fprintf(f, "BOI - ELEMENTKNOTENKARTE\n");

  while (citr != ceitr)
  {
    TetVolMesh::Node::array_type nodes; // 0 based
    mesh->get_nodes(nodes, *citr);

    // Write nodes
    fprintf(f,  "  303: %6d%6d%6d%6d\n",
            nodes[0]+1, nodes[1]+1, nodes[2]+1, nodes[3]+1);

    ++citr;
  }

  // Write tail, bottom part.
  fprintf(f, "EOI - ELEMENTKNOTENKARTE\n");
  fprintf(f, "===================================================================\n");
  fprintf(f, "===================================================================\n");
  fprintf(f, "BOI - GEOMETRIEFILE\n");
  
  fclose(f);
  
  return true;
}



static bool
write_knw_file(ProgressReporter *pr, FieldHandle field, const char *filename)
{
  TetVolField<Tensor> *tfield =
    dynamic_cast<TetVolField<Tensor> *>(field.get_rep());
  TetVolMesh *mesh = dynamic_cast<TetVolMesh *>(field->mesh().get_rep());
  if (mesh == 0)
  {
    pr->error("Field does not contain a TetVolMesh.");
    return false;
  }

  FILE *f = fopen(filename, "w");
  //FILE *f =fopen("/home/sci/slew/ncrr/dataset/tensor.con","w");
  if (f == NULL)
  {
    pr->error(string("Unable to open file '") + filename + "' for writing.");
    return false;
  }

  // Write header.
  fprintf(f, "BOI - TENSORVALUEFILE\n");
  fprintf(f, "========================================================\n");
  fprintf(f, "========================================================\n");
  fprintf(f, "BOI - TENSOR\n");  

  // Write it out.
  TetVolMesh::Cell::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  
  while (itr != eitr)
  {
    Tensor t;
    tfield->value(t, *itr);

    fprintf(f, "%10d  %11f  %11f  %11f\n           %11f  %11f  %11f\n",
            1+*itr,  // fortran index, +1
            t.mat_[0][0], t.mat_[1][1], t.mat_[2][2],
            t.mat_[0][1], t.mat_[0][2], t.mat_[1][2]);

    ++itr;
  }

  // Write tail, bottom part.
  fprintf(f, "EOI - TENSOR\n");
  fprintf(f, "========================================================\n");
  fprintf(f, "========================================================\n");
  fprintf(f, "EOI - TENSORVALUEFILE\n");

  fclose(f);

  return true;
}


static bool
write_elc_file(ProgressReporter *pr, FieldHandle fld, const char *filename)
{
   PointCloudMesh *mesh = dynamic_cast<PointCloudMesh *>(fld->mesh().get_rep());

    if (mesh == 0)
  {
    pr->error("Field does not contain a PointCloudMesh.");
    return false;
  }

  PointCloudMesh::Node::iterator niter; 
  PointCloudMesh::Node::iterator niter_end; 
  PointCloudMesh::Node::size_type nsize; 
  mesh->begin(niter);
  mesh->end(niter_end);
  mesh->size(nsize);
  
  FILE *f = fopen(filename, "w");
  if (f == NULL)
  {
    pr->error(string("Unable to open file '") + filename + "' for writing.");
    return false;
  }

  //write header
  fprintf(f, "# %d Electrodes\n", (unsigned)(nsize));
  fprintf(f, "ReferenceLabel	avg\n");
  fprintf(f, "NumberPositions=\t%d\n",(unsigned)(nsize));
  fprintf(f, "UnitPosition	mm\n");
  fprintf(f, "Positions\n");

  //write it out
  while(niter != niter_end) {
    Point p;
    mesh->get_center(p, *niter);
    fprintf(f, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }     

  fprintf(f,"Labels\n");
  fprintf(f,"FPz \n");

  fclose(f);
  
  return true;
}


static bool
write_dip_file(ProgressReporter *pr, MatrixHandle mh1, MatrixHandle mh2, const char *filename)
{

  DenseMatrix *dp_pos = dynamic_cast<DenseMatrix *>(mh1.get_rep());
  if (!dp_pos) {
    pr->error("Error -- input field wasn't a DenseMatrix");
    return false;
  }

  int nd=dp_pos->nrows();
  
  DenseMatrix *dp_vec = dynamic_cast<DenseMatrix *>(mh2.get_rep());
  if (!dp_vec) {
    pr->error("Error -- input field wasn't a DenseMatrix");
    return false;
  }
  
  int nr=dp_vec->nrows();
  int nc=dp_vec->ncols();
  cerr << "Number of rows = " << nr << "  number of columns = " << nc << "\n";
  
  FILE *f = fopen(filename, "wt");
  if (!f) {
    pr->error(string("Error -- couldn't open output file: ") + filename );
    return false;
  }

  // write header
  fprintf(f,"# 1 dipole(s) at %d time steps", nr);
  fprintf(f,"UnitPosition\tmm");
  fprintf(f,"UnitMoment\tnAm");
  fprintf(f,"UnitTime\tms");  
  fprintf(f,"NumberPositions=\t%d\n",nd);
  fprintf(f,"NumberTimeSteps=\t%d\n",nr);
  fprintf(f,"TimeSteps\t0(1)%d\n",(unsigned)nr-1);
  fprintf(f,"FirstTimeStep\t0\n");
  fprintf(f,"LastTimeStep\t%d\n",(unsigned)(nr-1));

  // write dipole moment
  fprintf(f,"MomentsFixed\n");
  double sum = 0.0;
  for (int c=0; c<nc; c++){
    sum += ((*dp_vec)[0][c])*((*dp_vec)[0][c]);
  }
  
  sum = sqrt(sum);
  fprintf(f,"%f\t%f\t%f\n",(*dp_vec[0][0])/sum,(*dp_vec[0][1])/sum, (*dp_vec[0][2])/sum);

  // write dipole position
  fprintf(f,"PositionFixed\n");
  for (int i=0; i<3; i++)
    fprintf(f,"%10.6f",(*dp_pos)[0][i]);
  fprintf(f,"\n");

  //  write dipole magnitude
  fprintf(f,"Magnitudes");
  for (int r=0; r<nr; r++) {
    double sum = 0.0;
    for (int c=0; c<nc; c++){
      sum += ((*dp_vec)[r][c])*((*dp_vec)[r][c]);
      //      fprintf(f, "%11.5e\t", (*dp_vec)[r][c]);
    }
    fprintf(f,"%e ",sqrt(sum));
  }
  fprintf(f,"\n");
  fprintf(f,"NoLabels");
  fclose(f);

  return true;
}


static bool
send_result_file(MatrixOPort *result_port, string filename)
{
  ifstream matstream(filename.c_str(),ios::in);
  if (matstream.fail()) {
    cerr << "Error -- Could not open file " << filename << "\n";
    return false;
  }

  string tmp;
  int nc,nr;
  matstream >> tmp >> nc;
  matstream >> tmp >> nr;
  cerr << "Number of Electrode = " << nc << "\n" << "Number of Time Step = " << nr << "\n";

  //skip text lines
  for (int i=0; i<4; ++i){
    getline(matstream, tmp);
    // cerr << tmp << "\n";
  }

  DenseMatrix *dm = scinew DenseMatrix(nr,nc);

  // write number of row & column
  (*dm)[0][0] = nr;
  (*dm)[0][1] = nc;

  // write potentials on electrodes for each time step 
  int r,c;
  for (r=1; r<nr; r++)
    for (c=0; c<nc; c++) {
      double d;
      matstream >> d;
      (*dm)[r][c]=d;
      //	cerr << "matrix["<<r<<"]["<<c<<"]="<<d<<"\n";
    }
  cerr << "done building matrix.\n";

  //  result->set_raw(false);
  result_port->send(MatrixHandle(dm));

  return true;
}


bool
ForwardIPM::write_par_file(string filename)
{
  FILE *f = fopen(filename.c_str(), "w");
  if (f == NULL)
  {
    error("Unable to open file '" + filename + "' for writing.");
    return false;
  }

  fprintf(f, "#Parameter file: FEM for source simulation\n");
  fprintf(f, "\n");
  fprintf(f, "[ReferenceData]\n");
  fprintf(f, "#Number of channels\n");
  fprintf(f, "numchan= 71\n");
  fprintf(f, "numsamples=  1\n");
  fprintf(f, "startsample= 0\n");
  fprintf(f, "stopsample=  0\n");
  fprintf(f, "\n");
  fprintf(f, "[RegularizationTikhonov]\n");
  fprintf(f, "# estimated signal to noise ratio\n");
  fprintf(f, "estimatedSNR= 10000.0\n");
  fprintf(f, "\n");
  fprintf(f, "[STR]\n");
  fprintf(f, "# spatial regularization parameter\n");
  fprintf(f, "lambda= 0.1\n");
  fprintf(f, "# temporal regularization parameter\n");
  fprintf(f, "mue= 0.1\n");
  fprintf(f, "# degree of temporal smoothness\n");
  fprintf(f, "degree= 2\n");
  fprintf(f, "# fast modus 1 leads to approximated solution, much faster.\n");
  fprintf(f, "fast= 1\n");
  fprintf(f, "\n");
  fprintf(f, "[NeuroFEMSimulator]\n");
  fprintf(f, "# GENERAL PARAMETERS\n");
  fprintf(f, "# residuum of the forward solution \n");
  fprintf(f, "toleranceforwardsolution= .10000E-08\n");
  fprintf(f, "# degree of Gaussian integration (2 is recommended) \n");
  fprintf(f, "degreeofintegration= 2\n");
  fprintf(f, "# different analytical solutions for the EEG problem \n");
  fprintf(f, "analyticalsolution= 1\n");
  fprintf(f, "\n");
  fprintf(f, "# should METIS repartition for parallel run?\n");
  fprintf(f, "metisrepartitioning= 1\n");
  fprintf(f, "\n");
  fprintf(f, "# SOLVER (1:Jakobi-CG, 2:IC(0)-CG, 3:AMG-CG, 4:PILUTS(ILDLT)-CG) \n");
  fprintf(f, "solvermethod= 3\n");
  fprintf(f, "# use or use not lead field basis approach\n");
  fprintf(f, "associativity= 1\n");
  fprintf(f, "# NeuroFEM Solver\n");
  fprintf(f, "# parameter file for Pebbles solver \n");
  fprintf(f, "pebbles= pebbles.inp\n");
  fprintf(f, "# ONLY for MultiRHS-Pebbles: number of right-hand-sides which are solved simultaneously\n");
  fprintf(f, "pebblesnumrhs= 1\n");
  fprintf(f, "\n");
  fprintf(f, "# SOURCE SIMULATION\n");
  fprintf(f, "# threshold (percentage of the greatest dipole strength) of all dipoles to appear in the result files \n");
  fprintf(f, "dipolethreshold= .10000E+01\n");
  fprintf(f, "# blurring single loads to adjacent nodes by means of the Gaussian (normal) distribution \n");
  fprintf(f, "sourcesimulationepsilondirac= .10000E-08\n");
  fprintf(f, "\n");
  fprintf(f, "# DIPOLE MODELING\n");
  fprintf(f, "# dipole modeling; weighting of the source distribution with the power of the declared value \n");
  fprintf(f, "dipolemodelingsmoothness= 2\n");
  fprintf(f, "# power of the dipole moments to be considered \n");
  fprintf(f, "dipolemodelingorder= 2\n");
  fprintf(f, "# necessary internal scaling factor; should be larger than twice the element edge length \n");
  fprintf(f, "dipolemodelingscale= 20.000\n");
  fprintf(f, "# Lagrangian multiplier for the (inverse) dipole modeling \n");
  fprintf(f, "dipolemodelinglambda= .10000E-05\n");
  fprintf(f, "# source-sink separation of the analytical dipole model \n");
  fprintf(f, "dipolemodelingdistance= 1.0000\n");
  fprintf(f, "# use rango dipole model \n");
  fprintf(f, "dipolemodelingrango= 0\n");
  fprintf(f, "\n");
  fprintf(f, "# Monopole/Dipole\n");
  fprintf(f, "# calculation of the potential distribution for spherical, homogeneous  structures by means of a analytical description \n");
  fprintf(f, "analyticaldipole= 0\n");
  fprintf(f, "# forward solution computation with monopoles or dipoles\n");
  fprintf(f, "dipolesources= 1\n");
  fprintf(f, "# spread a nodal load to adjacent nodes\n");
  fprintf(f, "sourcesimulation= 1\n");
  fprintf(f, "#to compare analytical and numerical solutions, an integral average value of the potential is subtracted\n");
  fprintf(f, "#from the analytical results because they are not related to a mean potential (in contrast to the numerical solution)\n");
  fprintf(f, "averagecorrection= 0\n");
  fprintf(f, "\n");
  fprintf(f, "[NeuroFEMGridGenerator] \n");
  fprintf(f, "# Number of materials \n");
  fprintf(f, "nummaterials= 7\n");
  fprintf(f, "# Conductivities of fem head model\n");
  fprintf(f, "\n");
  fprintf(f, "#define EXTRA           3       // tissue codes (see D1.2b)\n");
  fprintf(f, "#define SKULL           1\n");
  fprintf(f, "#define CSF             8\n");
  fprintf(f, "#define GREY_MATTER     7\n");
  fprintf(f, "#define WHITE_MATTER    6\n");
  fprintf(f, "\n");
  fprintf(f, "#skin_cond       0.33;\n");
  fprintf(f, "#skull_cond      0.0042;\n");
  fprintf(f, "#csf_cond        1.79;\n");
  fprintf(f, "#grey_cond       0.337;\n");
  fprintf(f, "#white_cond_iso  0.14;\n");
  fprintf(f, "\n");
  fprintf(f, "# The first value corresponds to the electrode which will be added to the mesh \n");
  fprintf(f, "conductivities= \n");
  fprintf(f, "1.0 0.33 0.0042 0.33 0.33 0.33 0.033\n");
  fprintf(f, "# Labels in the head model corresponding to the different tissues\n");
  fprintf(f, "# The first value corresponds to the electrode which will be added to the mesh \n");
  fprintf(f, "labels=\n");
  fprintf(f, "1000 3 1 8 7 6 9\n");
  fprintf(f, "\n");
  fprintf(f, "# Tissue labels in the head model for which the tensor valued coductivity \n");
  fprintf(f, "# should be used if available\n");
  fprintf(f, "tensorlabels=\n");
  fprintf(f, "#0 0 1 0 0 1 0\n");
  fprintf(f, "0 0 1 0 0 1 0\n");
  fprintf(f, "\n");
  fprintf(f, "# -1- for sorted conductivities\n");
  fprintf(f, "sortedconductivities= 0\n");
  fprintf(f, "\n");
  fprintf(f, "[InitialGuess]\n");
  fprintf(f, "numinitialguesspos= 1\n");
  fprintf(f, "posx=\n");
  fprintf(f, "#0.044175999 \n");
  fprintf(f, "0.097\n");
  fprintf(f, "posy=\n");
  fprintf(f, "#0.077307999\n");
  fprintf(f, "0.154\n");
  fprintf(f, "posz=\n");
  fprintf(f, "#0.135539993\n");
  fprintf(f, "0.128\n");
  fprintf(f, "numinitialguessdir= 1\n");
  fprintf(f, "dirx=\n");
  fprintf(f, "#0.0\n");
  fprintf(f, "1.0  \n");
  fprintf(f, "diry=\n");
  fprintf(f, "#1.0\n");
  fprintf(f, "0.0\n");
  fprintf(f, "dirz=\n");
  fprintf(f, "#1.0\n");
  fprintf(f, "0.0 \n");
  fprintf(f, "\n");
  fprintf(f, "[FileFormats]   \n");
  fprintf(f, "#ReferenceData file: 1= Vista, 2= ASA, 3= ASCII\n");
  fprintf(f, "ReferenceDataIn= 2\n");
  fprintf(f, "#LeadfieldMatrix input file: 1= Vista, 2= ASA \n");
  fprintf(f, "LeadfieldIn= 1\n");
  fprintf(f, "#SensorConfiguration file: 1= Vista, 2= ASA \n");
  fprintf(f, "SensorconfigurationIn= 2 \n");
  fprintf(f, "#Source Space Grid: 1= Vista, 2= ASA, 3= CAUCHY\n");
  fprintf(f, "SourceSpaceIn= 2\n");
  fprintf(f, "#FEM HeadGrid: 1= Vista, 2= ASA, 3= CAUCHY\n");
  fprintf(f, "HeadGridIn= 3\n");
  fprintf(f, "#Output File; Format: 1= Vista, 2 = ASA, 3 = Vista + ASA, 4 = Vista + Ascii, 5 = Cauchy, 6 = SCIRun \n");
  fprintf(f, "ResultOut= 4\n");
  
  fclose(f);
  return true;
}


void
ForwardIPM::execute()
{
  FieldIPort *condmesh_port = (FieldIPort *)get_iport("CondMesh");
  FieldHandle condmesh;
  if (!(condmesh_port->get(condmesh) && condmesh.get_rep()))
  {
    warning("CondMesh field required to continue.");
    return;
  }

  if (condmesh->mesh()->get_type_description()->get_name() != "TetVolMesh")
  {
    error("CondMesh must contain a TetVolMesh.");
    return;
  }
  if (condmesh->query_tensor_interface(this) == 0)
  {
    error("CondMesh must be a tensor field.");
    return;
  }
  if (condmesh->basis_order() != 0)
  {
    error("CondMesh must have conductivities at the cell centers.");
    return;
  }

  FieldIPort *electrodes_port = (FieldIPort *)get_iport("Electrodes");
  FieldHandle electrodes;
  if (!(electrodes_port->get(electrodes) && electrodes.get_rep()))
  {
    warning("Electrodes field required to continue.");
    return;
  }
  
  MatrixIPort *dipole_port1 = (MatrixIPort *)get_iport("Dipole Positions");
  MatrixHandle dipole1;
  
  if (!(dipole_port1->get(dipole1) && dipole1.get_rep()))
  {
    warning("Dipole required to continue.");
    return;
  }

  if (false) // Some dipole error checking.
  {
    error("Dipole must contain Vectors at Nodes.");
    return;
  }

  MatrixIPort *dipole_port2 = (MatrixIPort *)get_iport("Dipole Moments");
  MatrixHandle dipole2;
  
  if (!(dipole_port2->get(dipole2) && dipole2.get_rep()))
  {
    warning("Dipole required to continue.");
    return;
  }

  if (false) // Some dipole error checking.
  {
    error("Dipole must contain Vectors at Nodes.");
    return;
  }
  
  // Make our tmp directory
  const string tmpdir = "/tmp/ForwardIPM" + to_string(getpid()) +"/";
  const string tmplog = tmpdir + "forward.log";
  const string resultfile = tmpdir + "result.msr";
  const string condmeshfile = tmpdir + "ca_head.geo";
  const string condtensfile = tmpdir + "ca_perm.knw";
  const string electrodefile = tmpdir + "electrode.elc";
  const string dipolefile = tmpdir + "dipole.dip";
  const string parafile = tmpdir + "forward.par";

  try {
    // Make our tmp directory.
    mode_t umsk = umask(00);
    if (mkdir(tmpdir.c_str(), 0700) == -1)
    {
      error("Unable to open a temporary working directory.");
      throw false;
    }
    umask(umsk);

    // Write out condmesh
    if (!write_geo_file(this, condmesh, condmeshfile.c_str()))
    {
      error("Unable to export CondMesh geo file.");
      throw false;
    }
    // Write out condmesh
    if (!write_knw_file(this, condmesh, condtensfile.c_str()))
    {
      error("Unable to export CondMesh knw file.");
      throw false;
    }

    // Write out Electrodes
    if (!write_elc_file(this, electrodes, electrodefile.c_str()))
    {
      error("Unable to export electrode file.");
      throw false;
    }

    // Write out Dipole
    if (!write_dip_file(this, dipole1, dipole2, dipolefile.c_str()))
    {
      error("Unable to export dipole file.");
      throw false;
    }

    // Write out our parameter file.
    if (!write_par_file(parafile)) { throw false; }

    // Construct our command line.
    const string ipmfile = "ipm";
    const string command = "(cd " + tmpdir + ";" +
      ipmfile + " -i sourcesimulation" +
      " -h " + condmeshfile + " -p " + parafile + " -s " + electrodefile +
      " -dip " + dipolefile + " -o " + resultfile + " -fwd FEM -sens EEG)";

    
    // Execute the command.  Last arg just a tmp file for logging.
    if (!Exec_execute_command(this, command, tmplog))
    {
      error("The ipm program failed to run for some unknown reason.");
      throw false;
    }

    // Read in the results and send them along.
    MatrixOPort *mat_oport = (MatrixOPort *)get_oport("DenseMatrix");
    if (!send_result_file(mat_oport,resultfile))
    {
	error("Unable to send denseMatrix");
	throw false;
    }

    throw true; // cleanup.
  }
  catch (...)
  {
    // Clean up our temporary files.
    unlink(condmeshfile.c_str());
    unlink(condtensfile.c_str());
    unlink(parafile.c_str());
    unlink(electrodefile.c_str());
    unlink(dipolefile.c_str());
    unlink(resultfile.c_str());
    unlink(tmplog.c_str());
    rmdir(tmpdir.c_str());
  }
}


} // End namespace SCIRun
