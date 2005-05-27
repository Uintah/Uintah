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
 *  InverseIPM.cc:
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
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>

namespace SCIRun {

class InverseIPM : public Module {
public:
  GuiString ipm_pathTCL_;
  GuiInt numchanTCL_;
  GuiInt numsamplesTCL_;
  GuiInt startsampleTCL_;
  GuiInt stopsampleTCL_;
  GuiInt associativityTCL_;
  GuiDouble posxTCL_;
  GuiDouble posyTCL_;
  GuiDouble poszTCL_;
  GuiDouble dirxTCL_;
  GuiDouble diryTCL_;
  GuiDouble dirzTCL_;
  GuiDouble eps_matTCL_;

  
  InverseIPM(GuiContext* ctx);

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:

  bool write_par_file(string filename);
  bool write_pebbles_file(string filename);
};


DECLARE_MAKER(InverseIPM)

InverseIPM::InverseIPM(GuiContext* ctx)
  : Module("InverseIPM", ctx, Filter, "NeuroFEM", "BioPSE"),
    ipm_pathTCL_(ctx->subVar("ipm_pathTCL")),
    numchanTCL_(ctx->subVar("numchanTCL")),
    numsamplesTCL_(ctx->subVar("numsamplesTCL")),
    startsampleTCL_(ctx->subVar("startsampleTCL")),
    stopsampleTCL_(ctx->subVar("stopsampleTCL")),
    associativityTCL_(ctx->subVar("associativityTCL")),
    posxTCL_(ctx->subVar("posxTCL")),
    posyTCL_(ctx->subVar("posyTCL")),
    poszTCL_(ctx->subVar("poszTCL")),
    dirxTCL_(ctx->subVar("dirxTCL")),
    diryTCL_(ctx->subVar("diryTCL")),
    dirzTCL_(ctx->subVar("dirzTCL")),
    eps_matTCL_(ctx->subVar("eps_matTCL"))
{
}


bool
write_geo_file(ProgressReporter *pr, FieldHandle field, const char *filename);

bool
write_knw_file(ProgressReporter *pr, FieldHandle field, const char *filename);

bool
write_elc_file(ProgressReporter *pr, FieldHandle fld, const char *filename);

bool
write_pebbles_file(string filename);

  
bool
write_potential_file(ProgressReporter *pr, MatrixHandle mat,
                     const char *filename)
{
  DenseMatrix *dm = dynamic_cast<DenseMatrix *>(mat.get_rep());
  if (!dm) {
    pr->error("Error -- input field wasn't a DenseMatrix");
    return false;
  }

  int nr=dm->nrows(); // time
  int nc=dm->ncols(); // number of electrode
  
  cerr << "Number of rows = " << nr << "  number of columns = " << nc << "\n";
  
  FILE *f = fopen(filename, "wt");
  //FILE *f =fopen("/home/sci/slew/ncrr/ref_from_scirun.msr","wt");
  if (!f) {
    pr->error(string("Error -- couldn't open output file: ") + filename );
    return false;
  }

  // write header
  fprintf(f,"NumberPositions=	%d\n",(unsigned)(nc));
  fprintf(f,"NumberTimeSteps=	%d\n",(unsigned)(nr));
  fprintf(f,"UnitTime	ms\n");
  fprintf(f,"UnitMeas	µV\n");
  fprintf(f,"ValuesTransposed\n");

  // write potential data in time
  for (int r=0; r<nr; r++) {
    fprintf(f,"%d: ",r+1);
    for (int c=0; c<nc; c++){
      fprintf(f, "%11.5e\t", (*dm)[r][c]);
    }
    fprintf(f, "\n");
  }

  fprintf(f,"Labels\n");
  /*
  for (int c=0; c<nc; c++){
    fprintf(f,"E%d ",c+1);
  }
  */
  // for testing model
  /*  fprintf(f,"FPz	Cz	Oz	Pz	Fz	T10	T7	C3	T8	C4	F3	F4	Fp1	Fp2	F7	F8	O1	O2	P3	P4	P7	P8	FT7	FT8	FC3	FC4	CP5	CP6	AFz	AF3	AF4	AF7	AF8	F5	F6	F9	F10	FC5	FC6	FT9	FT10	C5	C6	CPz	CP3	TP7	TP9	CP4	TP8	TP10	P5	P9	P6	P10	POz	PO3	PO7	PO4	PO8	FCz	F1	F2	FC1	FC2	C1	C2	CP1	CP2	P1	P2	T9	 \n");
   */

  // Only for epilepsy model
  fprintf(f, "CZ	A1	A2	A3	A4	A5	A6	A7	A8	A9	A10	A11	A12\n");
  fprintf(f, "A13	A14	A15	A16	A17	A18	A19	A20	A21	A22	A23	A24	B1\n");
  fprintf(f, "B2	B3	B4	B5	B6	B7	B8	B9	B10	B11	B12	B13	B14\n");
  fprintf(f, "B15	B16	B17	B18	B19	B20	B21	B22	B23	B24	B25	B26	B27\n");
  fprintf(f, "B28	B29	B30	B31	B32	B35	B36	B37	B38	B39	B40	B41	B42\n");
  fprintf(f, " B43	B44	B45	B46	B47	B48	STR1	STR2	STR3	STR4	STR5	STR6	STR7	STR8\n");
  
  
  fclose(f);

  return true;
}


static bool
send_result_file(MatrixOPort *positions_port, MatrixOPort *moments_port,
                 string filename)
{
  string extfilename = filename + ".dip";
  ifstream matstream(extfilename.c_str(),ios::in);
  if (matstream.fail()) {
    cerr << "Error -- Could not open file " << filename << "\n";
    return false;
  }

  string tmp;

  // skipping text lines - 
  for (int i=0; i<4; ++i){
    getline(matstream,tmp);
  }

  int pos, step; 
  matstream >> tmp >> pos;
  cerr << "Number of Dipole Position = " << pos << "\n";
  matstream >> tmp>> step;
  cerr << "Number of Time Step = " << step << "\n";

  // skipping text lines
  for (int i=0; i<4; ++i){
    getline(matstream, tmp);
  }

  // Get dipole position
  double px, py, pz;
  matstream >> tmp >> px >> py >> pz;

  // write dipole position to DenseMatrix
  DenseMatrix *dm_pos = scinew DenseMatrix(pos,3);

  (*dm_pos)[0][0] = px;
  (*dm_pos)[0][1] = py;
  (*dm_pos)[0][2] = pz;

  MatrixHandle pos_handle(dm_pos);
  positions_port->send(pos_handle);

  // Get dipople momnent
  getline(matstream, tmp);
  getline(matstream, tmp);
  double mx,my,mz;

  matstream >> tmp >> mx >> my >> mz;

  // write dipole vector to DenseMatrix
  DenseMatrix *dm_vec = scinew DenseMatrix(pos,3);

  getline(matstream, tmp);
  getline(matstream, tmp); // Magnitude
		
  // Get dipole magnitude & write dipole vector
  for (int r=0; r<step+1; r++){
    matstream >> tmp;
    double mag;
    matstream >> mag;
    // just for moment only
    (*dm_vec)[r][0] = mx;
    (*dm_vec)[r][1] = my;
    (*dm_vec)[r][2] = mz;

    // for vector moment with magnitude
    /*    (*dm_vec)[r][0] = mag*mx;
    (*dm_vec)[r][1] = mag*my;
    (*dm_vec)[r][2] = mag*mz; */
  }

  MatrixHandle vec_handle(dm_vec);
  moments_port->send(vec_handle);
	
  matstream.close();
 
  return true;
  
}

bool
InverseIPM::write_par_file(string filename)
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
  fprintf(f, "numchan= %d\n", (unsigned int) numchanTCL_.get());
  fprintf(f, "numsamples= %d\n", (unsigned int) numsamplesTCL_.get());
  fprintf(f, "startsample= %d\n", (unsigned int) startsampleTCL_.get());
  fprintf(f, "stopsample=  %d\n", (unsigned int) stopsampleTCL_.get());
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
  fprintf(f, "associativity= %d\n", (unsigned int) associativityTCL_.get());
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
  fprintf(f, "#define EXTRA           1       // tissue codes (see D1.2b)\n");
  fprintf(f, "#define SKULL           2\n");
  fprintf(f, "#define CSF             3\n");
  fprintf(f, "#define GREY_MATTER     4\n");
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
  fprintf(f, "1.0 0.33 0.0042 1.79 0.33 1.0 0.33 \n");
  fprintf(f, "# Labels in the head model corresponding to the different tissues\n");
  fprintf(f, "# The first value corresponds to the electrode which will be added to the mesh \n");
  fprintf(f, "labels=\n");
  fprintf(f, "1000 1 2 3 4 5 4\n");
  fprintf(f, "\n");
  fprintf(f, "# Tissue labels in the head model for which the tensor valued coductivity \n");
  fprintf(f, "# should be used if available\n");
  fprintf(f, "tensorlabels=\n");
  fprintf(f, "#0 0 1 0 0 1 0\n");
  fprintf(f, "0 0 0 0 0 0 0\n");
  fprintf(f, "\n");
  fprintf(f, "# -1- for sorted conductivities\n");
  fprintf(f, "sortedconductivities= 0\n");
  fprintf(f,  "# -1- for anisotropic conductivities\n");
  fprintf(f, "anisotropicconductivities= 0\n");
  fprintf(f, "\n");
  fprintf(f, "[InitialGuess]\n");
  fprintf(f, "numinitialguesspos= 1\n");
  fprintf(f, "posx=\n%5.3f\n", (double) posxTCL_.get());
  fprintf(f, "posy=\n%5.3f\n", (double) posyTCL_.get());
  fprintf(f, "posz=\n%5.3f\n", (double) poszTCL_.get());
  fprintf(f, "numinitialguessdir= 1\n");
  fprintf(f, "dirx=\n%5.3f\n", (double) dirxTCL_.get());
  fprintf(f, "diry=\n%5.3f\n", (double) diryTCL_.get());
  fprintf(f, "dirz=\n%5.3f\n", (double) dirzTCL_.get());
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

bool
InverseIPM::write_pebbles_file(string filename)
{
  FILE *f = fopen(filename.c_str(), "wt");
  if (f == NULL)
  {
    cerr << "Unable to open file"<< filename <<" for writing."<<"\n";
    return false;
  }

  fprintf(f,"EPS_PCCG\n");
  fprintf(f,"1e-8\n");
  fprintf(f,"EPS_ITER\n");
  fprintf(f,"1e-8\n");
  fprintf(f,"EPS_INT\n");
  fprintf(f,"1e-6\n");
  fprintf(f,"EPS_MAT\n");
  fprintf(f,"%.1e\n", (double) eps_matTCL_.get());
  fprintf(f,"ALPHA\n");
  fprintf(f,"0.01\n");
  fprintf(f,"BETA\n");
  fprintf(f,"0.0\n");
  fprintf(f,"SOLVER:_AMG=1,_Jacobi=2,_ILU=3\n");
  fprintf(f,"1\n");
  fprintf(f,"SOLUTION_STRATEGY:_ITER=1,_PCCG=2,_BiCGStab=3\n");
  fprintf(f,"2\n");
  fprintf(f,"PRECOND_STEP\n");
  fprintf(f,"1\n");
  fprintf(f,"COARSE_SYSTEM\n");
  fprintf(f,"1000\n");
  fprintf(f,"COARSE_SOLVER:_LR=1,_LL=2,_DIAG=3,_NAG=4\n");
  fprintf(f,"2\n");
  fprintf(f,"MAX_NUMBER_ITER\n");
  fprintf(f,"200\n");
  fprintf(f,"MAX_NUMBER_SIZE\n");
  fprintf(f,"2000000\n");
  fprintf(f,"MAX_NEIGHBOUR\n");
  fprintf(f,"100\n");
  fprintf(f,"COARSENING:_strong=1,_easy=2,_vmb=3\n");
  fprintf(f,"1\n");
  fprintf(f,"INTERPOLATION:_sophisticated=1,_easy=2,_foolproof=3\n");
  fprintf(f,"3\n");
  fprintf(f,"NORM:_energy=1,_L2=2,_Max=3\n");
  fprintf(f,"1\n");
  fprintf(f,"CYCLE:_V=1,_W=2,_VV=3\n");
  fprintf(f,"1\n");
  fprintf(f,"SMOOTHING_TYPE\n");
  fprintf(f,"1\n");
  fprintf(f,"SMOOTHING_STEP\n");
  fprintf(f,"1\n");
  fprintf(f,"ELEMENT_PRECOND\n");
  fprintf(f,"0\n");
  fprintf(f,"ELEMENT_KAPPA\n");
  fprintf(f,"10\n");
  fprintf(f,"MATRIX_DIFF\n");
  fprintf(f,"1e-1\n");
  fprintf(f,"SHOW_DISPLAY:_nothing=0,_medium=1,_full=2\n");
  fprintf(f,"1\n");
  fprintf(f,"CONV_FACTOR\n");
  fprintf(f,"0\n");

  fclose(f);
  return true;
}


void
InverseIPM::execute()
{
  FieldIPort *condmesh_port = (FieldIPort *)get_iport("CondMesh");
  FieldHandle condmesh;
  if (!(condmesh_port->get(condmesh) && condmesh.get_rep()))
  {
    warning("CondMesh field required to continue.");
    return;
  }

  if (condmesh->mesh()->get_type_description()->get_name() != "TetVolMesh" &&
      condmesh->mesh()->get_type_description()->get_name() != "HexVolMesh")
  {
    error("CondMesh must contain a TetVolMesh or HexVolMesh.");
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
  
  MatrixIPort *potential_port = (MatrixIPort *)get_iport("ElectrodePotentials");
  MatrixHandle potentials;
  
  if (!(potential_port->get(potentials) && potentials.get_rep()))
  {
    warning("Potentials required to continue.");
    return;
  }

  // Make our tmp directory
  //const string tmpdir = "/tmp/InverseIPM" + to_string(getpid()) +"/";
  const string tmpdir = "/tmp/InverseIPM/";
  const string tmplog = tmpdir + "inverse.log";
  const string resultfile = tmpdir + "inv_result";
  const string condmeshfile = tmpdir + "ca_head.geo";
  const string condtensfile = tmpdir + "ca_perm.knw";
  const string electrodefile = tmpdir + "electrode.elc";
  const string potentialsfile = tmpdir + "ref_potential.msr"; // TODO: fix
  const string parafile = tmpdir + "inverse.par";

  try {
    // Make our tmp directory.
    mode_t umsk = umask(00);
#ifndef _WIN32
    if (mkdir(tmpdir.c_str(), 0700) == -1)
#else
    if (mkdir(tmpdir.c_str()) == -1)
#endif
    {
      //error("Unable to open a temporary working directory.");
      //throw false;
      warning("Unable to open a temporary working directory.");
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

    // Write out Potentials
    if (!write_potential_file(this, potentials, potentialsfile.c_str()))
    {
      error("Unable to export potentials file.");
      throw false;
    }

    // Write out our parameter file.
    if (!write_par_file(parafile)) { throw false; }

    // write out pebble.inp
    if(!write_pebbles_file(tmpdir+"pebbles.inp")) {throw false; }
    
    // Construct our command line.
    const string ipmfile = ipm_pathTCL_.get();
    const string command = "(cd " + tmpdir + ";" +
      ipmfile + " -i movingdipolefit" +
      " -h " + condmeshfile + " -p " + parafile + " -s " + electrodefile +
      " -r " + potentialsfile + " -o " + resultfile + " -fwd FEM -sens EEG" +
      "-ndip 1 -opt Simplex -inv TruncatedSVD -guess Standard)";

    
    // Execute the command.  Last arg just a tmp file for logging.
    msgStream_flush();
    if (!Exec_execute_command(this, command, tmplog))
    {
      error("The ipm program failed to run for some unknown reason.");
      throw false;
    }
    msgStream_flush();

    // Read in the results and send them along.
    MatrixOPort *dip1_oport = (MatrixOPort *)get_oport("Dipole Positions");
    MatrixOPort *dip2_oport = (MatrixOPort *)get_oport("Dipole Moments");
    if (!send_result_file(dip1_oport, dip2_oport, resultfile))
    {
	error("Unable to send denseMatrix");
	throw false;
    }

    throw true; // cleanup.
  }
  catch (...)
  {
#if 0
    // Clean up our temporary files.
    unlink(condmeshfile.c_str());
    unlink(condtensfile.c_str());
    unlink(parafile.c_str());
    unlink(electrodefile.c_str());
    unlink(potentialsfile.c_str());
    unlink(resultfile.c_str());
    unlink(tmplog.c_str());
    rmdir(tmpdir.c_str());
#endif    
  }
}


void
InverseIPM::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun
