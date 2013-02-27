#include <CCA/Components/Arches/SourceTerms/PCTransport.h>

using namespace std;
using namespace Uintah; 

PCTransport::PCTransport( std::string src_name, SimulationStateP& shared_state,
                      vector<std::string> req_label_names, std::string type ) 
: SourceTermBase( src_name, shared_state, req_label_names, type)
{
  _label_sched_init = false; 

  //Source Label
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 

}

PCTransport::~PCTransport()
{
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
PCTransport::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->require("PC_SCAL_FILE",_pc_scal_file);
  db->require("PC_ST_SCAL_FILE",_pc_st_scal_file);
  db->require("SVM_MODEL_BASE_NAME",_svm_base_name);
  db->require("N_PCS",_N_PCS ); 
  db->require("N_STS",_N_STS ); 
  db->require("N_IND",_N_IND ); 
  db->require("N_TOT",_N_TOT ); 

  ostringstream s_num; 
  for (int i=0; i < _N_PCS; i++){
    s_num << i; 
    _svm_models.push_back(_svm_base_name+s_num.str());
  }
 
//	///////////////////THINGS THAT NEED TO BE DONE ONLY ONCE///////////////////
//	// get index
//	int ROWi=N_IND; int COLi=1; int LENi=ROWi*COLi; double IND_I[LENi];
//	string file_indx;
//	file_indx=index;
//	load_mat(IND_I,ROWi,COLi,file_indx);
//	// get X_ave and gamma for obtaining state space
//	int ROWX=N_TOT; int COLX=1; int LENX=ROWX*COLX; double X_ave_Xe[LENX]; double gamma_Xe[LENX];
//	string file_scale_Xave;
//	string file_scale_gamma;
//	file_scale_Xave=X_ave_X;
//	file_scale_gamma=gamma_X;
//	load_mat(X_ave_Xe,ROWX,COLX,file_scale_Xave);
//	load_mat(gamma_Xe,ROWX,COLX,file_scale_gamma);
//	double gamma_SS[LENi]; double X_ave_SS[LENi];
//	for (int i=0; i<LENi+1;i++)
//		{
//			int ival=IND_I[i];
//			gamma_SS[i]=gamma_Xe[ival];
//			X_ave_SS[i]=X_ave_Xe[ival];
//		}
//
//	//get A mat
//	int ROWA=N_TOT; int COLA=N_PCS; int LENA=ROWA*COLA; double AMAT[LENA];
//	string file_AMAT;
//	file_AMAT=A_X;
//	load_mat(AMAT,ROWA,COLA,file_AMAT);
//	double AMAT_MAT[N_TOT][N_PCS];
//	int count1=-1;
//	for(int r=0; r<N_TOT; ++r){
//		for(int c=0; c<N_PCS; ++c){
//			count1++;
//			AMAT_MAT[r][c]=AMAT[count1];
//		}
//	}
//	
//	double AMAT_SS[N_IND*N_PCS];
//	int count2=-1;
//	for(int r=0; r<N_IND; ++r){
//		for(int c=0; c<N_PCS; ++c){
//			count2++;
//			int ival=IND_I[r];
//			
//			AMAT_SS[count2]=AMAT_MAT[ival][c];
//		}
//	}
//
//
//	
//	// get X_ave and gamma for centering and scaling PC's
//	int ROWS=N_PCS; int COLS=2; int LEN=ROWS*COLS; double scale_mat[LEN];
//	string file_scale1;
//	file_scale1=PC_SCAL_FILE;
//	load_mat(scale_mat,ROWS,COLS,file_scale1);
//
//	double scale_mat_M[ROWS][COLS];
//	count=-1;
//	for(int r=0; r<ROWS; ++r){
//		for(int c=0; c<COLS; ++c){
//			count++;
//			scale_mat_M[r][c]=scale_mat[count];
//		}}
//	double X_ave[ROWS];
//	double gamma[ROWS];
//	for(int r=0; r<ROWS; ++r){
//		X_ave[r]=scale_mat_M[r][0];
//		gamma[r]=scale_mat_M[r][1];
//	}
//	
//	// get X_ave_st and gamma_st for uncentering and unscaling PC Source terms
//	ROWS=N_STS; COLS=2; LEN=ROWS*COLS; scale_mat[LEN];
//	string file_scale2;
//	file_scale2=PC_ST_SCAL_FILE;
//	load_mat(scale_mat,ROWS,COLS,file_scale2);
//	scale_mat_M[ROWS][COLS];
//	count=-1;
//	for(int r=0; r<ROWS; ++r){
//		for(int c=0; c<COLS; ++c){
//			count++;
//			scale_mat_M[r][c]=scale_mat[count];
//		}}
//	double X_ave_st[ROWS];
//	double gamma_st[ROWS];
//	for(int r=0; r<ROWS; ++r){
//		X_ave_st[r]=scale_mat_M[r][0];
//		gamma_st[r]=scale_mat_M[r][1];
//	}
//	
//	// declare variables that will be used below
//	ROWS=1; COLS=N_PCS; LEN=ROWS*COLS;
//	double pc_test[LEN];                // this is the current pc_matrix (we need to know the size) 
//	double st_test[COLS];			// this is the source term matrix that is used with functions
//	
//	struct svm_model* model;
//	
//	svm_node x[N_PCS+1];						// this is the indexing for the model
//	
//	for (int i = 0; i < N_PCS; i++){
//		x[i].index=i+1;
//	}
//	x[N_STS].index=-1;

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
PCTransport::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "PCTransport::eval";
  Task* tsk = scinew Task(taskname, this, &PCTransport::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;

    tsk->computes(_src_label);

  } else {

    tsk->modifies(_src_label); 

  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
PCTransport::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

//	///////////////////THINGS THAT NEED TO BE AT EVERY TIME STEP///////////////////
//	
//	
//	////////////////// SVM SOURCE TERM SECTION
//	// GET THE CURRENT PC's  then center and scale
//	pc_test[0]=4.2644566061128297e-01;
//	pc_test[1]=1.1209306975793001e-02;
//	pc_test[2]=-1.9714699689318001e-02;
//
///////////////////Get STATE SPACE///////////////////////////////////
//
//	double STATESPACE[N_IND];
//	//X=A*Z'
//	cblas_dgemv(CblasColMajor, CblasTrans, N_PCS, N_IND, 1.0, AMAT_SS, N_PCS, pc_test, 1, 0, STATESPACE, 1);
//	//unscale
//	for (int unsc=0; unsc<N_IND; ++unsc){
//		STATESPACE[unsc]=STATESPACE[unsc]*gamma_SS[unsc];
//	}
//	//uncenter
//   	cout << "CURRENT STATE SPACE:" << endl;
//	for (int unc=0; unc<N_IND; ++unc){
//		STATESPACE[unc]=STATESPACE[unc]+X_ave_SS[unc];
//		printf("State Space Variable %1.0i: %1.10f\n",unc+1,STATESPACE[unc]);
//	}
//	cout << "\n";
//
////////////////////////////////////////////////////////////////////
//	
//	
//	// GET ST source terms
//	center(pc_test,ROWS,COLS,X_ave);
//	scale(pc_test,ROWS,COLS,gamma);
//	clock_t start, stop;
//	double t =0.0;
//	assert((start=clock())!=-1);
//	for (int i = 0; i < N_STS; i++){
//		x[i].value=(double) pc_test[i];
//	}
//	// GET ST source terms
//	for (int i = 0; i < N_STS; i++){
//		model=svm_load_model(SVM_MODELS[i].c_str()); 
//		//cout << SVM_MODELS[i].c_str()<<endl;
//		st_test[i] = svm_predict(model, x);
//	}
//	// Unscale and Uncenter the ST
//	unscale(st_test,ROWS,COLS,gamma_st);
//	uncenter(st_test,ROWS,COLS,X_ave_st);
//	
//	stop=clock();
//	t=double(stop-start)/CLOCKS_PER_SEC;
//	printf ("Time for Source Term Evaluation: %f",t);
//	cout << " seconds\n";
//	cout << "\n";
//	
//	
//	// to see variables
//	printf ("CURRENT SCORES: \n");
//	print_mat(pc_test,N_PCS,1);
//	cout << "\n";
//	printf ("CURRENT SCORE SOURCE TERMS: \n");
//	print_mat(st_test,N_PCS,1);
//	cout << "\n";
//    // add actual calculation of the source here. 

  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
PCTransport::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  //not needed
}
void 
PCTransport::dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw )
{
  //not needed
}

