
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

namespace CardioWave {

using namespace SCIRun;

class UnpackState : public Module {
public:
  UnpackState(GuiContext*);

  virtual void execute();
};


DECLARE_MAKER(UnpackState)
UnpackState::UnpackState(GuiContext* ctx)
  : Module("UnpackState", ctx, Source, "Tools", "CardioWave")
{
}


void UnpackState::execute()
{
  MatrixHandle State, StateOffset, Index;
  
  if (!(get_input_handle("State",State,true))) return;
  if (!(get_input_handle("StateOffset",StateOffset,true))) return;
  if (!(get_input_handle("Index",Index,true))) return;
  
  if ((State->ncols()!=1)&&(State->nrows()!=1))
  {
    error("UnpackState: State needs to be vector");
    return;
  }

  if ((StateOffset->ncols()!=1)&&(StateOffset->nrows()!=1))
  {
    error("UnpackState: StateOffset needs to be vector");
    return;
  }
  
  if ((Index->ncols()!=1)||(Index->nrows()!=1))
  {
    error("UnpackState: Index needs to be a scalar");
    return;
  }
   
  {
    MatrixHandle Temp;
    Temp = State; State = dynamic_cast<Matrix *>(Temp->dense()); 
    Temp = StateOffset; StateOffset = dynamic_cast<Matrix *>(Temp->dense()); 
    Temp = Index; Index = dynamic_cast<Matrix *>(Temp->dense()); 
  }
   
  double *sptr, *optr, *iptr;
  sptr = State->get_data_pointer();
  optr = StateOffset->get_data_pointer();
  iptr = Index->get_data_pointer();
  
  int nums, numo;
  nums = State->ncols()*State->nrows();
  numo = StateOffset->ncols()*StateOffset->nrows();
  
  MatrixHandle Output = dynamic_cast<Matrix *>(scinew DenseMatrix(numo,1));
  if (Output.get_rep() == 0)
  {
    error("UnpackState: Could not allocate new vector");
    return;  
  }
    
  double *outptr = Output->get_data_pointer();
  
  int s,e;
  int idx = static_cast<int>(iptr[0]);
  
  for (int p=0; p<numo; p++)
  {
    if (p < numo-1)
    {
      s = static_cast<int>(optr[p]);
      e = static_cast<int>(optr[p+1]);
    }
    else
    {
      s = static_cast<int>(optr[p]);
      e = nums;      
    }
    
    if (e-s > idx+1)
    {
      outptr[p] = sptr[s+idx];
    }
    else
    {
      outptr[p] = 0.0;
    }
  }
  
  send_output_handle("State",Output,true);
}


} // End namespace CardioWave


