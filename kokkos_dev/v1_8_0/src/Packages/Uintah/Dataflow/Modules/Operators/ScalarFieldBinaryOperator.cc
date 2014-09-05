#include <Packages/Uintah/Dataflow/Modules/Operators/ScalarFieldBinaryOperator.h>
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <iostream>

using namespace std;

//#include <SCICore/Math/Mat.h>

#define TYPES_MUST_MATCH 1

using namespace SCIRun;

namespace Uintah {
 
DECLARE_MAKER(ScalarFieldBinaryOperator)

ScalarFieldBinaryOperator::ScalarFieldBinaryOperator(GuiContext* ctx)
  : Module("ScalarFieldBinaryOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation"))
{
}
  
void ScalarFieldBinaryOperator::execute(void) {
  //  cout << "ScalarFieldBinaryOperator::execute:start"<<endl;
  
  in_left = (FieldIPort *) get_iport("Scalar Field Left Operand");
  in_right = (FieldIPort *) get_iport("Scalar Field Right Operand");
  sfout =  (FieldOPort *) get_oport("Scalar Field");

  FieldHandle left_FH;
  FieldHandle right_FH;
  
  if(!in_left->get(left_FH)){
    cerr<<"Didn't get a handle to left field\n";
    return;
  } else if ( left_FH->get_type_name(1) != "double" &&
	      left_FH->get_type_name(1) != "float" &&
	      left_FH->get_type_name(1) != "long64"){
    cerr<<"Left operand is not a Scalar field\n";
    return;
  }

  if(!in_right->get(right_FH)){
    cerr<<"Didn't get a handle to right field\n";
    return;
  } else if ( right_FH->get_type_name(1) != "double" &&
	      right_FH->get_type_name(1) != "float" &&
	      right_FH->get_type_name(1) != "long64"){
    cerr<<"Right operand is not a Scalar field\n";
    return;
  }

#ifdef TYPES_MUST_MATCH
  cout << "ScalarFieldBinaryOperator::execute:entering field setup.  Types must match.\n";
  
  if (left_FH->get_type_name(1) != right_FH->get_type_name(1)) {
    cerr <<"Types do not match!\n";
    cerr <<"type of left operand is "<<left_FH->get_type_name(1)<<"\n";
    cerr <<"type of right operand is "<<right_FH->get_type_name(1)<<"\n";
  }

  FieldHandle fh = 0;
  if( LatVolField<double> *scalarField_left =
      dynamic_cast<LatVolField<double>*>(left_FH.get_rep())) {
    
    // since it passed one of the types above it should get cast properly
    if ( LatVolField<double> *scalarField_right =
	 dynamic_cast<LatVolField<double>*>(right_FH.get_rep())) {

      LatVolField<double>  *scalarField_result =
	scinew LatVolField<double>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else {
      cerr <<"ScalarFieldBinaryOperator::execute: Error - right operand did not cast properly\n";
    }

  } else if( LatVolField<float> *scalarField_left =
	     dynamic_cast<LatVolField<float>*>(left_FH.get_rep())) {

    if ( LatVolField<float> *scalarField_right =
	 dynamic_cast<LatVolField<float>*>(right_FH.get_rep())) {

      LatVolField<float>  *scalarField_result =
	scinew LatVolField<float>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else {
      cerr <<"ScalarFieldBinaryOperator::execute: Error - right operand did not cast properly\n";
    }

  } else if( LatVolField<long64> *scalarField_left =
	     dynamic_cast<LatVolField<long64>*>(left_FH.get_rep())) {

    if ( LatVolField<long64> *scalarField_right =
	 dynamic_cast<LatVolField<long64>*>(right_FH.get_rep())) {

      LatVolField<long64>  *scalarField_result =
	scinew LatVolField<long64>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else {
      cerr <<"ScalarFieldBinaryOperator::execute: Error - right operand did not cast properly\n";
    }
  }

#else // ifdef TYPES_MUST_MATCH
  //cout << "ScalarFieldBinaryOperator::execute:entering field setup.  Types do not have to match.\n";
  
  FieldHandle fh = 0;
  if( LatVolField<double> *scalarField_left =
      dynamic_cast<LatVolField<double>*>(left_FH.get_rep())) {
    
    // since it passed one of the types above it should get cast properly
    if ( LatVolField<double> *scalarField_right =
	 dynamic_cast<LatVolField<double>*>(right_FH.get_rep())) {

      LatVolField<double>  *scalarField_result =
	scinew LatVolField<double>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else if ( LatVolField<long64> *scalarField_right =
		dynamic_cast<LatVolField<long64>*>(right_FH.get_rep())) {

      LatVolField<long64>  *scalarField_result =
	scinew LatVolField<long64>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else if ( LatVolField<float> *scalarField_right =
		dynamic_cast<LatVolField<float>*>(right_FH.get_rep())) {

      LatVolField<double>  *scalarField_result =
	scinew LatVolField<double>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else {
      cerr <<"ScalarFieldBinaryOperator::execute: Error - right operand did not cast properly\n";
    }

  } else if( LatVolField<float> *scalarField_left =
	     dynamic_cast<LatVolField<float>*>(left_FH.get_rep())) {

    // since it passed one of the types above it should get cast properly
    if ( LatVolField<double> *scalarField_right =
	 dynamic_cast<LatVolField<double>*>(right_FH.get_rep())) {

      LatVolField<double>  *scalarField_result =
	scinew LatVolField<double>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else if ( LatVolField<long64> *scalarField_right =
		dynamic_cast<LatVolField<long64>*>(right_FH.get_rep())) {

      LatVolField<long64>  *scalarField_result =
	scinew LatVolField<long64>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else if ( LatVolField<float> *scalarField_right =
		dynamic_cast<LatVolField<float>*>(right_FH.get_rep())) {

      LatVolField<float>  *scalarField_result =
	scinew LatVolField<float>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else {
      cerr <<"ScalarFieldBinaryOperator::execute: Error - right operand did not cast properly\n";
    }

  } else if( LatVolField<long64> *scalarField_left =
	     dynamic_cast<LatVolField<long64>*>(left_FH.get_rep())) {

    // since it passed one of the types above it should get cast properly
    if ( LatVolField<double> *scalarField_right =
	 dynamic_cast<LatVolField<double>*>(right_FH.get_rep())) {

      LatVolField<long64>  *scalarField_result =
	scinew LatVolField<long64>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else if ( LatVolField<long64> *scalarField_right =
		dynamic_cast<LatVolField<long64>*>(right_FH.get_rep())) {

      LatVolField<long64>  *scalarField_result =
	scinew LatVolField<long64>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else if ( LatVolField<float> *scalarField_right =
		dynamic_cast<LatVolField<float>*>(right_FH.get_rep())) {

      LatVolField<long64>  *scalarField_result =
	scinew LatVolField<long64>(left_FH->data_at());
      performOperation(scalarField_left, scalarField_right,
		       scalarField_result);
      fh = scalarField_result;

    } else {
      cerr <<"ScalarFieldBinaryOperator::execute: Error - right operand did not cast properly\n";
    }
  }
#endif // ifdef TYPES_MUST_MATCH
  
  if( fh.get_rep() != 0 )
    sfout->send(fh);
  //  cout << "ScalarFieldBinaryOperator::execute:end\n";
}

} // end namespace Uintah



