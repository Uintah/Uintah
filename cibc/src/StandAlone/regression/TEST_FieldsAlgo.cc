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


#include <Core/Algorithms/DataIO/DataIOAlgo.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Algorithms/Regression/RegressionAlgo.h>
#include <Core/Util/RegressionReporter.h>
#include <Core/Util/Environment.h>


using namespace SCIRun;

void print_usage()
{
    std::cout << "USAGE: TEST_FieldsAlgo [test|generate] [testname]\n";
    std::cout << "       test|generate  : indicates whether regression files need to be generated or\n";
    std::cout << "                        whether the algorithms should be tested against known results.\n";
    std::cout << "       testname       : indicates which test needs to be run, if non specified all will\n";
    std::cout << "                        be run\n."; 
}


int main(int argc, char *argv[], char **environment) 
{

  create_sci_environment(environment,0);
  if (!sci_getenv("SCIRUN_ON_THE_FLY_LIBS_DIR"))
  {
    std::string objdir = sci_getenv("SCIRUN_OBJDIR");
    sci_putenv("SCIRUN_ON_THE_FLY_LIBS_DIR",objdir +"/on-the-fly-libs");
  }


  // Parse input arguments
  
  std::string mode = "";
  std::string test = "all";
  std::string testname = "";

  if ((argc < 2)||(argc > 3))
  {
    std::cerr << "ERROR: Improper number of input arguments\n\n";
    print_usage();
    exit (1);
  }

  std::string modestr(argv[1]);
  if (modestr == "generate") mode = "generate";
  if (modestr == "test") mode = "test";
  
  if (mode == "")
  {
    std::cerr << "ERROR: Mode needs to be 'test' or 'generate'\n\n";
    print_usage();
    exit (1);
  }

  if (argc == 3)
  {
    test = std::string(argv[2]);
  }

  const char* regressiondir = sci_getenv("SCIRUN_REGRESSION_DIR");
  if (regressiondir == 0)
  {
    std::cerr << "ERROR: No regression directory has been setup on this system\n";
    std::cerr << "  Please add the environment variable 'SCIRUN_REGRESSION_DIR'.\n";
    std::cerr << "  This directory contains the gold-standards for each of the individual tests.\n";
    std::cerr << "  Without this directory these tests cannot be performed.\n\n";
    print_usage();
    exit (1);
  }

  std::string testresult_dir = std::string(regressiondir) + "/test_fieldsalgo/";
  
  RegressionReporter *rr = scinew RegressionReporter("regression_fieldsalgo.log");
  
  SCIRunAlgo::RegressionAlgo ralgo(rr);
  SCIRunAlgo::FieldsAlgo     falgo(rr);
  SCIRunAlgo::DataIOAlgo     dalgo(rr);
  
  std::vector<FieldHandle> fields;
  std::vector<FieldHandle> testfields;
  
  if (!(ralgo.LoadTestFields(fields)))
  {
    std::cerr << "ERROR: Could not load test fields.\n";
    exit(1);
  }

  if (!(dalgo.CreateDir(testresult_dir)))
  {
    std::cerr << "ERROR: Could not create directory to store results.\n";
    exit(1);
  }
  

  try
  {
    
    std::cout << "\n -- TESTING THE FIELD ALGORITHMS -- \n";

    
    // -----------------------------------------------
    // Test - converttotetvol (FieldsAlgo.ConvertToTetVol)
    // -----------------------------------------------
      testname = "converttotetvol";

    // Check whether we should do this test
      if ((test == testname)||(test == "all"))
      {
        std::cout << "\nTESTING: CONVERTTOTETVOL\n";

        // Select the fields to test with
        if(!( ralgo.FindTestFields(fields,testfields,"latvol|hexvol|linear")))
        {
          std::cerr << "ERROR: Could not obtain any fields to test algorithm with.\n";
          exit (1);
        }
        
        // Loop through all the fields and test the algorithm
        for (unsigned int p=0; p < testfields.size(); p++)
        {
          FieldHandle input;
          FieldHandle  output, refoutput;

          input = testfields[p];

          // Define under which names we saved the gold standard
          std::string field_outputname = testresult_dir + testname + "-" + input->get_name() + ".fld";
          
          if (mode == "test")
          {
            // Check whether we have a gold standard for comparison
            if (!(dalgo.FileExists(field_outputname))) 
            {
              std::cout << "SKIPPING TEST: " << input->get_name() << " (no standard available)\n";
              continue;
            }
          }

          // Inform the user what we are doing
          std::cout << "TESTING: " << input->get_name() << "\n";
          
          // Test the algorithm
          if (!(falgo.ConvertToTetVol(input,output)))
          {
            std::cerr << "ERROR: ConvertToTetVol Algorithm failed.\n";
            exit (1);
          }
          
          if (mode == "generate")
          {
            // Save the gold standard
            if (output.get_rep())
            {
              if (!(dalgo.WriteField(field_outputname,output))) exit (1);
            }
          }
          else
          {
            // Retrieve the gold standard
            if (!(dalgo.ReadField(field_outputname,refoutput))) exit (1);

            // Test against the gold standard
            if ( !(ralgo.CompareFields(output,refoutput)) )
            {
              std::cerr << "ERROR: ConvertToTetVol output is not equal.\n";
              exit (1);
            }
          }
        }
      }


    // -----------------------------------------------
    // Test - converttotrisurf (FieldsAlgo.ConvertToTriSurf)
    // -----------------------------------------------
      testname = "converttotrisurf";

    // Check whether we should do this test
      if ((test == testname)||(test == "all"))
      {
        std::cout << "\nTESTING: CONVERTTOTRISURF\n";

        // Select the fields to test with
        if(!( ralgo.FindTestFields(fields,testfields,"quadsurf|image|linear")))
        {
          std::cerr << "ERROR: Could not obtain any fields to test algorithm with.\n";
          exit (1);
        }
        
        // Loop through all the fields and test the algorithm
        for (unsigned int p=0; p < testfields.size(); p++)
        {
          FieldHandle input;
          FieldHandle  output, refoutput;

          input = testfields[p];

          // Define under which names we saved the gold standard
          std::string field_outputname = testresult_dir + testname + "-" + input->get_name() + ".fld";
          
          if (mode == "test")
          {
            // Check whether we have a gold standard for comparison
            if (!(dalgo.FileExists(field_outputname))) 
            {
              std::cout << "SKIPPING TEST: " << input->get_name() << " (no standard available)\n";
              continue;
            }
          }

          // Inform the user what we are doing
          std::cout << "TESTING: " << input->get_name() << "\n";
          
          // Test the algorithm
          if (!(falgo.ConvertToTriSurf(input,output)))
          {
            std::cerr << "ERROR: ConvertToTriSurf Algorithm failed.\n";
            exit (1);
          }
          
          if (mode == "generate")
          {
            // Save the gold standard
            if (output.get_rep())
            {
              if (!(dalgo.WriteField(field_outputname,output))) exit (1);
            }
          }
          else
          {
            // Retrieve the gold standard
            if (!(dalgo.ReadField(field_outputname,refoutput))) exit (1);

            // Test against the gold standard
            if ( !(ralgo.CompareFields(output,refoutput)) )
            {
              std::cerr << "ERROR: ConvertToTriSurf output is not equal.\n";
              exit (1);
            }
          }
        }
      }




    // -----------------------------------------------
    // Test - fieldboundary (FieldsAlgo.FieldBoundary)
    // -----------------------------------------------
      testname = "fieldboundary";

      // Check whether we should do this test
      if ((test == testname)||(test == "all"))
      {
        std::cout << "\nTESTING: FIELDBOUNDARY\n";

        // Select the fields to test with
        if(!( ralgo.FindTestFields(fields,testfields,"surface|volume|linear")))
        {
          std::cerr << "ERROR: Could not obtain any fields to test algorithm with.\n";
          exit (1);
        }
        
        // Loop through all the fields and test the algorithm
        for (unsigned int p=0; p < testfields.size(); p++)
        {
          FieldHandle input;
          FieldHandle  output, refoutput;
          MatrixHandle mapping, refmapping;

          input = testfields[p];

          // Define under which names we saved the gold standard
          std::string field_outputname = testresult_dir + testname + "-" + input->get_name() + ".fld";
          std::string matrix_outputname = testresult_dir + testname + "-" + input->get_name() + ".mat";
          
          if (mode == "test")
          {
            // Check whether we have a gold standard for comparison
            if (!(dalgo.FileExists(field_outputname)) || !(dalgo.FileExists(matrix_outputname))) 
            {
              std::cout << "SKIPPING TEST: " << input->get_name() << " (no standard available)\n";
              continue;
            }
          }

          // Inform the user what we are doing
          std::cout << "TESTING: " << input->get_name() << "\n";
          
          // Test the algorithm
          if (!(falgo.FieldBoundary(input,output,mapping)))
          {
            std::cerr << "ERROR: FieldBoundary Algorithm failed.\n";
            exit (1);
          }
          
          if (mode == "generate")
          {
            // Save the gold standard
            if (output.get_rep() && mapping.get_rep())
            {
              if (!(dalgo.WriteField(field_outputname,output))) exit (1);
              if (!(dalgo.WriteMatrix(matrix_outputname,mapping))) exit (1);
            }
          }
          else
          {
            // Retrieve the gold standard
            if (!(dalgo.ReadField(field_outputname,refoutput))) exit (1);
            if (!(dalgo.ReadMatrix(matrix_outputname,refmapping))) exit (1);

            // Test against the gold standard
            if ( !(ralgo.CompareFields(output,refoutput)) ||
                 !(ralgo.CompareMatrices(mapping,refmapping)) )
            {
              std::cerr << "ERROR: FieldBoundary output is not equal.\n";
              exit (1);
            }
          }
        }
      }
    
        
    // -----------------------------------------------
    // Test - unstructure (FieldsAlgo.Unstructure)
    // -----------------------------------------------
      testname = "unstructure";

      // Check whether we should do this test
      if ((test == testname)||(test == "all"))
      {
        std::cout << "\nTESTING: UNSTRUCTURE\n";

        // Select the fields to test with
        if(!( ralgo.FindTestFields(fields,testfields,"regular|structured|linear")))
        {
          std::cerr << "ERROR: Could not obtain any fields to test algorithm with.\n";
          exit (1);
        }
        
        // Loop through all the fields and test the algorithm
        for (unsigned int p=0; p < testfields.size(); p++)
        {
          FieldHandle input;
          FieldHandle  output, refoutput;

          input = testfields[p];

          // Define under which names we saved the gold standard
          std::string field_outputname = testresult_dir + testname + "-" + input->get_name() + ".fld";
          
          if (mode == "test")
          {
            // Check whether we have a gold standard for comparison
            if (!(dalgo.FileExists(field_outputname))) 
            {
              std::cout << "SKIPPING TEST: " << input->get_name() << " (no standard available)\n";
              continue;
            }
          }

          // Inform the user what we are doing
          std::cout << "TESTING: " << input->get_name() << "\n";
          
          // Test the algorithm
          if (!(falgo.Unstructure(input,output)))
          {
            std::cerr << "ERROR: Unstructure Algorithm failed.\n";
            exit (1);
          }
          
          if (mode == "generate")
          {
            // Save the gold standard
            if (output.get_rep())
            {
              if (!(dalgo.WriteField(field_outputname,output))) exit (1);
            }
          }
          else
          {
            // Retrieve the gold standard
            if (!(dalgo.ReadField(field_outputname,refoutput))) exit (1);

            // Test against the gold standard
            if ( !(ralgo.CompareFields(output,refoutput)) )
            {
              std::cerr << "ERROR: Unstructure output is not equal.\n";
              exit (1);
            }
          }
        }
      }
    
    
    
    
    std::cout << "\n\n -- SUCCESSFULLY PASSED ALL TESTS --\n";
  }
  
  // CATCH ANY ERRORS
  catch (const Exception &e)
  {
    std::cerr << "TEST PROGRAM CRASHED WITH FOLLOWING ERROR:\n";
    std::cerr << " " << e.message() << "\n";
    if (e.stackTrace())
    {
      std::cerr << "STACK TRACE:\n";
      std::cerr << e.stackTrace() << "\n";
    }
    exit (1);
  }
  catch (const std::string a)
  {
    std::cerr << a << "\n";
    exit (1);
  }
  catch (const char* a)
  {
    std::cerr << std::string(a) << "\n";
    exit (1);
  }
  catch (...)
  {
    std::cerr << "PROGRAM CRASHED WITH NO GIVEN REASON\n";
    exit (1);
  }
}
