/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  SimpleTextToTriSurf_IEPlugin.cc
 *
 *  Written by:
 *   Jeroen Stinstra
 *   Department of Computer Science
 *   University of Utah
 *
 *  Copyright (C) 2005 SCI Group
 */
 
#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Init/init.h>

#include <iostream>
#include <fstream>
#include <sstream>

namespace SCIRun {

FieldHandle SimpleTextFileToTriSurf_reader(ProgressReporter *pr, const char *filename)
{
	FieldHandle result = 0;
	
	std::string fac_fn(filename);
	std::string pts_fn(filename);
	
	
	// Check whether the .fac or .tri file exists
	std::string::size_type pos = fac_fn.find_last_of(".");
	if (pos == std::string::npos)
	{
		fac_fn = fac_fn + ".fac";
		try
		{
			std::ifstream inputfile;
			inputfile.open(fac_fn);						
		}
		
		catch (...)
		{
			if (pr) pr->error("Could not open file: "+fac_fn);
			return (result);
		}		
	}
	else
	{
		std::string base = fac_fn.substr(0,pos);
		std::string ext  = fac_fn.substr(pos);
		if ((ext != ".fac" )||(ext != ".tri"))
		{
			try
			{
				std::ifstream inputfile;		
				fac_fn = base + ".fac"; 
				inputfile.open(fac_fn);
			}
			catch (...)
			{
				try
				{
					std::ifstream inputfile;
					fac_fn = base + ".tri"; 
					inputfile.open(fac_fn);						
				}
				
				catch (...)
				{
					if (pr) pr->error("Could not open file: "+base + ".fac");
					return (result);
				}
			}
		}
		else
		{
			try
			{
				std::ifstream inputfile;
				inputfile.open(fac_fn);						
			}
			
			catch (...)
			{
				if (pr) pr->error("Could not open file: "+fac_fn);
				return (result);
			}				
		}
	}
	
	
	std::string::size_type pos = pts_fn.find_last_of(".");
	if (pos == std::string::npos)
	{
		fac_fn = fac_fn + ".pts";
		try
		{
			std::ifstream inputfile;
			inputfile.open(pts_fn);						
		}
		
		catch (...)
		{
			if (pr) pr->error("Could not open file: "+pts_fn);
			return (result);
		}		
	}
	else
	{
		std::string base = pts_fn.substr(0,pos);
		std::string ext  = pts_fn.substr(pos);
		if ((ext != ".pts" )||(ext != ".pos"))
		{
			try
			{
				std::ifstream inputfile;		
				pts_fn = base + ".pts"; 
				inputfile.open(pts_fn);
			}
			catch (...)
			{
				try
				{
					std::ifstream inputfile;
					pts_fn = base + ".pos"; 
					inputfile.open(pts_fn);						
				}
				
				catch (...)
				{
					if (pr) pr->error("Could not open file: "+base + ".pts");
					return (result);
				}
			}
		}
		else
		{
			try
			{
				std::ifstream inputfile;
				inputfile.open(pts_fn);						
			}
			
			catch (...)
			{
				if (pr) pr->error("Could not open file: "+pts_fn);
				return (result);
			}				
		}
	}	
	
	
  int ncols = 0;
  int nrows = 0;
  int line_ncols = 0;
	
  std::string line;
  double data;
   
  // STAGE 1 - SCAN THE FILE TO DETERMINE THE NUMBER OF NODES
  // AND CHECK THE FILE'S INTEGRITY.

	bool has_header = false;
	bool first_line = true;

  {
    std::ifstream inputfile;
    inputfile.open(pts_fn);

    while( getline(inputfile,line,'\n'))
    {
			if (line.size() > 0)
			{
				// block out comments
				if ((line[0] == '#')||(line[0] == '%')) continue;
			}
			
      // replace comma's and tabs with white spaces
      for (size_t p = 0;p<line.size();p++)
      {
        if ((line[p] == '\t')||(line[p] == ',')||(line[p]=='"')) line[p] = ' ';
      }
      std::istringstream iss(line);
      iss.exceptions( std::ifstream::failbit | std::ifstream::badbit);

      try
      {
        line_ncols = 0;
        while(1)
        {
          iss >> data;
          line_ncols++;
        }
      }
      catch(...)
      {
      }

			if (first_line)
			{
				if (ncols > 0)
				{
					if (line_ncols == 1)
					{
						has_header = true;
					}
					else if ((line_ncols == 3)||(line_ncols == 2))
					{
						has_header = false;
						first_line = false;
						nrows++;
						ncols = line_ncols; 
					}
					else
					{
						if (pr)  pr->error("Improper format of text file, some lines contain more than 3 entries");
						return (result);
					}
				}
			}
			else
			{
				if (line_ncols > 0)
				{
					nrows++;
					if (ncols > 0)
					{
						if (ncols != line_ncols)
						{
							if (pr)  pr->error("Improper format of text file, not every line contains the same amount of coordinates");
							return (result);
						}
					}
					else
					{
						ncols = line_ncols;
					}
				}
			}
    }
    inputfile.close();
  }

	int num_nodes = nrows;
	
	nrows = 0;
	ncols = 0;
	line_ncols = 0;
	
	bool zerro_based = false;
	
  {
    std::ifstream inputfile;
    inputfile.open(fac_fn);

    while( getline(inputfile,line,'\n'))
    {
			if (line.size() > 0)
			{
				// block out comments
				if ((line[0] == '#')||(line[0] == '%')) continue;
			}
			
      // replace comma's and tabs with white spaces
      for (size_t p = 0;p<line.size();p++)
      {
        if ((line[p] == '\t')||(line[p] == ',')||(line[p]=='"')) line[p] = ' ';
      }
      std::istringstream iss(line);
      iss.exceptions( std::ifstream::failbit | std::ifstream::badbit);

      try
      {
        line_ncols = 0;
        while(1)
        {
          iss >> data;
					
					if (data == 0) 
					{
					  zero_based = true;
					}
          line_ncols++;
        }
      }
      catch(...)
      {
      }

			if (first_line)
			{
				if (ncols > 0)
				{
					if (line_ncols == 1)
					{
						has_header = true;
					}
					else if (line_ncols == 3)
					{
						has_header = false;
						first_line = false;
						nrows++;
						ncols = line_ncols; 
					}
					else
					{
						if (pr)  pr->error("Improper format of text file, some lines do not contain 3 entries");
						return (result);
					}
				}
			}
			else
			{
				if (line_ncols > 0)
				{
					nrows++;
					if (ncols > 0)
					{
						if (ncols != line_ncols)
						{
							if (pr)  pr->error("Improper format of text file, not every line contains the same amount of coordinates");
							return (result);
						}
					}
					else
					{
						ncols = line_ncols;
					}
				}
			}
    }
    inputfile.close();
  }

	int num_elems = nrows;
	
	
	// Now create field
	typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;
	typedef GenericField<TSMesh, NoData<double>, vector<double> > TSField;
	
	TSMesh *mesh = scinew TSMesh();
	MeshHandle meshhandle = mesh;
	TSField *field = scinew TSField(); 
	result = field;

	mesh->node_reserve(num_nodes);
	mesh->elem_reserve(num_elems);
	
  {
    std::ifstream inputfile;

    try
    {
      inputfile.open(pts_fn);
    }
    catch (...)
    {
      if (pr) pr->error("Could not open file: "+std::string(pts_fn));
      return (result);
    }
    
		std::vector<double> vdata(3);
		int k = 0;
		
    while( getline(inputfile,line,'\n'))
    {
			if (line.size() > 0)
			{
				// block out comments
				if ((line[0] == '#')||(line[0] == '%')) continue;
			}		
		
      // replace comma's and tabs with white spaces
      for (size_t p = 0;p<line.size();p++)
      {
        if ((line[p] == '\t')||(line[p] == ',')||(line[p]=='"')) line[p] = ' ';
      }
      std::istringstream iss(line);
      iss.exceptions( std::ifstream::failbit | std::ifstream::badbit);
      try
      {
				k = 0;
        while(k <3)
        {
          iss >> data;
          vdata[k] = data;
					k++;
        }
      }
      catch(...)
      {
      }
			
			if (k == 3) mesh->add_point(Point(vdata[0],vdata[1],vdata[2]));
			if (k == 2) mesh->add_point(Point(vdata[0],vdata[1],0.0));
    }
    inputfile.close();
  }
	
  {
    std::ifstream inputfile;

    try
    {
      inputfile.open(fac_fn);
    }
    catch (...)
    {
      if (pr) pr->error("Could not open file: "+std::string(fac_fn));
      return (result);
    }
    
		unsigned int idata;
		typename TSMesh::Node::array_type vdata;
		vdata.resize(3);
		
		int k = 0;
		
    while( getline(inputfile,line,'\n'))
    {
			if (line.size() > 0)
			{
				// block out comments
				if ((line[0] == '#')||(line[0] == '%')) continue;
			}		
		
      // replace comma's and tabs with white spaces
      for (size_t p = 0;p<line.size();p++)
      {
        if ((line[p] == '\t')||(line[p] == ',')||(line[p]=='"')) line[p] = ' ';
      }
      std::istringstream iss(line);
      iss.exceptions( std::ifstream::failbit | std::ifstream::badbit);
      try
      {
				k = 0;
        while(k <3)
        {
          iss >> idata;
					if (zero_based) vdata[k] = idata;
					else vdata[k] = (idata-1);
					k++;
        }
      }
      catch(...)
      {
      }
			
			if (k == 3) mesh->add_elem(vdata);
    }
    inputfile.close();
  }	
	
	return (result);
}

static MatrixIEPlugin SimpleTextFileToTriSurf_plugin("TextFile","{.fac} {.tri} {.pts} {.pos}", "",SimpleTextFileToTriSurf_reader,0);

} // end namespace
