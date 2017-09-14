/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "vtkfile.hpp"

#include <vtkSmartPointer.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkXMLImageDataWriter.h>

Uintah::VtkFile::~VtkFile()
{
    if ( _grid )
    {
        _grid->Delete();
    }
}

void Uintah::VtkFile::set_grid ( const double & hx, const double & hy, const double & hz, const int & Nx, const int & Ny, const int & Nz, const double & x0, const double & y0, const double & z0 )
{
    if ( _grid )
    {
        _grid->Delete();
    }
    _grid = vtkImageData::New();
    _grid->SetSpacing ( hx, hy, hz );
    _grid->SetDimensions ( Nx, Ny, Nz );
    _grid->SetOrigin ( x0, y0, z0 );
}

void Uintah::VtkFile::add_time ( const double & time )
{
    vtkSmartPointer<vtkDoubleArray> array = vtkSmartPointer<vtkDoubleArray>::New();
    array->SetName ( "TIME" );
    array->SetNumberOfTuples ( 1 );
    array->SetTuple1 ( 0, time );
    _grid->GetFieldData()->AddArray ( array );
}

void Uintah::VtkFile::add_node_data ( const std::string & name, const constNCVariable<double> & var, const IntVector & low, const IntVector & high )
{
    const unsigned & n = _grid->GetNumberOfPoints();
    vtkSmartPointer< vtkDoubleArray> array = vtkSmartPointer<vtkDoubleArray>::New();

    double * tmp = new double[n];
    int l = 0;
    serial_for ( BlockRange ( low, high ), [&tmp, &var, &l] ( int i, int j, int k )->void { tmp[l++] = ( var ) ( i, j, k ); } );
    array->SetNumberOfComponents ( 1 );
    array->SetNumberOfTuples ( n );
    array->SetArray ( tmp, n, 0, 1 );
    array->SetName ( name.c_str() );
    _grid->GetPointData()->AddArray ( array );
}

void Uintah::VtkFile::add_cell_data ( const std::string & name, const constCCVariable<double> & var, const IntVector & low, const IntVector & high )
{
    unsigned n = _grid->GetNumberOfPoints() - 1;
    vtkSmartPointer< vtkDoubleArray> array = vtkSmartPointer<vtkDoubleArray>::New();

    double * tmp = new double[n];
    int l = 0;
    serial_for ( BlockRange ( low, high ), [&tmp, &var, &l] ( int i, int j, int k )->void { tmp[l++] = ( var ) ( i, j, k ); } );
    array->SetNumberOfComponents ( 1 );
    array->SetNumberOfTuples ( n );
    array->SetArray ( tmp, n, 0, 1 );
    array->SetName ( name.c_str() );
    _grid->GetCellData()->AddArray ( array );
}

void Uintah::VtkFile::add_cell_data ( const std::string & name, const constCCVariable<int> & var, const IntVector & low, const IntVector & high )
{
    unsigned n = _grid->GetNumberOfPoints() - 1;
    vtkSmartPointer< vtkDoubleArray> array = vtkSmartPointer<vtkDoubleArray>::New();

    double * tmp = new double[n];
    int l = 0;
    serial_for ( BlockRange ( low, high ), [&tmp, &var, &l] ( int i, int j, int k )->void { tmp[l++] = ( var ) ( i, j, k ); } );
    array->SetNumberOfComponents ( 1 );
    array->SetNumberOfTuples ( n );
    array->SetArray ( tmp, n, 0, 1 );
    array->SetName ( name.c_str() );
    _grid->GetCellData()->AddArray ( array );
}

void Uintah::VtkFile::save()
{
    vtkXMLImageDataWriter * writer = vtkXMLImageDataWriter::New();
    writer->SetFileName ( std::string ( _path + "/" + _name ).c_str() );
    writer->SetInputData ( _grid );
    writer->Write();
    writer->Delete();
}

