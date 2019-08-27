/*
 * read_rb.h
 *
 *  Created on: Sep 6, 2018
 *      Author: damodars
 *      read matrix from rb file. format described at https://sparse.tamu.edu/files/DOC/rb.pdf
 *      to be used for UofFs sparse matrix collection
 */

#ifndef SPARSEMATVEC_READ_RB_H_
#define SPARSEMATVEC_READ_RB_H_

#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <Kokkos_Core.hpp>

#include "simd.h"

//these are 1 d arrays. Layout wont make difference
typedef Kokkos::View<value_type*, Kokkos::LayoutRight, execution_space> viewtype;
typedef Kokkos::View<int*, Kokkos::LayoutRight, execution_space> intviewtype;
typedef Kokkos::View<value_type*, Kokkos::LayoutRight, execution_space>::HostMirror viewtype_mirror;
typedef Kokkos::View<int*, Kokkos::LayoutRight, execution_space>::HostMirror intviewtype_mirror;


struct spm {
	int m_num_of_rows, m_num_of_cols, m_num_of_entries;

	//csc format from file
	int * m_ptr_in_row_id{NULL};	//1 extra entry needed for column + 1.
	int * m_row_id{NULL};
	value_type * m_data{NULL};

	//csr to be used for cublas and all routines
	intviewtype m_ptr_in_col_id_gpu;	//1 extra entry needed for row + 1.
	intviewtype m_col_id_gpu;
	viewtype    m_csr_data_gpu;

	intviewtype_mirror m_ptr_in_col_id;	//1 extra entry needed for row + 1.
	intviewtype_mirror m_col_id;
	viewtype_mirror    m_csr_data;


	spm(int num_of_rows, int num_of_cols, int num_of_entries) : m_num_of_rows(num_of_rows), m_num_of_cols(num_of_cols), m_num_of_entries(num_of_entries)
	{
		m_ptr_in_row_id = new int[m_num_of_cols+1];
		m_row_id		= new int[m_num_of_entries];
		m_data			= new value_type[m_num_of_entries];

		m_ptr_in_col_id_gpu = intviewtype("m_ptr_in_col_id_gpu", m_num_of_rows+1);
		m_col_id_gpu 		= intviewtype("m_col_id_gpu", m_num_of_entries);
		m_csr_data_gpu		= viewtype("m_csr_data_gpu", m_num_of_entries);

		m_ptr_in_col_id		= mirror(m_ptr_in_col_id_gpu);
		m_col_id			= mirror(m_col_id_gpu);
		m_csr_data			= mirror(m_csr_data_gpu);
	}

	inline void csc2csr()
	{
		for (int i=0; i<=m_num_of_rows; i++) m_ptr_in_col_id[i] = 0;
		/* determine row lengths : first count elements in every row. then add up from previous row*/
		for (int i=0; i<m_num_of_entries; i++) m_ptr_in_col_id[m_row_id[i]+1]++;

		for (int i=0; i<m_num_of_rows; i++) m_ptr_in_col_id[i+1] += m_ptr_in_col_id[i];


		for(int i=0; i<m_num_of_cols; i++)
		{
			for(int j=m_ptr_in_row_id[i]; j<m_ptr_in_row_id[i+1]; j++)
			{
				int row = m_row_id[j];	//row_id
				int l = m_ptr_in_col_id[row]++;	//location to place col id and data in m_col_id and m_csr_data. increment location as it is filled
				const int col = i;
				m_col_id[l]   = col;
				m_csr_data[l] = m_data[j];
			}
		}

		/* shift back col_start. compensate incremented values, except last 1 which is needed as it is */
		for (int i=m_num_of_rows; i>0; i--) m_ptr_in_col_id[i] = m_ptr_in_col_id[i-1];

		m_ptr_in_col_id[0] = 0;

/*		printf("matrix in csr format: \n");
		for (int i=0; i<m_num_of_rows+1; i++)
			printf("%d ", m_ptr_in_col_id[i]);
		for (int i=0; i<m_num_of_entries; i++)
			printf("%d ", m_col_id[i]);
		for (int i=0; i<m_num_of_entries; i++)
			printf("%f ", m_csr_data[i]);
		printf("\n");
*/
		// copy to gpu
		Kokkos::deep_copy(m_ptr_in_col_id_gpu, m_ptr_in_col_id);
		Kokkos::deep_copy(m_col_id_gpu, m_col_id);
		Kokkos::deep_copy(m_csr_data_gpu, m_csr_data);

		delete []m_ptr_in_row_id;
		delete []m_row_id;
		delete []m_data;
	}

	__host__ __device__ spm(const spm&a)=default;
};


spm read_rb(const char * file_name)
{
	//int num_of_cols, num_of_rows, num_of_entries;
	int num_of_lines_for_m_ptr_in_row_id, num_of_lines_for_m_row_id, num_of_lines_for_m_data;

	std::ifstream file(file_name);
	std::string str, buf;

	//read header as per format to know dimensions

	//------------------------------------------------------------------------------------------------------------------------------------------------------
	/*Line 1 (A72, A8)
	     Col. 1 - 72   Title (TITLE)
	     Col. 73 - 80  Matrix name / identifier (MTRXID)*/

	std::getline(file, str);
	// Ignore first line. Just the matrix name. Nothing to do.

	//------------------------------------------------------------------------------------------------------------------------------------------------------
	/*Line 2 (I14, 3(1X, I13))
	     Col. 1 - 14   Total number of lines excluding header (TOTCRD)
	     Col. 16 - 28  Number of lines for pointers (PTRCRD)
	     Col. 30 - 42  Number of lines for row (or variable) indices (INDCRD)
	     Col. 44 - 56  Number of lines for numerical values (VALCRD)*/

	//VIMP: indexing of header format start from 1, while that of c / c++ from 0. Hence subtracting 1 from every index.
	// length of substring - add 1 because substring is including starting and ending character both. e.g. 28 - 16 + 1 = 13

	std::getline(file, str);
	num_of_lines_for_m_ptr_in_row_id = atoi(str.substr(15, 13).data());	// Col. 16 - 28  Number of lines for pointers (PTRCRD)
	num_of_lines_for_m_row_id = atoi(str.substr(29, 13).data());	// Col. 30 - 42  Number of lines for row (or variable) indices (INDCRD)
	num_of_lines_for_m_data = atoi(str.substr(43, 13).data());	// Col. 44 - 56  Number of lines for numerical values (VALCRD)

	//------------------------------------------------------------------------------------------------------------------------------------------------------
	/*Line 3 (A3, 11X, 4(1X, I13))
	     Col. 1 - 3    Matrix type (see below) (MXTYPE)
	     Col. 15 - 28  Number of rows (NROW)
	     Col. 30 - 42  Number of columns (NCOL)
	     Col. 44 - 56  Number of entries (NNZERO)
	     Col. 58 - 70  Unused, explicitly zero*/

	std::getline(file, str);
	const int num_of_rows = atoi(str.substr(14, 14).data());	// Col. 15 - 28  Number of rows (NROW)
	const int num_of_cols = atoi(str.substr(29, 13).data());
	const int num_of_entries = atoi(str.substr(43, 13).data());

	//------------------------------------------------------------------------------------------------------------------------------------------------------
	/*Line 4 (2A16, A20)
	     Col. 1 - 16   Fortran format for pointers (PTRFMT)
	     Col. 17 - 32  Fortran format for row (or variable) indices (INDFMT)
	     Col. 33 - 52  Fortran format for numerical values of coefficient matrix
	                   (VALFMT)
	                   (blank in the case of matrix patterns)*/

	std::getline(file, str);
	// Ignore 4th line. Just the matrix name. Nothing to do.

	//------------------------------------------------------------------------------------------------------------------------------------------------------

	printf("num_of_lines_for_m_ptr_in_row_id %d, num_of_lines_for_m_row_id %d, num_of_lines_for_m_data %d, num_of_rows %d, num_of_cols %d, num_of_entries %d\n",
			num_of_lines_for_m_ptr_in_row_id,    num_of_lines_for_m_row_id,    num_of_lines_for_m_data,    num_of_rows,    num_of_cols,    num_of_entries);
	fflush(stdout);
	//declare spm as per dimensions.
	spm M(num_of_rows, num_of_cols, num_of_entries);

	//read data from file into M

	//read m_ptr_in_row_id
	int j=0;
	for(int i=0; i<num_of_lines_for_m_ptr_in_row_id; i++)
	{
		std::getline(file, str);
		std::stringstream ss(str);
		while (ss >> buf)
			M.m_ptr_in_row_id[j++] = atoi(buf.data()) - 1;
	}

	//read num_of_lines_for_m_row_id
	j=0;
	for(int i=0; i<num_of_lines_for_m_row_id; i++)
	{
		std::getline(file, str);
		std::stringstream ss(str);
		while (ss >> buf)
			M.m_row_id[j++] = atoi(buf.data()) - 1;
	}

	//read num_of_lines_for_m_data
	j=0;
	for(int i=0; i<num_of_lines_for_m_data; i++)
	{
		std::getline(file, str);
		std::stringstream ss(str);
		while (ss >> buf)
			M.m_data[j++] = atof(buf.data());


	}

	M.csc2csr();

	return M;
}



void print(const spm& M)
{
	printf("m_ptr_in_row_id: ");
	for(int i=0; i<M.m_num_of_cols+1; i++)
			printf("%d ", M.m_ptr_in_row_id[i]);

	printf("\n\nm_row_id: ");
	for(int i=0; i<M.m_num_of_entries; i++)
			printf("%d ", M.m_row_id[i]);

	printf("\n\nm_data: ");
	for(int i=0; i<M.m_num_of_entries; i++)
			printf("%f ", M.m_data[i]);

	printf("\n");
}

#endif /* SPARSEMATVEC_READ_RB_H_ */


