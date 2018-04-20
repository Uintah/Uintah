/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>

#include <cstring>
#include <iostream>
#include <map>

using namespace Uintah;

ProcessorGroup::ProcessorGroup( const ProcessorGroup * parent
                              ,       MPI_Comm         comm
                              ,       int              rank
                              ,       int              nRanks
                              ,       int              threads
                              )
  : m_rank(rank)
  , m_nRanks(nRanks)
  , m_threads(threads)
  , m_comm(comm)
  , m_parent_group(parent)
{
  m_node = -1;
  m_nNodes = 0;

  int resultlen;

  // For each rank get the processor name.
  MPI::Get_processor_name( m_proc_name, &resultlen );

  // Gather all of the processor names on the root node.
  char all_proc_names[nRanks*MPI_MAX_PROCESSOR_NAME+1];
  all_proc_names[nRanks*MPI_MAX_PROCESSOR_NAME] = '\0';
  
  MPI::Gather(  m_proc_name,  MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
	      all_proc_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

  char proc_name[MPI_MAX_PROCESSOR_NAME];

  // For the root node create a map that stores the unique processor
  // name.
  if( m_rank == 0 )
  {
    std::map< std::string, unsigned int > proc_name_map;

    // For each rank get its processor name.
    for( int i=0; i<nRanks; ++i )
    {
      // The rank's processor name.
      unsigned int cc = i * MPI_MAX_PROCESSOR_NAME;      
      strncpy( proc_name, &(all_proc_names[cc]), MPI_MAX_PROCESSOR_NAME);

      // Find the processor name in the map.
      if( proc_name_map.find(std::string(proc_name) ) == proc_name_map.end() )
      {
	// Store the name in the map for uniqueness.
	proc_name_map[ std::string(proc_name) ] = m_nNodes;

	// Store the processor name in an array for broadcasting.
	cc = m_nNodes * MPI_MAX_PROCESSOR_NAME;
	
	strncpy( &(all_proc_names[cc]), proc_name, MPI_MAX_PROCESSOR_NAME);
	
	++m_nNodes;	
      }	  
    }
  }    

  // From the root node broadcast the number of unique processor names.
  MPI::Bcast(&m_nNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // From the root node broadcast the unique processor names.
  MPI::Bcast(all_proc_names, m_nNodes*MPI_MAX_PROCESSOR_NAME,
	     MPI_CHAR, 0, MPI_COMM_WORLD);

  // Find the local processor name for this rank and get an associated
  // processor name index.
  for( int i=0; i<m_nNodes; ++i )
  {
    unsigned int cc = i * MPI_MAX_PROCESSOR_NAME;    
    strncpy( proc_name, &(all_proc_names[cc]), MPI_MAX_PROCESSOR_NAME);

    // Check the local processor name with the unique processor name
    // list, if a match store the index.
    if( std::string(m_proc_name) == std::string(proc_name) )
    {
      m_node = i;
      break;
    }
  }
  
  // Create a node based communicator. The node index becomes the
  // "color" while using the world rank which is unique for the
  // ordering.
  MPI::Comm_split(m_comm, m_node, m_rank, &m_node_comm);

  // Get the number of ranks for this node and the rank on this node.
  MPI::Comm_rank(m_node_comm, &m_node_rank);
  MPI::Comm_size(m_node_comm, &m_node_nRanks);
}

ProcessorGroup::~ProcessorGroup() 
{
  // MPI::Comm_free( &m_node_comm );
};


void ProcessorGroup::setGlobalComm(int num_comms) const
{
  if (m_threads <= 1) {
    return;
  }

  int curr_size = m_global_comms.size();
  if (num_comms <= curr_size) {
    return;
  }

  m_global_comms.resize(num_comms);
  for (int i = curr_size; i < num_comms; i++) {
    if (Uintah::MPI::Comm_dup(m_comm, &m_global_comms[i]) != MPI_SUCCESS) {
      std::cerr << "Rank: " << m_rank << " - MPI Error in Uintah::MPI::Comm_dup\n";
      Parallel::exitAll(1);
    }
  }
}
