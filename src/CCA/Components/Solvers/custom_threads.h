#ifndef custom_thread_h
#define custom_thread_h

//#define USE_FUNNELLED_COMM

//#define help_neighbor


#include<cmath>
#include<vector>
#include<omp.h>
#include<stdio.h>
#include<chrono>
#include <ctime>
#include<iostream>
#include<string.h>
#include<functional>
#include<atomic>
#include<mutex>
#include<condition_variable>
#include<sched.h>
#include<mpi.h>
#include<unistd.h>
#include<omp.h>

thread_local int do_setup=1;
void custom_partition_master(int b, int e, std::function<void(int)> f);
void custom_parallel_for(int b, int e, std::function<void(int)> f, int active_threads);

#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */
extern "C++" {
#endif

extern thread_local int cust_g_team_id, cust_g_thread_id;
extern int g_rank_temp;

int get_custom_team_id();
int get_custom_thread_id();
int get_team_size();
//int l1, l2;
//int patch_dim, number_of_patches, N;//, max_vec_rank;

void cpartition_master(int b, int e, void(*f)(int));
void cparallel_for(int b, int e, void(*f)(int), int active_threads);
void thread_init(int num_partitions, int threads_per_partition, int *affinity, int g_nodal_rank);
void destroy();
void wait_for_init(int p, int t, int *affinity, int g_nodal_rank);

#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */
}
#endif



double loop_time = 0.0;
int l1, l2;
volatile int spin=1;

#ifdef USE_FUNNELLED_COMM
thread_local int cust_g_team_id=-1, cust_g_thread_id=-1;
#else
thread_local int cust_g_team_id=0, cust_g_thread_id=0;
#endif



int get_custom_team_id()
{
	return cust_g_team_id;
}

int get_custom_thread_id()
{
	return cust_g_thread_id;
}

int get_team_size()
{
	return l2;
}


typedef struct thread_team
{
	std::function<void(int)> m_fun;	//size 32 bytes
	std::atomic<int> m_begin;	//beginning of team loop.	//size 4 bytes
	volatile int m_end;	//beginning of team loop.	//size 4 bytes
	std::atomic<int> m_completed{0};	//size 4 bytes
	volatile int m_num_of_calls{0};		//size 4 bytes
	volatile int m_active_threads{0};		//size 4 bytes
	char dummy[12];

} thread_team;


thread_team * g_team;



void team_worker(int team_id, int thread_id) //worker threads keep spinning. lightweight openmp
{
	int num_of_calls=0;
	thread_team *my_team = &g_team[cust_g_team_id];
//	int rank;
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//printf("worker team: %d, thread %d\n", team_id, thread_id);
	while(spin)
	{
		//printf("worker waiting %d-%d\n", team_id, thread_id);

				while(!my_team->m_fun && spin==1){
					asm("pause");
				}
				if(num_of_calls < g_team[cust_g_team_id].m_num_of_calls)
				{
					num_of_calls++;
//					printf("%d worker %d-%d active threads:%d\n",rank, team_id, thread_id, my_team->m_active_threads);
					if(thread_id < my_team->m_active_threads)
					{
						int e = my_team->m_end;
						int chunk = e - my_team->m_begin;
						chunk = chunk / l2 / 2;
						chunk = std::max(chunk, 1);
						int b;
						while((b = my_team->m_begin.fetch_add(chunk, std::memory_order_seq_cst))< e)
						{
							int b_next = std::min(e, b+chunk);
							for(; b<b_next; b++)
							{
								my_team->m_fun(b);
							}
						}
					}
					my_team->m_completed++;
				}


	#ifdef help_neighbor
				//help neighbor is not working with m_completed < l2 logic

				int t;
				for(t=(cust_g_team_id + 1)%l1; t != cust_g_team_id; t=(t+1)%l1)//iterate over all teams
				{

					if(g_team[t].m_begin < g_team[t].m_end &&
						  g_team[t].m_completed ==0 &&
						  g_team[t].m_fun!=NULL  ) //g_team[t].m_completed < 0 means work is in progress for the team. help!
					{
						//printf("%d-%d helping %d\n", team_id, thread_id, t);
						g_team[t].m_completed--;

						int e = g_team[t].m_end;
						int chunk = e - g_team[t].m_begin;
						chunk = chunk / l2 / 2;
						chunk = std::max(chunk, 1);
						int b;
						while((b = g_team[t].m_begin.fetch_add(chunk, std::memory_order_seq_cst))< e)
						{
							int b_next = std::min(e, b+chunk);
							for(; b<b_next; b++)
							{
								g_team[t].m_fun(b);
							}
						}
						g_team[t].m_completed++;

					}

					if(num_of_calls < g_team[cust_g_team_id].m_num_of_calls || spin==0)	//continue own work if available
						break;
				}
	#endif


	}
	//printf("team worker exited %d\n", team_id);
}

template<typename Function>
void custom_partition_master(int num_partitions, int threads_per_partition, Function f)
{
	spin=1;
	l1 = num_partitions;
	l2 = threads_per_partition;
	//printf("thread_init: %d\n", omp_get_thread_num());

	g_team = new thread_team[l1];

#ifdef USE_FUNNELLED_COMM
	int num_teams = l1+1, first_team = -1; //create 1 extra thread for comm at l1
#else
	int num_teams = l1, first_team = 0;
#endif

	std::atomic<int> completed=0;
	#pragma omp parallel num_threads(num_teams)
	{
		int team_id = omp_get_thread_num() + first_team; //offset -1 for funneled comm. compute threads reimain from 0 to n-1.

		//printf("%d threads %d %d on core %d\n",g_nodal_rank, team_id, thread_id, affinity[g_nodal_rank * l1*l2 + omp_thread_id]);

		if(team_id >= 0){//spawn l2 threads only for compute threads, not for comm thread
			#pragma omp parallel num_threads(l2)
			{
				int thread_id = omp_get_thread_num();

				cust_g_team_id = team_id;
				cust_g_thread_id = thread_id;

				if(thread_id==0){
					f(cust_g_team_id); //master thread calls the partition master fynction
					completed++;
					if(completed==l1) spin=0; // if all teams completed signal shutdown for worker threads
				}
				else
					team_worker(team_id, thread_id);	//workers spin waiting for parallel_for
			}
		}
		else
			f(-1);//this is comm thread
	}

//	delete []g_team;
	//printf("thread_init returning \n");
}


void custom_parallel_for(int s, int e, std::function<void(int)> f, int active_threads)
{
	thread_team *my_team = &g_team[cust_g_team_id];
	my_team->m_active_threads = active_threads;
	my_team->m_fun = f;
	std::atomic_thread_fence(std::memory_order_seq_cst);
	my_team->m_end = e;
	my_team->m_begin = s;
	my_team->m_completed = 0;
	my_team->m_num_of_calls++;


	int chunk = e - s;
	chunk = chunk / l2 / 2;
	chunk = std::max(chunk, 1);


	int b;
	while((b = my_team->m_begin.fetch_add(chunk, std::memory_order_seq_cst))< e)
	{
		int b_next = std::min(e, b+chunk);
		for(; b<b_next; b++)
		{
			my_team->m_fun(b);
		}
	}

	my_team->m_completed++;

	while(my_team->m_completed.load(std::memory_order_seq_cst)<l2)	//wait for execution to complete
	{
		asm("pause");
		//printf("waiting %d %d %d\n", g_completed.load(std::memory_order_seq_cst), g_begin.load(std::memory_order_seq_cst), g_end);
	}

	if(my_team->m_fun)
		my_team->m_fun=NULL;
	my_team->m_active_threads = 0;
	//printf("%d - %d %d: completed custom_parallel_for %d / %d\n",
	//		g_rank_temp, cust_g_team_id, cust_g_thread_id, my_team->m_completed.load(std::memory_order_seq_cst), l2);
}


void ccustom_partition_master(int num_partitions, int threads_per_partition, void(*f)(int))
{
	custom_partition_master(num_partitions, threads_per_partition, f);
}

void cparallel_for(int b, int e, void(*f)(int), int active_threads)
{
	custom_parallel_for(b, e, f, active_threads);
}


#endif
