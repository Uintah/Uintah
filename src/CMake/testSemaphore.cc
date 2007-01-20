#include <semaphore.h>
int main()
{
  sem_t sem;
  sem_init(&sem, 0, 0);
  sem_post(&sem);
}
