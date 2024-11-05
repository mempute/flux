
#include "islet.h"

void *CydOrigin::operator new(size_t size, Cydrome *cyd)
{
	return cyd->cydalloc(size);
}

Cydrome *loadCydrome(intx sem_base, intx nallocable)
{
	Cydrome *cyd = new Cydrome(sem_base, nallocable);
	return cyd;
}
void unloadCydrome(Cydrome *cyd)
{
	delete cyd;
}
