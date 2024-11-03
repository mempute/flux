
#include "psort.h"

void *SortOrigin::operator new(size_t size, SortAllocator *salc, ParallelSort *ps)
{
	if(salc) return salc->sortalloc(size);
	else return ps->psalloc(size, 1);
}

void TeleSort::tsInit(TeleSort *chief_ts)
{
	chiefTs = chief_ts;
	if(chiefTs) {
		LOCK_MUT_(chief_ts->semts->mutobj);
		chief_ts->itsMember++;
		/*if(chief_ts->imaxMember < chief_ts->itsMember) {
			chief_ts->imaxMember = chief_ts->itsMember;
			printf("gg %d\n", sortProsTs->iwaits);
		}*/
		UNLOCK_MUT_(chief_ts->semts->mutobj);
	} else {
		semts = sortProsTs->getsemps();
		itsMember = 1;
		imaxMember = 0;
	}
}
void TeleSort::tsReturn(void)
{
	if(chiefTs) {
		bool sig;
		LOCK_MUT_(chiefTs->semts->mutobj);
		sig = (--chiefTs->itsMember == 0 ? 1 : 0);
		UNLOCK_MUT_(chiefTs->semts->mutobj);
		if(sig) SIG_LSEM_(chiefTs->semts->semobj);
		chiefTs = nullx;
	} else {
		sortProsTs->putsemps(semts);
		semts = nullx;
	}
}
void TeleSort::chiefSig(void) //chief만 수행될때 동기 처리
{
	bool sig;
	LOCK_MUT_(semts->mutobj);
	sig = (--itsMember == 0 ? 1 : 0);
	UNLOCK_MUT_(semts->mutobj);
	if(sig) SIG_LSEM_(semts->semobj);
}
void TeleSort::rsort(void)
{
	sortProsTs->runSort(this);
}
void TeleSort::doneTeles(void)
{
	baseSort->putTeles(this);
}
thrret ThrParalls(thrarg arg)
{
	ParallelSort *ps = (ParallelSort *)arg;

	ps->runParallels();

	return 0;
}
void ParallelSort::bootParallels(intx num_thr)
{
	for(intx i = 0;i < num_thr; i++) {
		xthread_create((void *)ThrParalls, this);
	}
}