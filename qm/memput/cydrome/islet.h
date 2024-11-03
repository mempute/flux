#pragma once

#include "../trace.h"
#include "psort.h"

class Cydrome;
class CydOrigin {
public:
	void *operator new(size_t size, Cydrome *cyd);
};
typedef struct _CydChain {
	intt sem_iden;
	struct _CydChain *ptrLeft, *ptrRight;
} CydChain;

class Cydrome {
public:
	QueryContext cydAllocator, *qccyd;
	intt sembase, nsemalc, isemalc;
	hmutex mucyd;
	CydChain *cydc_freed;
	ParallelSort *parallels;
	Cydrome(intx sem_base, intx nallocable)
	{
		nsemalc = nallocable;
		sembase = sem_base;
		isemalc = 0;
		qccyd = &cydAllocator;
		InitSelPage(qccyd);

		intt rv;
		CRE_MUT_(mucyd, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre mut fail");
		}
		cydc_freed = nullx;
		parallels = new ParallelSort(sem_base + nallocable, nallocable);
		parallels->bootParallels(40);
	}
	~Cydrome()
	{
		CLOSE_MUT_(mucyd);
		ReleaseSelPage(qccyd);
		delete parallels;
	}
	void *cydalloc(size_t size)
	{
		bytex *rp;

		SelAlloc(qccyd, size, rp);

		return rp;
	}
	CydChain *getcydid(void)
	{
		CydChain *cydc;

		LOCK_MUT_(mucyd);
		if(cydc_freed) {
			GET_LIST(cydc_freed, cydc);
		} else {
			cydc = (CydChain *)cydalloc(sizeof(CydChain));
			cydc->sem_iden = sembase + isemalc++;
			if(isemalc > nsemalc) {
				UNLOCK_MUT_(mucyd);
				throwFault(-1, "overflow sem\n");
			}
		}
		UNLOCK_MUT_(mucyd);
		return cydc;
	}
	void putcydid(CydChain *cydc)
	{
		LOCK_MUT_(mucyd);
		APPEND_LIST(cydc_freed, cydc);
		UNLOCK_MUT_(mucyd);
	}
};