#pragma once

#include "misc/mtree.h"
#include "qnet/qapi.h"
#include "misc/mut.h"
#include "misc/xf.h"
#include "../trace.h"

class Cydrome;

class CydOrigin {
public:
	void *operator new(size_t size, Cydrome *cyd)
	{
		return cyd->cydalloc(size);
	}
};

class Cydrome {
public:
	QueryContext cydAllocator, *qccyd;

	Cydrome()
	{
		qccyd = &cydAllocator;
		InitSelPage(qccyd);
	}
	~Cydrome()
	{
		ReleaseSelPage(qccyd);
	}
	void *cydalloc(size_t size)
	{
		bytex *rp;

		SelAlloc(qccyd, size, rp);

		return rp;
	}
};