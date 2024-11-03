#pragma once

#if OPT_LINUX
#include <iconv.h>
#endif
#include "misc/mtree.h"
#include "qnet/qapi.h"
#include "misc/mut.h"
#include "misc/sem.h"
#include "misc/xf.h"

class SortAllocator {//사용자가 소팅 겍제 할당 관리 할려면 이 클래스를 오버라이드한다.
public:
	virtual void *sortalloc(size_t size) = 0;//소팅외에 다른 곳에서 동시에 호출되면 안된다.
};
class ParallelSort;
class SortOrigin {
public:
	void *operator new(size_t size, SortAllocator *salc, ParallelSort *ps);
};

typedef struct _SemChain {
	lsemx semobj;
	hmutex mutobj;
	struct _SemChain *ptrLeft, *ptrRight;
} SemChain;
class BaseSort;
class TeleSort : public SortOrigin {
public:
	bool psShutdown;
	intx itsMember, imaxMember;
	SemChain *semts;
	ParallelSort *sortProsTs;
	BaseSort *baseSort;
	TeleSort *ptrLeft, *ptrRight, *chiefTs;
	TeleSort *ptrLeft2, *ptrRight2;//사용측에서 사용.
	TeleSort(ParallelSort *ps, BaseSort *bs)
	{
		sortProsTs = ps;
		baseSort = bs;
		psShutdown = 0;
		semts = nullx;
		itsMember = 0;
	}
	virtual void sorting(void) {}
	virtual void settings(intx s, intx e, void *dc, void *sc = nullx) {}
	void tsInit(TeleSort *tsParent);
	void tsReturn(void);
	void chiefSig(void);
	void rsort(void);
	void wsort(void)
	{
		if(semts) WAIT_LSEM_(semts->semobj);
		doneTeles();
	}
	void doneTeles(void);
};

#define PSM_BASE	800000
class ParallelSort {
public:
	QueryContext psortAllocator, *qcpsort;
	TeleSort *qteles;
	intx sembps, nsemalc, isemaps, nruns, iwaits;
	hmutex mutps;
	lsemx semps;
	SemChain *sempool;
	ParallelSort(intx sem_base, intx nsems)
	{
		sembps = sem_base;
		nsemalc = nsems;
		isemaps = 0;
		sempool = nullx;
		qcpsort = &psortAllocator;
		InitSelPage(qcpsort);

		intx rv;
		CRE_MUT_(mutps, rv);
		if(rv < 0) {
			throwExpt(-1, (bytex *)"cre mut fail");
		}
		intx isem = sembps + isemaps++;
		CRE_LSEM_(semps, 0, isem, rv);
		if(rv < 0) {
			throwExpt(-1, (bytex *)"cre sem fail\n");
		}
		nruns = iwaits = 0;
	}
	~ParallelSort()
	{
		CLOSE_MUT_(mutps);
		for(SemChain *sc = sempool;sc; sc = sc->ptrRight) {
			CLOSE_MUT_(sc->mutobj);
			CLOSE_LSEM_(sc->semobj);
		}
		ReleaseSelPage(qcpsort);
	}
	void *psalloc(size_t size, bool block)
	{
		bytex *rp;

		if(block) LOCK_MUT_(mutps);
		SelAlloc(qcpsort, size, rp);
		if(block) UNLOCK_MUT_(mutps);

		return rp;
	}
	SemChain *getsemps(void)
	{
		SemChain *sc;

		LOCK_MUT_(mutps);
		if(sempool) {
			GET_LIST(sempool, sc);
			UNLOCK_MUT_(mutps);
			return sc;
		}
		sc = (SemChain *)psalloc(sizeof(SemChain), 0);
		intx rv;
		CRE_MUT_(sc->mutobj, rv);
		if(rv < 0) {
			throwExpt(-1, (bytex *)"cre mut fail");
		}
		intx isem = sembps + isemaps++;
		if(isemaps > nsemalc) {
			UNLOCK_MUT_(mutps);
			throwExpt(-1, (bytex *)"overflow sem\n");
		}
		CRE_LSEM_(sc->semobj, 0, isem, rv);
		if(rv < 0) {
			throwExpt(-1, (bytex *)"cre sem fail\n");
		}
		UNLOCK_MUT_(mutps);
		return sc;
	}
	void putsemps(SemChain *sc)
	{
		LOCK_MUT_(mutps);
		APPEND_LIST(sempool, sc);
		UNLOCK_MUT_(mutps);
	}
	void runSort(TeleSort *ts)
	{
		LOCK_MUT_(mutps);
		APPEND_LIST(qteles, ts);
		if(iwaits) SIG_LSEM_(semps);//요청 대기하고 있는 쓰레드가 있으면 시그널
		UNLOCK_MUT_(mutps);
	}
	void runParallels(void)
	{
		TeleSort *ts;

		while(1) {
			try {
				LOCK_MUT_(mutps);
				while(1) {
					if(qteles) {
						GET_LIST(qteles, ts);
						UNLOCK_MUT_(mutps);
						if(ts->psShutdown) return;//전채 종료
						ts->sorting();
						if(ts->chiefTs) ts->doneTeles();
						else ts->chiefSig();//chief는 호출측에서 반납하고 시그널 처리만
						LOCK_MUT_(mutps);
					} else {
						iwaits++;
						UNLOCK_MUT_(mutps);
						WAIT_LSEM_(semps);
						LOCK_MUT_(mutps);
						iwaits--;
					}
				}
			} catch(ExptObject eo) {
				printf("sort parallel error\n%s", eo.msg);
			}
		}
	}
	void bootParallels(intx num_thr);
};

class BaseSort {
public:
	SortAllocator *allocators;
	ParallelSort *sortProsBs;
	TeleSort *tspool;
	hmutex mutbs;

	virtual TeleSort *newTeles(void)
	{
		TeleSort *ts = new(allocators, sortProsBs) TeleSort(sortProsBs, this);
		return ts;
	}
	virtual void devideTeles(TeleSort *ts) {}
	//이 클래스 객제를 할당한 후 해제되지 않고 계속 사용하거나 테스트할때는 사용자 할당자 alc를 설정하지 않는다.
	BaseSort(ParallelSort *psort, SortAllocator *alc = nullx) 
	{
		allocators = alc;
		sortProsBs = psort;
		tspool = nullx;

		intx rv;
		CRE_MUT_(mutbs, rv);
		if(rv < 0) {
			throwExpt(-1, (bytex *)"cre mut fail");
		}
	}
	~BaseSort() //allocators가 설정되있으면 사용자측에서 자원관리를 하는 것이고 아니면 해제하지않고 게속 사용하는
	{			//것이니 소멸자에서 할일 없다.
	}
	TeleSort *getTeles(TeleSort *tsParent = nullx)
	{
		TeleSort *ts;

		LOCK_MUT_(mutbs);
		if(tspool) {
			GET_LIST(tspool, ts);
		} else ts = newTeles();
		UNLOCK_MUT_(mutbs);

		ts->tsInit(tsParent);

		return ts;
	}
	void putTeles(TeleSort *ts)
	{
		ts->tsReturn();
		LOCK_MUT_(mutbs);
		APPEND_LIST(tspool, ts);
		UNLOCK_MUT_(mutbs);
	}
};
//				이하 파생 클래스
template<typename DT> class QSort : public TeleSort {
public:
	DT *scomp, *dcomp;
	intx cstart, cend, nRecursion;
	bool ascending;

	QSort(ParallelSort *ps, BaseSort *bs, intx nrecurs, bool ascend) : TeleSort(ps, bs)
	{
		nRecursion = nrecurs;
		ascending = ascend;
	}
	void settings(intx s, intx e, void *dc, void *sc = nullx)
	{
		cstart = s; dcomp = (DT *)dc; scomp = (DT *)sc;
		if(sc) cend = e -1;//소팅 첫번째는 e가 길이로 설정되 있으므로 -1을 한다.
		else cend = e;
	}
	void recursion_sort(DT *dpart, intx spart, intx epart)
	{
		DT vpivot, v, v2, s2;
		intx i = spart * 2, j = epart * 2;
		//printf("bb %d %d %f %f\n", spart, epart, *(dpart + spart * 2), *(dpart + epart * 2));
		vpivot = *(dpart + i);//첫번째 값을 피봇으로 설정.
		s2 = *(dpart + i + 1);//인덱스
		if(ascending) {
			for(i += 2;;) {
				for(;*(dpart + i) <= vpivot && i <= j; i += 2);
				for(;vpivot <= *(dpart + j) && i <= j; j -= 2);
				if(i > j) break;//교차되면 탈룰
				v = *(dpart + i);
				v2 = *(dpart + i + 1);
				*(dpart + i) = *(dpart + j);
				*(dpart + i + 1) = *(dpart + j + 1);//교환
				*(dpart + j) = v;
				*(dpart + j + 1) = v2;
				i += 2; j -= 2;
			}
		} else {//descend
			for(i += 2;;) {
				for(;*(dpart + i) >= vpivot && i <= j; i += 2);
				for(;vpivot >= *(dpart + j) && i <= j; j -= 2);
				if(i > j) break;
				v = *(dpart + i);
				v2 = *(dpart + i + 1);
				*(dpart + i) = *(dpart + j);
				*(dpart + i + 1) = *(dpart + j + 1);
				*(dpart + j) = v;
				*(dpart + j + 1) = v2;
				i += 2; j -= 2;
			}
		}
		i = spart * 2;//첫번째 인덱스 설정, 
		*(dpart + i) = *(dpart + j);//j위치의 값과 첫번째 위치인 피봇값을 교환.
		*(dpart + i + 1) = *(dpart + j + 1);
		*(dpart + j) = vpivot;//피봇괎은 첫번째 위치값
		*(dpart + j + 1) = s2;

		j /= 2;//데이터:인덱스 pair를 데이터만의 인덱스로 변환
		if(j - spart > 1) recursion_sort(dpart, spart, j - 1);//피봇값을 제외한 좌측이 2개 이상이면 호출
		if(epart - j > 1) recursion_sort(dpart, j + 1, epart);//피봇값을 제외한 우측이 2개 이상이면 호출
	}
	void sorting(void)
	{
		DT vpivot, v, v2, s2;
		intx i, j;
		//printf("aa %d %d %f %f\n", cstart, cend, *(dcomp + cstart * 2), *(dcomp + cend * 2));
		if(scomp) {//소팅 첫번째는 소스의 인덱스는 데이터값 한개 단위이고 인덱싱(값이 위치하는 배열상의 순번)을 동시에 수행한다. 
			vpivot = *(scomp + cstart);//첫번째 값을 피봇으로 설정.
			s2 = cstart;//인덱스
			if(ascending) {
				for(i = cstart + 1, j = cend;;) {
					for(;*(scomp + i) <= vpivot && i <= j; i++) {
						*(dcomp + i * 2) = *(scomp + i);
						*(dcomp + i * 2 + 1) = i;//인덱싱
					}
					for(;vpivot <= *(scomp + j) && i <= j; j--) {
						*(dcomp + j * 2) = *(scomp + j);
						*(dcomp + j * 2 + 1) = j;
					}
					if(i > j) break;
					*(dcomp + i * 2) = *(scomp + j);//교환
					*(dcomp + i * 2 + 1) = j;
					*(dcomp + j * 2) = *(scomp + i);
					*(dcomp + j * 2 + 1) = i;
					i++; j--;
				}
			} else {//descend
				for(i = cstart + 1, j = cend;;) {
					for(;*(scomp + i) >= vpivot && i <= j; i++) {
						*(dcomp + i * 2) = *(scomp + i);
						*(dcomp + i * 2 + 1) = i;//인덱싱
					}
					for(;vpivot >= *(scomp + j) && i <= j; j--) {
						*(dcomp + j * 2) = *(scomp + j);
						*(dcomp + j * 2 + 1) = j;
					}
					if(i > j) break;
					*(dcomp + i * 2) = *(scomp + j);//교환
					*(dcomp + i * 2 + 1) = j;
					*(dcomp + j * 2) = *(scomp + i);
					*(dcomp + j * 2 + 1) = i;
					i++; j--;
				}
			}
		} else {//소팅 두번째 이후는 주어진 값만의 인덱스를 
			intx _cstart = cstart * 2, _cend = cend * 2;//값:인덱스 pair로 변환하여 수행
			i = ((cstart + cend) / 2) * 2;
			if(*(dcomp + _cstart) > *(dcomp + _cend)) {
				if(*(dcomp + i) > *(dcomp + _cstart)) i = _cstart;
				else if(*(dcomp + _cend) > *(dcomp + i)) i = _cend;
			} else if(*(dcomp + i) < *(dcomp + _cstart)) i = _cstart;
			if(i == _cstart) {
				vpivot = *(dcomp + _cstart);
				s2 = *(dcomp + _cstart + 1);
			} else {
				vpivot = *(dcomp + i);
				s2 = *(dcomp + i + 1);
				*(dcomp + i) = *(dcomp + _cstart);
				*(dcomp + i + 1) = *(dcomp + _cstart + 1);
				*(dcomp + _cstart) = vpivot;
				*(dcomp + _cstart + 1) = s2;
			}
			if(ascending) {//등호 빼도되고 어느 한쪽 또는 양측에 등호가 있으면된다. 어느쪽도 등호가 없으면 중복데이터 정렬할수 없다
				for(i = _cstart + 2, j = _cend;;) {
					for(;*(dcomp + i) <= vpivot && i <= j; i += 2);
					for(;vpivot <= *(dcomp + j) && i <= j; j -= 2);
					if(i > j) break;
					v = *(dcomp + i);
					v2 = *(dcomp + i + 1);
					*(dcomp + i) = *(dcomp + j);
					*(dcomp + i + 1) = *(dcomp + j + 1);
					*(dcomp + j) = v;
					*(dcomp + j + 1) = v2;
					i += 2; j -= 2;
				}
			} else {//descend
				for(i = _cstart + 2, j = _cend;;) {
					for(;*(dcomp + i) >= vpivot && i <= j; i += 2);
					for(;vpivot >= *(dcomp + j) && i <= j; j -= 2);
					if(i > j) break;
					v = *(dcomp + i);
					v2 = *(dcomp + i + 1);
					*(dcomp + i) = *(dcomp + j);
					*(dcomp + i + 1) = *(dcomp + j + 1);
					*(dcomp + j) = v;
					*(dcomp + j + 1) = v2;
					i += 2; j -= 2;
				}
			}
			j /= 2;//데이터:인덱스 pair를 데이터만의 인덱스로 변환
		}
		*(dcomp + cstart * 2) = *(dcomp + j * 2);//j위치의 값과 첫번째 위치인 피봇값을 교환.
		*(dcomp + cstart * 2 + 1) = *(dcomp + j * 2 + 1);
		*(dcomp + j * 2) = vpivot;//피봇괎은 첫번째 위치값
		*(dcomp + j * 2 + 1) = s2;

		if(j - cstart > 1) {
			if(j - cstart < nRecursion) recursion_sort(dcomp, cstart, j - 1);
			else {
				QSort<DT> *left_sort = (QSort<DT> *)baseSort->getTeles(chiefTs ? chiefTs : this);
				left_sort->settings(cstart, j - 1, dcomp);
				left_sort->rsort();
			}
		}
		if(cend - j > 1) {
			if(cend - j < nRecursion) recursion_sort(dcomp, j + 1, cend);
			else {
				QSort<DT> *right_sort = (QSort<DT> *)baseSort->getTeles(chiefTs ? chiefTs : this);
				right_sort->settings(j + 1, cend, dcomp);
				right_sort->rsort();
			}
		}
	}
};

template<typename DT> class BaseQSort : public BaseSort {
public:
	bool ascendOrder;
	intx nRecurs;

	BaseQSort(ParallelSort *psort, SortAllocator *alc = nullx, intx nrecurs = 32, bool ascend = 1) : BaseSort(psort, alc)
	{
		nRecurs = nrecurs;
		ascendOrder = ascend;
	}
	TeleSort *newTeles(void)
	{
		TeleSort *ts = new(allocators, sortProsBs)QSort<DT>(sortProsBs, this, nRecurs, ascendOrder);
		return ts;
	}
};
