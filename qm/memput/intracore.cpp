
#include "intracore.h"
#include "baper.h"

Flux *ApFork::forkout(void)
{
	Capsule *cap = new(apTcr)Capsule;
	bool b = TRCB(apTcr)->ebaper;
	TRCB(apTcr)->ebaper = 1;
	Flux *fxo = new(apTcr)Flux(apTcr, apIn->fshape, apIn->fdim, apIn->qType, memput::mp::apply, nullx, nullx);
	TRCB(apTcr)->ebaper = b;
	cap->vcaps = fxo;
	APPEND_LIST(apOuts, cap);
	nfanOut++;
	return fxo;
}

class ThreadTrack : public Tracker {
public:
	static Tracker *trkPool;
	Trace *trctrk;
	void *entryP;
	bool fwPropg, lockArrange;
	intt add_weight_multiple, iGround;
	void checkTcr(void *vp, Trace *tcr)
	{
		if(vp == nullx) return;
		if(*(sytet *)vp == GEN_T_FLUX) {
			if(((Flux *)vp)->fxTcr != tcr) {
				exit(1);
			}
		} else if(*(sytet *)vp == GEN_T_TACT) {
			if(((Apply *)((Contact *)vp)->vcontact)->apTcr != tcr) {
				exit(1);
			}
		} else if(*(sytet *)vp == GEN_T_CAP) {
			if(((Capsule *)vp)->vcaps->fxTcr != tcr) {
				exit(1);
			}
		}
	}
	void mSetTrack(void *fx, bool fwd, intt gid, Trace *trc)
	{
		entryP = fx;
		fwPropg = fwd;
		trctrk = trc;
		//checkTcr(entryP, trctrk);
		gpuSetting(-1, nullptr);//set gpu device 안하도록 널 설정. rdtei)에서 gpu focus실행안되게
		lockArrange = 0;
		iGround = gid;//seqGpuAlc실행이거나 배치확장후 첫 실행이 아니면 여기 gid설정 의미없다.
	}
	ThreadTrack(RunTrack *rt) : Tracker(rt)
	{
		trkType = 0;
	}
	intt nextgpu(intt gid, Trace *trc)
	{
		intt i = 0;
		for(; i < trc->numdid; i++) {
			if(*(trc->didIndice + i) == gid) break;
		}
		if(i == trc->numdid) throwFault(-1, "next gpu not found gid\n");
		if(++i == trc->numdid) i = 0;
		return *(trc->didIndice + i);
	}
	intt availgpu(intt gid, longt mcalc, Trace *trc)
	{
		longt sz_free = getmfreegpu(gid);//주어진 gpu에 할당량이 있으면 gpu id 리턴
		if(mcalc + trc->DEV_AVAIL_MARGIN < sz_free) return gid;

		intt i, n = 0;
		for(; n < trc->numdid; n++) {//gpu 검색
			if(*(trc->didIndice + n) == gid) break;
		}
		if(n == trc->numdid) throwFault(-1, "avail gpu not found gid\n");

		for(i = n + 1; i < trc->numdid; i++) {//검색된 gpu 인덱스 다음 gpu에 할당량 체크
			sz_free = getmfreegpu(*(trc->didIndice + i));
			//printf("avail device: %d %d\n", sz_free, i);
			if(mcalc + trc->DEV_AVAIL_MARGIN < sz_free) return *(trc->didIndice + i);
		}
		for(i = 0; i < n; i++) {//검색된 gpu 인덱스 이전 gpu에 할당량 체크
			sz_free = getmfreegpu(*(trc->didIndice + i));
			//printf("avail device: %d %d\n", sz_free, i);
			if(mcalc + trc->DEV_AVAIL_MARGIN < sz_free) return *(trc->didIndice + i);
		}
		throwFault(-1, "device memory lack, allocation fail req: %lld capa: %lld\n", mcalc, sz_free);
		return -1;
	}
	void freeGroundEqualShadow(Flux *operout, intt iground)
	{
		DMATFX(operout)->shadowFree(iground);//adgzb.쉐도우와 그라운드가 같으면 adgz)에서
		GMATFX(operout)->shadowFree(iground);//adgzb.할당한 쉐도우는 필요없으므로 해제
	}
	void applyDevArrange(Apply *oper, Trace *trc, Flux *origin, intt iground, intt n, longt &mcalc, intt i_arrange)
	{//oper 어플라이의 출력 그라운드에 모든 arrange되야하는 입력의 쉐도우와 출력 메모리를 arrange한다. 
		//n은 확장 배치사이즈
		if(mcalc >= 0) mcalc += oper->mcalcapp(n);//어플라이 내부 메모리 계산
		for(FxAnchor *fxa = oper->lapOuput; fxa; fxa = fxa->ptrRight) {
			if(mcalc >= 0) {//그라운드 메모리 집계, 출력은 동일한 gid 그라운드에 할당해야하므로 모두 집계
				mcalc += fxa->fxPoint->sizefx2(n);//내부에서 가중치와 같은 리사이저블이 아닌 것들은
												//배치 무시되고 본사이즈만 집계됨
			} else {//그라운드 매모리 재할당, 내부에서 사이즈가 작거나 그라운드 아이디가 다를 경우만
				//재할당됨. 리사이저블이 아닌 경우는 그라운드가 다를경우만 재할당됨, 어플라이의 내부
				//메모리는 어플라이 포워드 수행에서 레지스트 포토에의해 출력의 그라운드와 다를 경우에만
				//출력 그라운드에 재 할당됨.
				freeGroundEqualShadow(fxa->fxPoint, iground);
				//printf("333 %p %d %d\n", fxa->fxPoint, DMATFX(fxa->fxPoint)->didshadow, iground);
				//fxa->fxPoint->shape();
				//printf("alloc o-device %p %d %d %d %d\n", fxa->fxPoint, DMATFX_GID(fxa->fxPoint), iground, fxa->fxPoint->scaleout, oper->apCode);
				fxa->fxPoint->resizing5(n, iground);//adges.
				fxa->fxPoint->ofxArrange = i_arrange;
				//fxa->fxPoint->shape();
				//printf("%p 444 %p %p %d %d %d\n", this, oper, fxa->fxPoint, fxa->fxPoint->fxSize, iground, DMATFX_GID(fxa->fxPoint));
			}
			//printf("333 %p %d %d\n", fxa->fxPoint, oper->apCode, fxa->fxPoint->ofxArrange);
		}
		for(Capsule *cap = oper->lapInput; cap; cap = cap->ptrRight) {
			//다중참조되는 입력과 그 입력으로 출력되는 출력이 또 다른 다중 참조 입력을 입력으로 연결되는
			//체인상의 모든 입출력들을 그 출력 그라운드와 입력 쉐도우를 동시에 arrange하는데 그 출력들중에
			//첫번째 출력이 입력의 배치로 확장될때 arrange chain상의 모든 입력쉐도우와 출력 그라운드를
			//한꺼번에 arrange하므로 그래프 병렬실행에 의해 이들 체인상의 출력중 다른 하나가 나중에 어플라이에
			//의해 실행될때 이미 배치확장됐으므로 본 함수가 실행되지 않고 이 어플라이의 입력 쉐도우 또한
			//arrange가 된 상태이므로 문제없이 바로 arrangeDevice만 실행된다.
			if(mcalc >= 0) {//입력의 쉐도우 메모리 집계, 
				if(cap->vcaps->fxType == trainable) {//adcg.가중치당 할당되는 메모리 계산, 생성만
					//되고 참조되지 않는 가중치는 가중치 처리하는 모든곳에서 제외되므로 신결쓸것 없다.
					mcalc += (cap->vcaps->sizefx() * add_weight_multiple);
				} else mcalc += cap->vcaps->sizefx2(n);//리사이져블이면 배치반영, 아니면 본사이즈만 집계
				//가중치와 같은 리사이져블이 아닌데 집계된 메모리는 실제 할당의 ㄱ)에서 그라운드가 같으면
				//스킵되므로 메모리 낭비는 없다.
			} else {//나중에 const_apply관련 삭제, 아니면 아래 조건에 const_apply도 넣어야 함.
				if(DMATFX_GID(cap->vcaps) != iground || cap->vcaps->ofxArrange < trc->tcrArrange) {//ㄱ.
					//printf("alloc i-device %p %p %d %d %d %d\n", cap, cap->vcaps, DMATFX_GID(cap->vcaps), iground, cap->vcaps->scaleout, oper->apCode);
					intt sz;//입력의 shadow를 출력 iground에서 재할당
					if(cap->vcaps->scaleout > 0) sz = cap->vcaps->sizefx2(n) / 2;//data + grad 사이즈이므로 반으로 나누어 1개 사이즈로 전환
					else sz = cap->vcaps->sizefx();//배치 확장 없으므로 현재 사이즈
					if(cap->vcaps->fxType == trainable && cap->vcaps->ofxArrange < trc->tcrArrange) {
						if(DMATFX_GID(cap->vcaps) != iground) {//adar.가중치와 같이 배치확장 없고 어플라이의
							cap->vcaps->instTens(true, iground);// 출력이 아니면 현 그라운드에 재할당.
						}
						cap->vcaps->ofxArrange = i_arrange;
						//printf("%p apply dev arange %p %p %p %d %d %d %d\n", this, oper, cap, cap->vcaps, DMATFX_GID(cap->vcaps), iground, cap->vcaps->scaleout, oper->apCode);
					} else {
						//printf("%p apply dev arange2 %p %p %p %d %d %d %d\n", this, oper, cap, cap->vcaps, DMATFX_GID(cap->vcaps), iground, cap->vcaps->scaleout, oper->apCode);
						cap->allocShadow(sz, iground);//adgz.쉐도우 메모리 (재)할당
					}
				}
			}
			//printf("222 %p %p\n", cap->vcaps, origin);
			//printf("555++++ %p %p %d %d %d\n", cap->vcaps, oper, oper->apCode, iground, cap->vcaps->ofxArrange);
		}
	}
	void notResizeArrange(Apply *oper, Trace *trc, intt i_arrange)
	{//입력이 모두 리사이져블이 아닌경우 호출, 따라서 출력또한 리사이저블이 아니다.
		intt sz, iground, iGround = oper->oidground();
		longt done = 0, todo = oper->mcalcapp(-1);//어플라이 내부 메모리는 포워드 수행에서 할당되므로 할당될 메모리
		
		for(FxAnchor *fxa = oper->lapOuput; fxa; fxa = fxa->ptrRight) {//출력 메모리 집계
			//if(fxa->fxPoint->scaleout > 0) exit(1);
			//printf("1111 %p %p %d\n", oper, fxa->fxPoint, coi_arrangeoki);
			if(MATFX_INSTON(fxa->fxPoint)) done += fxa->fxPoint->sizefx2(-1);//할당된 메모리 집계
			else todo += fxa->fxPoint->sizefx2(-1);//메모리가 모잘라 빌드중 할당되지 못한 할당될 메모리 집계
			fxa->fxPoint->ofxArrange = i_arrange;
			//printf("444 %p %d %d\n", fxa->fxPoint, oper->apCode, fxa->fxPoint->ofxArrange);
		}
		if(todo) {//출력중 아직 할당되지 못한 매모리가 있으면 할당, 어플라이 내부 메모리는 포워드 실행에서 할당
			longt idone = 0;
			for(Capsule *cap = oper->lapInput; cap; cap = cap->ptrRight) {//입력 메모리 양 집계
				if(cap->vcaps->fxType == trainable) {//adcg.가중치당 할당되는 메모리 계산
					idone += (cap->vcaps->sizefx() * add_weight_multiple);
				} else idone += cap->vcaps->sizefx2(-1);//data + grad
			}//입력 메모리는 이시점에서 출력 그라운드가 고정되지 않기 때문에 모두 집계는 하지만 밑에서
			//그라운드가 다를 경우만 실제 할당되므로 메모리 낭비는 없다.
			iground = availgpu(iGround, todo + idone, trc);
			if(iground == iGround) {//출력 그라운드에 할당될 메모리 용량이 있으면 할당되지 못한 것만
				for(FxAnchor *fxa = oper->lapOuput; fxa; fxa = fxa->ptrRight) {//할당.
					freeGroundEqualShadow(fxa->fxPoint, iground);
					if(MATFX_INSTON(fxa->fxPoint) == 0) fxa->fxPoint->resizing5(-1, iground);//adges.
				}
			} else {//메모리 용량이 있는 다른 디바이스를 그라운드로 하여 메모리 할당, 출력의 그라운드는 모두
				iground = availgpu(iground, todo + done + idone, trc);//일치해야하므로 할당된 것들도
				for(FxAnchor *fxa = oper->lapOuput; fxa; fxa = fxa->ptrRight) {//다시 재할당
					freeGroundEqualShadow(fxa->fxPoint, iground);
					fxa->fxPoint->resizing5(-1, iground);//adges.
				}
			}
		} else iground = iGround;
		for(Capsule *cap = oper->lapInput; cap; cap = cap->ptrRight) {//출력그라운드와 다른
			//printf("%p aaa %p %p %d %d %d\n", this, oper, cap->vcaps, cap->vcaps->ofxArrange, DMATFX_GID(cap->vcaps), iground);
			if(DMATFX_GID(cap->vcaps) != iground || cap->vcaps->ofxArrange < trc->tcrArrange) {//ㄱ.
				sz = cap->vcaps->sizefx();
				if(cap->vcaps->fxType == trainable && cap->vcaps->ofxArrange < trc->tcrArrange) {
					if(DMATFX_GID(cap->vcaps) != iground) {//가중치와 같이 배치확장 없고 어플라이의
						cap->vcaps->instTens(true, iground);// 출력이 아니면 현 그라운드에 재할당.
					}
					cap->vcaps->ofxArrange = i_arrange;
					//printf("%p not resize arange %p %p %d %d %d %d\n", this, oper, cap->vcaps, DMATFX_GID(cap->vcaps), iground, cap->vcaps->scaleout, oper->apCode);
				} else {
					cap->allocShadow(sz, iground);//쉐도우 메모리 (재)할당
				}
			}
		}
		iGround = iground;
	}
	void forwardDevArrange(Apply *oper, Trace *trc)//포워드 실행전에 호출
	{//입력과 동일한 디바이스에 출력을 할당하고 메모리가 모잘라 할당할수없으면 다른 디바이스에 출력을 할당한다.
		intt n;
		longt mcalc;
		Capsule *cap;

		LOCK_MUT_(trc->mutArrange);//ffff
		lockArrange = 1;
		if(trc->tcrArrange > oper->lapOuput->fxPoint->ofxArrange) {//빌드후 처음 실행이면 수행
			for(cap = oper->lapInput; cap; cap = cap->ptrRight) {//입력중에서 스케일러블(배치확장된)
				if(cap->vcaps->scaleout > 0) break;//플럭스 선택
			}
			if(cap) n = cap->vcaps->scaleout;//(확장된)배치 갯수
			else n = -1;
			if(trc->gpugear == 0 && n < 0) {//gpugear이 true이면 배치확장되지 않더라도 그래프 분기될때
				//다른 gpu에서 병렬로 실행되기위해 그래프 분기에서 다음 gid가 설정되어 주어진 다음 gid에
				//재할당 해야 하므로 이 블럭을 실행하지 않는다.
				//어플라이의 입력이 가중치와 같은 배치 확장되지 않는 것만 구성(split같은 경우)
				//입력이 initArrange에 의해서만 초기화 경우 ofxArrange는 1이고 여기서 출력이 arrange
				//되어 1로 되면 다음 수행에서는 여기 arrange수행은 스킵되어 이후에 이 케이스는 다시 
				//실핼되는 일 없다. 배치확장과 관계없으므로 입력의 ofxArrange바뀔일 없기때문에. 후에
				//이 오퍼레이션의 입출력이 재할당되는 경우는 아래 배치확장 케이스의 수행중에 arrange 
				//chain에 의해 여기 입출력이 다시 재 arrange 되는 경우이다. 이때 이번에 일치된 
				//ofxArrange에 의해 스킵되는 일 없이 정상 arrange수행된다. 이유는 아래 케이스 수행의
				//경우는 feed에의해 ofxArrange가 2부터 시작되기때문에 여기서 1이 되어도 인덱스가 
				//다르므로 스킵되지 않고 다시 출력 그라운드에 arrange된다.
				notResizeArrange(oper, trc, trc->tcrArrange);
			} else {//cap->vcaps->ofxArrange는 trc->tcrArrange와 동일
				//printf("aaa %p %d %d\n", oper->lapOuput->fxPoint, oper->apCode, oper->lapOuput->fxPoint->ofxArrange);
				mcalc = 0;//arrange될 memmory량 계산 표시
				add_weight_multiple = 4 + (trc->mastTrc ? 0 : 1);//어플라이의 입력이 가중치일때
				//가중치당 iGround 그라운드에 부가적으로 생성되는 매트릭스 갯수 설정, 가중치의 data와 grad의
				//쉐도우 2개 + 아담 옵티마이저를 고려하여 가중치당 2개의 매트릭스 할당 + 수평분할 실행이고 
				//마스터이면 마스터와 슬레이브의 실행후 마스터에서 기울기 값을 더해야 하는데 로컬머신내 수평분할
				//실행 경우이면 마스터와 슬레이브의 그라운드가 다를때 마스터에 메모리를 할당 준비하여 슬레이브에
				//설정하고 슬레이브의 기울기 결과값을 복사할 마스터의 devDock 1개 혹은 분산수평분할 실행이면
				//마스터의 iGround에 매트릭스를 생성하여 슬레이브의 기울기 매트릭스를 복사할 것 1개(즉 로컬이던
				//분산이던 수평분할의 마스터 실행이면 1개 추가)하여 4 + 1(?)
				if(trc->gpugear == 0) {//어플라이의 arrange되야할 출력과 모든 입력의 쉐도우를
					iGround = oper->iidground();//일단 입력의 그라운드에서 재할당 할수있는지 체크
				}//else gpu병렬 실행을 지도하기위해 주어진 gid에 할당한다.
				//oper 어플라이 실행을 위해 모든 arrange되는 소요 메모리 용량 계산
				applyDevArrange(oper, trc, nullptr, iGround, n, mcalc, trc->tcrArrange);
				iGround = availgpu(iGround, mcalc, trc);
				mcalc = -1;//arrange될 memmory 할당 표시
				applyDevArrange(oper, trc, nullptr, iGround, n, mcalc, trc->tcrArrange);
				//printf("%p 222 %p %d %d\n", this, oper, oper->apCode, iGround);
			}
		}
		lockArrange = 0;
		UNLOCK_MUT_(trc->mutArrange);
	}
	void focusDevice(Apply *ap, TCxtArray *tcxt_arr)
	{
		if(tcxt_arr->focusDevGThr != iGround) {//어플라이 수행전에 출력의 그라운드 기계에 포커싱한다.
			//printf("focus device %d %d %d %p\n", ap->apCode, tcxt_arr->focusDevGThr, iGround, tcxt_arr);
			tcxt_arr->focusDevGThr = iGround;
			CudaDevSet(iGround);//본 함수는 그래프 수행 쓰레드와 장비 공유되는 시점에서만 호출된다.
		}
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		void *nextp = entryP;
		Apply *ap = nullx;
		Flux *fx = nullx;
		Contact *ct;
		Capsule *cap;
		ThreadTrack *ttrk;
		Matrixr *mtx_bwmut = nullx;
		bool ap_exec = false;
		//checkTcr(nextp, trctrk);
		if(fwPropg) {
			while(nextp) {
				switch(*(sytet *)nextp) {
				case GEN_T_FLUX:
					fx = (Flux *)nextp;
					nextp = nullx;
					if(fx->partitionfx) {
						ap = (Apply *)fx->fwAp->vcontact;
						if(trctrk->devArrange && trctrk->cpmMode <= 0) {
							forwardDevArrange(ap, trctrk);
						}
						iGround = ap->oidground();//출력 그라운드
						focusDevice(ap, tcxt_arr);
						ap->arrangeGround = iGround;
						ap_exec = 1;
						ap->forward(tcxt_arr->getgputxt(iGround, trctrk), mtx_bwmut);
						ap_exec = 0;
						ct = fx->fwAp->ptrRight;
					} else ct = fx->fwAp;
LB1:;
					for(;ct; ct = ct->ptrRight) {
						ap = (Apply *)ct->vcontact;
						if(ap->checkInFwap() == false || ap->apTcr != trctrk) {
							continue;//trptn.플럭스의 포워드 apply가 플럭스가 생성된 트레이서가 아닌 이 플럭스를 입력으로하는 
						}			//트레이서에서 생성된 어플라이이면 이 이후는 실행하지 않는다.
						if(ct->ptrRight) {//플럭스 다중 참조 분기 실행
							//checkTcr(ct->ptrRight, trctrk);
							//((Trace *)trctrk)->incFwFork();
							ttrk = (ThreadTrack *)rsc::brutra->trkGet2<ThreadTrack>();
							ttrk->mSetTrack(ct->ptrRight, true, trctrk->gpugear == 3 ? nextgpu(iGround, trctrk) : iGround, trctrk);//ㄴ.분기되므로 다음 그라운드에서 실행
							ttrk->ontrack(srGate);
						}
						if(nextp) {//ㄱ)에서 온것, split분기 실행
							//checkTcr(nextp, trctrk);
							//((Trace *)trctrk)->incFwFork();
							ttrk = (ThreadTrack *)rsc::brutra->trkGet2<ThreadTrack>();
							ttrk->mSetTrack(nextp, true, trctrk->gpugear > 1 ? nextgpu(iGround, trctrk) : iGround, trctrk);//분기되므로 다음 그라운드에서 실행
							ttrk->ontrack(srGate);
						}
						if(trctrk->devArrange && trctrk->cpmMode <= 0) {
							forwardDevArrange(ap, trctrk);
						}
						iGround = ap->oidground();//출력 그라운드
						focusDevice(ap, tcxt_arr);
						ap->arrangeGround = iGround;
						ap_exec = 1;
						nextp = ap->forward(tcxt_arr->getgputxt(iGround, trctrk), mtx_bwmut);//ㄷ.
						ap_exec = 0;
						//checkTcr(nextp, trctrk);
						break;
					}
					break;
				case GEN_T_TACT:
					ct = (Contact *)nextp;//다른 쓰레드의 ㄴ)에서 온것
					//checkTcr(nextp, trctrk);
					nextp = nullx;
					fx = nullx;
					goto LB1;
				case GEN_T_CAP://split, ㄷ)의 리턴
					cap = (Capsule *)nextp;
					nextp = cap->ptrRight;//ㄱ.
					//checkTcr(nextp, trctrk);
					ct = cap->vcaps->fwAp;
					goto LB1;
				}
			}
		} else {
			try {
				if(*(sytet *)nextp != GEN_T_CAP && ((Trace *)trctrk)->mastTrc != TRACER(((Flux *)nextp)->fxTcr)->mastTrc) {
					exit(1);
				}
				if(*(sytet *)nextp == GEN_T_CAP || ((Flux *)nextp)->checkInBwfx(1)) {//ㄴ)에서 온것이면
					while(nextp) {								//ㄹ)에처 체크되었고 시작 프럭스이면 여기서 체크
						if(*(sytet *)nextp == GEN_T_FLUX) {
							mtx_bwmut = nullx;
							/*if((trctrk != ((Apply *)((Flux *)nextp)->bwAp)->apTcr) ||
								(((Apply *)((Flux *)nextp)->bwAp)->apTcr != ((Flux *)nextp)->fxTcr)) {
								printf("2 %d %d %d %d\n", ((Trace *)trctrk)->idTrace,
									((Apply *)((Flux *)nextp)->bwAp)->apTcr->idTrace, TRACER(((Flux *)nextp)->fxTcr)->idTrace,
									((Apply *)((Flux *)nextp)->bwAp)->apCode);
								exit(1);
							}
							if(((Trace *)trctrk)->mastTrc != ((Apply *)((Flux *)nextp)->bwAp)->apTcr->mastTrc ||
								((Trace *)trctrk)->mastTrc != TRACER(((Flux *)nextp)->fxTcr)->mastTrc) {
								printf("2 %p %p %p %p %p %d\n", trctrk, ((Flux *)nextp)->fxTcr, ((Trace *)trctrk)->mastTrc,
									((Apply *)((Flux *)nextp)->bwAp)->apTcr->mastTrc, TRACER(((Flux *)nextp)->fxTcr)->mastTrc,
									((Apply *)((Flux *)nextp)->bwAp)->apCode);
								exit(1);
							}*/
							//printf("111 %d %d\n", ((Flux *)nextp)->ibRefer, ((Flux *)nextp)->nbRefer);
							fx = (Flux *)nextp;
							ap = (Apply *)((Flux *)nextp)->bwAp;
							iGround = ap->oidground();//출력 그라운드
							focusDevice(ap, tcxt_arr);
							ap->arrangeGround = iGround;
							ap_exec = 2;
							nextp = ap->backward(tcxt_arr->getgputxt(iGround, trctrk), mtx_bwmut);
							ap_exec = 0;
							//checkTcr(nextp, trctrk);
						} else {
							cap = (Capsule *)nextp;
							nextp = nullx;
							for(;cap; cap = cap->ptrRight) {
								//checkTcr(cap, trctrk);
								if(cap->vcaps->bwbreak || cap->vcaps->fxTcr != trctrk || cap->vcaps->checkInBwfx2() == false) continue;//ㄹ.
								//trptn.플럭스가 다른 트레이서에서 생성되어 현 트레이서에서 입력으로 사용하는 것이면 역전파 수행 스킵
								//tcfnr.bwbreak이면 역전파 수행 안함 추가
								if(cap->ptrRight) {
									//checkTcr(cap->ptrRight, trctrk);
									//((Trace *)trctrk)->incBwFork();
									ttrk = (ThreadTrack *)rsc::brutra->trkGet2<ThreadTrack>();
									ttrk->mSetTrack(cap->ptrRight, false, 0, trctrk);//ㄴ.백워드는 포워드에서 설정된 
									ttrk->ontrack(srGate);					//그라운드에서 실행하므로 그라운드 아니디 의미 없음
								}
								nextp = cap->vcaps;
								//checkTcr(nextp, trctrk);
								break;
							}
						}
					}
				}
			} catch(FaultObj fo) {
				printf("thread track error\n%s", fo.fltmsg);
				if(mtx_bwmut) UNLOCK_MUT_(mtx_bwmut->mutmtx);
				if(lockArrange) UNLOCK_MUT_(trctrk->mutArrange);
				if(ap_exec == 1) ap->multiArrangeUnlock();
				else if(ap_exec == 2 && ap->apCode == APC_CONCAT) ap->multiArrangeUnlock(0);
			}
		}
		srGate->srReturn();
	}
};
Tracker *ThreadTrack::trkPool = nullx;

class SlaveTrack : public Tracker {
public:
	static Tracker *trkPool;
	//TContext tcxtSlav;
	Trace *slavTrc;
	vector<Flux *> *tgtfx;
	void mSetTrack(Trace *slav_trc, vector<Flux *> *target)
	{
		slavTrc = slav_trc;
		tgtfx = target;
		//tcxtSlav.tcxttrc = slav_trc;
		gpuSetting(-1, nullptr);//set gpu device 안하도록 널 설정. rdtei)에서 gpu focus실행안되게
	}
	SlaveTrack(RunTrack *rt) : Tracker(rt)
	{
	}
	void tracking(TCxtArray *tcxt_arr)
	{
		vector<Flux *> target;
		LOCK_MUT_(slavTrc->mastTrc->mutTrc);//kkk
		for(auto tfx : *tgtfx) {
			target.push_back(slavTrc->hbaper->getflux(tfx->bregid));
		}
		UNLOCK_MUT_(slavTrc->mastTrc->mutTrc);
		for(Capsule *cap = slavTrc->mastTrc->trainWeight;cap; cap = cap->ptrRight) {
			Flux *slave_wfx = slavTrc->hbaper->getflux(cap->vcaps->bregid);//prelu의 경우 가중치값이 배치 사이즈를 
			if(cap->vcaps->fxSize != slave_wfx->fxSize) slave_wfx->resizing2(cap->vcaps, "slav");//포함하므로 사이즈 변경되면 반영한다.
			DMATFX(slave_wfx)->inCopy(DMATFX(cap->vcaps), 0);
		}//마스터 가중치 데이터를 슬레이브로 복사
		try {
			if(tcxt_arr->focusDevGThr < 0) {//슬레이브 쓰레드의 장비가 포커스되지 않은 초기 한번만
				//이번 슬레이브 마스터 trc의 메인장비아이디로 이번 슬래이브trc의 run과 슬레이브 쓰레드의
				slavTrc->focusgpudev(slavTrc->mastTrc->didTrace);//포커스 장비 아이디를 설정한다.
				tcxt_arr->focusDevGThr = slavTrc->didTrace;
			} else {//슬레이브 쓰레드의 장비가 포커스된 후에는 매번 바인딩된 슬래이브trc의 run함수 실행용
				slavTrc->didTrace = tcxt_arr->focusDevGThr;//포커스를 스레이브 쓰레드의 포커스로 
			}											//설정하여 이번 슬레이브 트랙을 시행한다.
			slavTrc->run(&target);
			LOCK_MUT_(slavTrc->mastTrc->mutMerge);
			if(slavTrc->trainOpt) {//슬레이브의 기울기와 마스터의 기울기를 더함.
				for(Capsule *cap = slavTrc->mastTrc->trainWeight;cap; cap = cap->ptrRight) {
					/*if(cap->vcaps->fxName[6] == 'c') {
						printf("------------------------------\n");
						cap->vcaps->printg();
					}*/
					Matrixr *gmst = GMATFX(cap->vcaps);
					Matrixr *gslv = GMATFX(slavTrc->hbaper->getflux(cap->vcaps->bregid));
					if(gmst->didground != gslv->didground) {//슬레이브와 마스터의 그래디언트 그라운드 
						gmst->dockAlloc();//아이디가 다르면 마스터에 도크를 생성하고(공간 확보는 adcg 
						gmst->devmcopy(gmst->devmpoint(3), gslv->devmpoint(0));//에서 계산 됨) 
						gslv->devmsetting(0, gmst->devmpoint(3));//슬레이브를 도크로 복사한후 
					}							//마스터의 도크를 슬레이브의 포커스로 설정한다.
					if(tcxt_arr->focusDevGThr != gmst->didground) {
						tcxt_arr->focusDevGThr = gmst->didground;
						CudaDevSet(tcxt_arr->focusDevGThr);//슬레이브 수행 쓰레드와 장비 공유된다.
					}
					gmst->marith(tcxt_arr->getgputxt(GMATFX_GID(cap->vcaps), slavTrc),
							gslv, gmst, nullx, 0, nullx, nullx, AOP_PLUS);
					/*if(cap->vcaps->fxName[6] == 'c') {
						Flux *sfx = slavTrc->hbaper->getflux(cap->vcaps->bregid);
						sfx->printg();
						printf("%s\n", sfx->fxName);
						cap->vcaps->printg();
						printf("------------------------------\n");
					}*/
				}
			}
			for(auto tfx : *tgtfx) {//슬레이브의 타겟값과 마스터의 타겟값을 더함.(그라프 실행 타겟 값)
				Matrixr *tmst = DMATFX(tfx);
				Matrixr *tslv = DMATFX(slavTrc->hbaper->getflux(tfx->bregid));
				if(tmst->didground != tslv->didground) {//위와 동일 처리, 타겟은 사이즈가 적으므로
					tmst->dockAlloc();//adcg)에서 공간 확보 없이 DEV_AVAIL_MARGIN내에서 수용된다.
					tmst->devmcopy(tmst->devmpoint(3), tslv->devmpoint(0));
					tslv->devmsetting(0, tmst->devmpoint(3));
				}
				if(tcxt_arr->focusDevGThr != tmst->didground) {
					tcxt_arr->focusDevGThr = tmst->didground;
					CudaDevSet(tcxt_arr->focusDevGThr);//슬레이브 수행 쓰레드와 장비 공유된다.
				}
				tmst->marith(tcxt_arr->getgputxt(DMATFX_GID(tfx), slavTrc),
					tslv, tmst, nullx, 0, nullx, nullx, AOP_PLUS);
			}
			//for(auto tfx : *tgtfx) {//슬레이브의 타겟값을 마스터로 복사
			//	TENSOR(tfx->quantum)->mxData->inCopy(TENSOR(slavTrc->hbaper->getflux(tfx->bregid)->quantum)->mxData, 0);
			//}
			UNLOCK_MUT_(slavTrc->mastTrc->mutMerge);
		} catch(FaultObj fo) {
			printf("slave track error\n%s", fo.fltmsg);
		}
		srGate->srReturn();
	}
};
Tracker *SlaveTrack::trkPool = nullx;

void Trace::installBaper(sytet stepw)
{
	hbaper = new Baper(this, stepw);
}
void Trace::freeBaper(void)
{
	delete hbaper;
}
void Trace::portingGraph(Tracer *target)
{
	TRACER(target)->hbaper->transgraph(hbaper);
}
Flux *Trace::getFlux(Flux *sfx)
{
	return hbaper->getflux(sfx->bregid);
}
Tracer *Trace::division(void)
{
	Trace *trc = new Trace(-1, nullx);
	trc->mastTrc = this;
	trc->cpmMode = cpmMode;
	trc->lapType = lapType;
	APPEND_LIST(lstSlav, trc);
	nslavTrc++;
	trc->hbaper->transgraph(hbaper);
	return trc;
}
void Trace::infeed(Flux *feed_fx, void *pdat, sytet tsrc, intt sz) //sz은 원소 갯수
{
	intt nbatch = sz / (feed_fx->fxSize / feed_fx->fshape[0]);
	intt bsz, rsz;

	nexeSlav = (nbatch % batchPart ? nbatch / batchPart + 1 : nbatch / batchPart) - 1;//슬레이브 갯수이므로 -1
	if(nslavTrc < nexeSlav) {
		for(intt i = nslavTrc;i < nexeSlav; i++) division();
	}
	ibatchSize = (nexeSlav ? nbatch : 0);
	bsz = (nbatch < batchPart ? nbatch : batchPart);

	if(feed_fx->fshape[0] < bsz && feed_fx->realwiden(bsz)) {
		tcrArrange = ++feed_fx->ofxArrange;//메모리 사이즈가 확장된다면 arrange수행되게 설정.
	}
	if(tsrc < 0) rsz = feed_fx->copyf2(pdat, bsz);//rsz은 이번에 적재된 원시 바이트 수
	else rsz = feed_fx->copyt(pdat, tsrc, bsz);//rsz은 이번에 적재된 원시 바이트 수

	if(nexeSlav == 0) return;//run함수 실행 전이거나 배치 갯수가 마스터 하나 만으로 실행될거면 스킵
	Flux *slav_fx;
	nbatch -= bsz;
	for(Trace *trc = lstSlav;trc && nbatch > 0; trc = trc->ptrRight, nbatch -= bsz) {//슬레이브 입력
		trc->nexeSlav = nexeSlav;
		bsz = (nbatch < batchPart ? nbatch : batchPart);
		slav_fx = trc->hbaper->getflux(feed_fx->bregid);
		trc->tcrArrange = slav_fx->ofxArrange = feed_fx->ofxArrange;
		if(tsrc < 0) rsz += slav_fx->copyf2((bytet *)pdat + rsz, bsz);
		else rsz += slav_fx->copyt((bytet *)pdat + rsz, tsrc, bsz);
	}
}
void Trace::initVsync(void)
{
	putCap(lorigin);
	putCap(trainWeight);
	putApc(lapply);
	lorigin = nullx;
	trainWeight = nullx;
	lapply = nullx;
	++bwVersion;
}
void Trace::listOrigin(Flux *ori)
{
	Capsule *oricap = getCap();
	oricap->vcaps = ori;
	APPEND_LIST(lorigin, oricap);
}
void Trace::initArrange(void)
{
	tcrArrange = 1;
	for(Capsule *cap = lorigin; cap; cap = cap->ptrRight) {
		cap->vcaps->ofxArrange = 1;
	}
}
void Trace::listWeight(void)
{
	Capsule *acap;

	for(Capsule *cap = lorigin;cap; cap = cap->ptrRight) {//가중치는 실행 그래프상 이전이 있을수 없으므로
		//오리진 리스트에 모두 리스팅돼있다. 이외 다른 트레이서의 출력 플럭스를 입력으로 하는 경우 tcvy)에서 오리진 리스팅되나
		if(cap->vcaps->fxType != memput::mp::trainable) continue;//가중치가 아니므로 여기서 스킵되어 상관없다.
		acap = getCap();
		acap->vcaps = cap->vcaps;
		APPEND_LIST(trainWeight, acap); 
		//printf("%p %s\n", acap->vcaps, acap->vcaps->fxName);
	}
	//printf("\n");
}
void Trace::vSync(Flux *endp) //name, anonym 모두 싱크된다.
{
	Capsule *cap, *cap2, *lcap = nullx, *lrecyc = nullx;
	Apply *ap = (Apply *)endp->bwLink;
	ApCap *apc;
	Flux *first = nullx;
	intt v = bwVersion;
	//msgdisp((bytet *)"zzzzzzzzzzzz\n");
	//if(TRACER(endp->fxTcr)->mastTrc != mastTrc) exit(1);
	if(endp->backwv != v) {
		endp->backwv = v;
		endp->nbRefer = 1;
		//msgdisp((bytet *)"111 %p %d\n", endp, endp->nbRefer);
		if(endp->quantum) TENSOR(endp->quantum)->mxGrad->nbackw = 1;
		//if(endp->quantum) printf("aaa %p %d\n", TENSOR(endp->quantum)->mxGrad, TENSOR(endp->quantum)->mxGrad->nbackw);
		//fnisqgt.endp->instTens(false);
	} else {
		if(endp->fxTcr == this) endp->nbRefer++;//trptn.관련 제거, 트레이서간에 플럭스를 직접 참조하지 않는다.
		//msgdisp((bytet *)"222 %p %d\n", endp, endp->nbRefer);
		if(endp->quantum) TENSOR(endp->quantum)->mxGrad->nbackw = endp->nbRefer;
		//if(endp->quantum) msgdisp((bytet *)"bbb %p %d\n", TENSOR(endp->quantum)->mxGrad, TENSOR(endp->quantum)->mxGrad->nbackw);
		return;
	}
	while(1) {
		for(;ap; ap = (first && first->bwbreak == 0 ? (Apply *)first->bwLink : nullx)) {//ap의 첫번째는 리스팅하지않고 다음에 바로 ap탐색한다.
			//printf("aaa: %d %p\n", ap->apCode, ap->apTcr);
			if(ap->apCode >= APC_ADMOPT && ap->apCode <= APC_SGDOPT) {
				trainOpt = ap;
				if(((ApOptimizer *)trainOpt)->optLoadStep == 1) ((ApOptimizer *)trainOpt)->optLoadStep++;//옵티마이져  
			}	//코드 로드 후 처음 한번만 로드 스템을 증가시킨다. 다시 이 코드가 실행될려면 코드로드(그래프 빌드)가 다시 되야한다.
			ap->nbfanOut++;
			//msgdisp((bytet *)"xxx %d %d\n", ap->apCode, ap->nbfanOut);
			//if(ap->apTcr->mastTrc != mastTrc) exit(1);
			if(ap->vbackW == v) break;
			apc = getApc();
			apc->vapply = ap;
			APPEND_LIST(lapply, apc);
			first = ap->lapInput->vcaps;//계속 first를 따라 depth로 진행한다.
			if(first->fxTcr == this) first->nbRefer++;//trptn.
			//msgdisp((bytet *)"333 %p %d\n", first, first->nbRefer);
			if(first->quantum) TENSOR(first->quantum)->mxGrad->nbackw = first->nbRefer;
			//if(first->quantum) msgdisp((bytet *)"ccc %p %d\n", TENSOR(first->quantum)->mxGrad, TENSOR(first->quantum)->mxGrad->nbackw);
			//if(TRACER(first->fxTcr)->mastTrc != mastTrc) exit(1);
			if(first->backwv != v) {
				first->backwv = v;
				first->nbRefer = 1;
				//msgdisp((bytet *)"444 %p %d\n", first, first->nbRefer);
				if(first->quantum) TENSOR(first->quantum)->mxGrad->nbackw = 1;
				//if(first->quantum) printf("ddd %p %d\n", TENSOR(first->quantum)->mxGrad, TENSOR(first->quantum)->mxGrad->nbackw);
				//fnisqgt.first->instTens(false);//ap의 첫번째는 아래 루프에서 실행되지않으므로 여기서 초기화.
				if(first->bwLink == nullx || first->fxTcr != this || first->bwbreak) {//trptn.플럭스가 현 tcr이 아닌 다른 tcr의 출력 
					listOrigin(first);//tcvy.최 선두만 리스팅하기위해 널체크 //플럭스를 입력으로 하는 것이며 역전파않고 선두로 리스팅.
					//if(first->fxTcr != this) printf("bbb: %p %p\n", this, first->fxTcr);
				}
			} else first = nullx;//추가 나중에 검증.
			ap->nbfanOut = 1;
			//printf("xxx1 %p %d %d %d\n", ap, ap->vbackW, ap->apCode, ap->nbfanOut);
			for(ap->vbackW = v, cap = ap->lapInput->ptrRight;cap; cap = cap->ptrRight) {
				if(cap->vcaps->fxTcr == this) cap->vcaps->nbRefer++;//trptn.
				//msgdisp((bytet *)"555 %p %d\n", cap->vcaps, cap->vcaps->nbRefer);
				if(cap->vcaps->quantum) TENSOR(cap->vcaps->quantum)->mxGrad->nbackw = cap->vcaps->nbRefer;
				//if(cap->vcaps->quantum) msgdisp((bytet *)"eee %p %d\n", TENSOR(cap->vcaps->quantum)->mxGrad, TENSOR(cap->vcaps->quantum)->mxGrad->nbackw);
				//if(TRACER(cap->vcaps->fxTcr)->mastTrc != mastTrc) exit(1);
				if(cap->vcaps->backwv == v) continue;
				if(cap->vcaps->bwLink == nullx || cap->vcaps->fxTcr != this || cap->vcaps->bwbreak) {//trptn.
					listOrigin(cap->vcaps);//tcvy.최 선두만 리스팅하기위해 널체크
					//if(cap->vcaps->fxTcr != this) printf("ccc: %p %p\n", this, cap->vcaps->fxTcr);
				}
				cap->vcaps->backwv = v;
				cap->vcaps->nbRefer = 1;
				//printf("666 %p %d\n", cap->vcaps, cap->vcaps->nbRefer);
				if(cap->vcaps->quantum) TENSOR(cap->vcaps->quantum)->mxGrad->nbackw = 1;
				//if(cap->vcaps->quantum) printf("fff %p %d\n", TENSOR(cap->vcaps->quantum)->mxGrad, TENSOR(cap->vcaps->quantum)->mxGrad->nbackw);
				//fnisqgt.cap->vcaps->instTens(false);
				cap2 = getCap();//여기서 할당되는 cap은 여기에서만 일회성으로 사용되므로 밑에서 lrecyc반환
				cap2->vcaps = cap->vcaps;
				APPEND_LIST(lcap, cap2);
			}
		}
		do {
			GET_LIST(lcap, cap);
		} while(cap && cap->vcaps->bwbreak);
		if(cap == nullx || cap->vcaps->fxTcr != this) break;//trptn.플럭스가 현 tcr이 아닌 다른 tcr의 출력 플럭스면 현 tcr은 그 이전은 실행안한다.
		ap = (Apply *)cap->vcaps->bwLink;
		APPEND_LIST(lrecyc, cap);//밑에서 해제하기위해 리스팅
	}
	putCap(lrecyc);

}
void Trace::resetApply(void)
{
	for(ApCap *apc = lapply;apc; apc = apc->ptrRight) {
		apc->vapply->ibfanOut = apc->vapply->ifanIn = 0;
	}
}
void Trace::init_train(void)
{
	if(loadWeight()) return;

	for(Capsule *cap = trainWeight;cap; cap = cap->ptrRight) {
		if(cap->vcaps->vinitter) {
			cap->vcaps->vinitter(cap->vcaps);
			//printf("1: %p: %s\n", cap->vcaps, cap->vcaps->fxName);
		}
	}
}
void Trace::run(Flux *target)
{
	vector<Flux *> list_f;
	list_f.push_back(target);
	run(list_f);
}
void Trace::run(vector<Flux *> target)
{
	run(&target);
}
//훈련이 아닌 평가 단계 실행이면 배치로 실행하면 배치 분할 쓰레드 출력값을 합하므로 입력에 매칭되는 출력값을 얻을수 없으므로
//평가단계에서 배치로 실행하면 안된다.
void Trace::run(vector<Flux *> *target) 
{
	intt i = 0, n = target->size();
	Flux *fp = lstTarget, *cur;
	unit lap = (lapType == 3 ? xucurrenttime() : 0);

	sidTrace = didTrace;
	for(;i < n && fp; i++, fp = fp->ptrRight) {
		cur = target->at(i);
		if(fp != cur || cur->backwv != bwVersion) break;
	}
	if(i < n || fp) {//타겟 리스트가 변경됐으면 초기화 수행.
		lstTarget = nullx;
		trainOpt = nullx;
		initVsync();
		for(auto fxp : *target) {
			APPEND_LIST(lstTarget, fxp);
			vSync(fxp);
		}
		listWeight();//위 v sync후에 해야함
		if(mastTrc == nullx) {//마스터 trc에서만 실행.
			if(trainOpt) {
				if(((ApOptimizer *)trainOpt)->optLoadStep > 1) {//옵티마이져 코드가 로드된후 실행 타겟 패스에 
					((ApOptimizer *)trainOpt)->optLoadStep = 0;//옵티마이져가 있으면 훈련 상황이므로 코드 로드후 처음 
					init_train();								//한번만 가중치를 초기화 시킨다.
				}
			} else loadWeight();//학습과정이 아니면 가중치 로드
		}
		hbaper->baperEnding(1);
	}
	if(nexeSlav) {
		if(mastTrc == nullx) {//마스터 trc이면
			mastsr = rsc::prutra->srGet();
			Trace *trc = lstSlav;
			SlaveTrack *strk;
			for(i = 0;i < nexeSlav; i++, trc = trc->ptrRight) {
				strk = (SlaveTrack *)rsc::prutra->trkGet2<SlaveTrack>();
				strk->mSetTrack(trc, target);//배치 분할된 입력 및 타겟으로 각 분할 배치 슬레이브 포워드(run함수) 실행.
				strk->ontrack(mastsr);
			}
			LOCK_MUT_(mutMerge);//마스터의 실행이 끝나기 전에 슬레이브에서 먼저 기울기 add 하는 것을 방지
		} else {
			mastsr = mastTrc->mastsr;
		}
	} else mastsr = nullx;
	//ifwFork = ibwFork = 0;
	SignalR *sr = rsc::brutra->srGet();//srGet2()//mmm
	ThreadTrack *ttrk;
	//rsc::brutra->onSingle(true);
	resetApply();
	if(lapType == 3) printf("run#1 lap: %f\n", (xucurrenttime() - lap) / 1000000.0);lap = xucurrenttime();
	for(Capsule *cap = lorigin;cap; cap = cap->ptrRight) {
		//if(mastTrc == nullx) continue;
		ttrk = (ThreadTrack *)rsc::brutra->trkGet2<ThreadTrack>();
		ttrk->mSetTrack(cap->vcaps, true, didTrace, this);
		ttrk->ontrack(sr);
	}
	sr->srWait();
	//chkFwFork(sr);
	sr->srReset2();//kkk
	if(lapType == 3) printf("run#2 lap: %f\n", (xucurrenttime() - lap) / 1000000.0);lap = xucurrenttime();
	if(trainOpt == nullx) {//옵티마이저가 실행 안됨.
		sr->srPut<ThreadTrack>();
		if(nexeSlav && mastTrc == nullx) {//배치분할 실행이고 마스터이면 슬레이브가 종료할때까지 대기
			UNLOCK_MUT_(mutMerge);
			mastsr->srWait();
			mastsr->srPut<SlaveTrack>();
		}
		goto LB1;//optimizer forward가 실행되지 않았으면 백워드 과정 실행 않는다.
	}
	//rsc::brutra->onSingle(false);
	resetGrads();//이하 백워드 수행
	//if(lapType == 3) printf("run#3 lap: %lld\n", xucurrenttime() - lap);lap = xucurrenttime();
	for(auto fxp : *target) {
		//if(mastTrc == nullx) continue;
		ttrk = (ThreadTrack *)rsc::brutra->trkGet2<ThreadTrack>();
		ttrk->mSetTrack(fxp, false, 0, this);//백워드 실행은 gid 의미 없다.
		ttrk->ontrack(sr);
	}
	sr->srWait();
	//chkBwFork(sr);
	//printf("%d: %d %d\n", idTrace, ibwFork, nbwFork);
	//if(lapType == 3) printf("run#4 lap: %f\n", (xucurrenttime() - lap) / 1000000.0);lap = xucurrenttime();
	sr->srPut<ThreadTrack>();//mmm
	if(nexeSlav) {//배치분할 실행이고
		if(mastTrc) return;//슬래이브이면 이하 수행않고 리턴
		else {//마스터이면 슬레이브가 종료할때까지 대기
			UNLOCK_MUT_(mutMerge);
			mastsr->srWait();
			mastsr->srPut<SlaveTrack>();
			/*floatt f = nexeSlav + 1;//가중치 평균 구하기
			for(Capsule *cap = trainWeight;cap; cap = cap->ptrRight) {
				TENSOR(cap->vcaps->quantum)->mxGrad->marith(trcCxt(), nullx, TENSOR(cap->vcaps->quantum)->mxGrad, 
					nullx, cap->vcaps->qType, &f, nullx, AOP_DIV);
			}*/
			//for(Capsule *cap = trainWeight;cap; cap = cap->ptrRight) {//클리핑
			//	TENSOR(cap->vcaps->quantum)->mxGrad->mclip(trcCxt(), TENSOR(cap->vcaps->quantum)->mxGrad, -5.0, 5.0);
			//}
		}
	}
	((ApOptimizer *)trainOpt)->update(tcxtarr, this);

	if(nexeSlav) {//배치분할 실행이면 평균값 적용 플랙스들은 평균을 산출한다.
		for(auto fxp : *target) {//타겟 값들은 슬레이브 트랙에서 각 슬레이브 리턴 종료전에 sum되고 
			if(fxp->meanAfter) {//타겟들이 그래프상의 mean함수 이후의 플랙스에 해당하면 슬레이브 갯수로 평균 산출.
				TENSOR(fxp->quantum)->mxData->mmean(nexeSlav + 1);
			}
		}
	}
LB1:;
	//if(didTrace != sidTrace) {//슬레이브는 여기까지 실행되지 않으므로 해당없다.
	//	didTrace = sidTrace;
	//	CudaDevSet(didTrace);
	//}
	if(lapType == 3) printf("run#5 lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
}

intt scoopeout_size(bool scoop_inner, intt seqy, intt seqx, intt slidey, intt slidex, intt stridey, intt stridex, intt &outy, intt &outx)
{
	if(scoop_inner) {
		outx = (((seqx - slidex) + stridex - 1) / stridex) + 1;//[((n - k) + d -1) / d] + 1 , d -1은 나머지가 있을경우 +1하기위해
		if(seqy) outy = (((seqy - slidey) + stridey - 1) / stridey) + 1;
		else outy = 1;
	} else {
		outx = (seqx + stridex - 1) / stridex;//[(n + d -1) / d] , d -1은 나머지가 있을경우 +1하기위해
		if(seqy) outy = (seqy + stridey - 1) / stridey;
		else outy = 1;
	}
	return outx * outy;//scoop되는 피쳐 갯수 리턴
}
intt scoopeout_shape(bool scoop_inner, intt slidey, intt slidex, intt stridey, intt stridex, intt &outy, intt &outx, Flux *fxp, intt &ndim, intt axid[])
{
	intt n_derive;
	bool d2 = fxp->fdim > 3 ? 1 : 0;

	n_derive = scoopeout_size(scoop_inner, d2 ? fxp->fshape[1] : 0, d2 ? fxp->fshape[2] : fxp->fshape[1],
		slidey, slidex, stridey, stridex, outy, outx);

	axid[0] = fxp->fshape[0] * n_derive;
	axid[1] = slidex * (d2 ? slidey : 1);
	if(d2) {
		axid[2] = fxp->fshape[3];
		ndim = fxp->fdim - 1;
	} else {
		axid[2] = fxp->fshape[2];
		ndim = fxp->fdim;
	}
	return n_derive;//scoop되는 피쳐 갯수 리턴
}