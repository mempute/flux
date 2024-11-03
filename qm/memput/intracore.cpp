
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
		gpuSetting(-1, nullptr);//set gpu device ���ϵ��� �� ����. rdtei)���� gpu focus����ȵǰ�
		lockArrange = 0;
		iGround = gid;//seqGpuAlc�����̰ų� ��ġȮ���� ù ������ �ƴϸ� ���� gid���� �ǹ̾���.
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
		longt sz_free = getmfreegpu(gid);//�־��� gpu�� �Ҵ緮�� ������ gpu id ����
		if(mcalc + trc->DEV_AVAIL_MARGIN < sz_free) return gid;

		intt i, n = 0;
		for(; n < trc->numdid; n++) {//gpu �˻�
			if(*(trc->didIndice + n) == gid) break;
		}
		if(n == trc->numdid) throwFault(-1, "avail gpu not found gid\n");

		for(i = n + 1; i < trc->numdid; i++) {//�˻��� gpu �ε��� ���� gpu�� �Ҵ緮 üũ
			sz_free = getmfreegpu(*(trc->didIndice + i));
			//printf("avail device: %d %d\n", sz_free, i);
			if(mcalc + trc->DEV_AVAIL_MARGIN < sz_free) return *(trc->didIndice + i);
		}
		for(i = 0; i < n; i++) {//�˻��� gpu �ε��� ���� gpu�� �Ҵ緮 üũ
			sz_free = getmfreegpu(*(trc->didIndice + i));
			//printf("avail device: %d %d\n", sz_free, i);
			if(mcalc + trc->DEV_AVAIL_MARGIN < sz_free) return *(trc->didIndice + i);
		}
		throwFault(-1, "device memory lack, allocation fail req: %lld capa: %lld\n", mcalc, sz_free);
		return -1;
	}
	void freeGroundEqualShadow(Flux *operout, intt iground)
	{
		DMATFX(operout)->shadowFree(iground);//adgzb.������� �׶��尡 ������ adgz)����
		GMATFX(operout)->shadowFree(iground);//adgzb.�Ҵ��� ������� �ʿ�����Ƿ� ����
	}
	void applyDevArrange(Apply *oper, Trace *trc, Flux *origin, intt iground, intt n, longt &mcalc, intt i_arrange)
	{//oper ���ö����� ��� �׶��忡 ��� arrange�Ǿ��ϴ� �Է��� ������� ��� �޸𸮸� arrange�Ѵ�. 
		//n�� Ȯ�� ��ġ������
		if(mcalc >= 0) mcalc += oper->mcalcapp(n);//���ö��� ���� �޸� ���
		for(FxAnchor *fxa = oper->lapOuput; fxa; fxa = fxa->ptrRight) {
			if(mcalc >= 0) {//�׶��� �޸� ����, ����� ������ gid �׶��忡 �Ҵ��ؾ��ϹǷ� ��� ����
				mcalc += fxa->fxPoint->sizefx2(n);//���ο��� ����ġ�� ���� ������������ �ƴ� �͵���
												//��ġ ���õǰ� ������� �����
			} else {//�׶��� �Ÿ� ���Ҵ�, ���ο��� ����� �۰ų� �׶��� ���̵� �ٸ� ��츸
				//���Ҵ��. ������������ �ƴ� ���� �׶��尡 �ٸ���츸 ���Ҵ��, ���ö����� ����
				//�޸𸮴� ���ö��� ������ ���࿡�� ������Ʈ ���信���� ����� �׶���� �ٸ� ��쿡��
				//��� �׶��忡 �� �Ҵ��.
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
			//���������Ǵ� �Է°� �� �Է����� ��µǴ� ����� �� �ٸ� ���� ���� �Է��� �Է����� ����Ǵ�
			//ü�λ��� ��� ����µ��� �� ��� �׶���� �Է� �����츦 ���ÿ� arrange�ϴµ� �� ��µ��߿�
			//ù��° ����� �Է��� ��ġ�� Ȯ��ɶ� arrange chain���� ��� �Է½������ ��� �׶��带
			//�Ѳ����� arrange�ϹǷ� �׷��� ���Ľ��࿡ ���� �̵� ü�λ��� ����� �ٸ� �ϳ��� ���߿� ���ö��̿�
			//���� ����ɶ� �̹� ��ġȮ������Ƿ� �� �Լ��� ������� �ʰ� �� ���ö����� �Է� ������ ����
			//arrange�� �� �����̹Ƿ� �������� �ٷ� arrangeDevice�� ����ȴ�.
			if(mcalc >= 0) {//�Է��� ������ �޸� ����, 
				if(cap->vcaps->fxType == trainable) {//adcg.����ġ�� �Ҵ�Ǵ� �޸� ���, ������
					//�ǰ� �������� �ʴ� ����ġ�� ����ġ ó���ϴ� �������� ���ܵǹǷ� �Ű᾵�� ����.
					mcalc += (cap->vcaps->sizefx() * add_weight_multiple);
				} else mcalc += cap->vcaps->sizefx2(n);//�����������̸� ��ġ�ݿ�, �ƴϸ� ������� ����
				//����ġ�� ���� ������������ �ƴѵ� ����� �޸𸮴� ���� �Ҵ��� ��)���� �׶��尡 ������
				//��ŵ�ǹǷ� �޸� ����� ����.
			} else {//���߿� const_apply���� ����, �ƴϸ� �Ʒ� ���ǿ� const_apply�� �־�� ��.
				if(DMATFX_GID(cap->vcaps) != iground || cap->vcaps->ofxArrange < trc->tcrArrange) {//��.
					//printf("alloc i-device %p %p %d %d %d %d\n", cap, cap->vcaps, DMATFX_GID(cap->vcaps), iground, cap->vcaps->scaleout, oper->apCode);
					intt sz;//�Է��� shadow�� ��� iground���� ���Ҵ�
					if(cap->vcaps->scaleout > 0) sz = cap->vcaps->sizefx2(n) / 2;//data + grad �������̹Ƿ� ������ ������ 1�� ������� ��ȯ
					else sz = cap->vcaps->sizefx();//��ġ Ȯ�� �����Ƿ� ���� ������
					if(cap->vcaps->fxType == trainable && cap->vcaps->ofxArrange < trc->tcrArrange) {
						if(DMATFX_GID(cap->vcaps) != iground) {//adar.����ġ�� ���� ��ġȮ�� ���� ���ö�����
							cap->vcaps->instTens(true, iground);// ����� �ƴϸ� �� �׶��忡 ���Ҵ�.
						}
						cap->vcaps->ofxArrange = i_arrange;
						//printf("%p apply dev arange %p %p %p %d %d %d %d\n", this, oper, cap, cap->vcaps, DMATFX_GID(cap->vcaps), iground, cap->vcaps->scaleout, oper->apCode);
					} else {
						//printf("%p apply dev arange2 %p %p %p %d %d %d %d\n", this, oper, cap, cap->vcaps, DMATFX_GID(cap->vcaps), iground, cap->vcaps->scaleout, oper->apCode);
						cap->allocShadow(sz, iground);//adgz.������ �޸� (��)�Ҵ�
					}
				}
			}
			//printf("222 %p %p\n", cap->vcaps, origin);
			//printf("555++++ %p %p %d %d %d\n", cap->vcaps, oper, oper->apCode, iground, cap->vcaps->ofxArrange);
		}
	}
	void notResizeArrange(Apply *oper, Trace *trc, intt i_arrange)
	{//�Է��� ��� ������������ �ƴѰ�� ȣ��, ���� ��¶��� ������������ �ƴϴ�.
		intt sz, iground, iGround = oper->oidground();
		longt done = 0, todo = oper->mcalcapp(-1);//���ö��� ���� �޸𸮴� ������ ���࿡�� �Ҵ�ǹǷ� �Ҵ�� �޸�
		
		for(FxAnchor *fxa = oper->lapOuput; fxa; fxa = fxa->ptrRight) {//��� �޸� ����
			//if(fxa->fxPoint->scaleout > 0) exit(1);
			//printf("1111 %p %p %d\n", oper, fxa->fxPoint, coi_arrangeoki);
			if(MATFX_INSTON(fxa->fxPoint)) done += fxa->fxPoint->sizefx2(-1);//�Ҵ�� �޸� ����
			else todo += fxa->fxPoint->sizefx2(-1);//�޸𸮰� ���߶� ������ �Ҵ���� ���� �Ҵ�� �޸� ����
			fxa->fxPoint->ofxArrange = i_arrange;
			//printf("444 %p %d %d\n", fxa->fxPoint, oper->apCode, fxa->fxPoint->ofxArrange);
		}
		if(todo) {//����� ���� �Ҵ���� ���� �Ÿ𸮰� ������ �Ҵ�, ���ö��� ���� �޸𸮴� ������ ���࿡�� �Ҵ�
			longt idone = 0;
			for(Capsule *cap = oper->lapInput; cap; cap = cap->ptrRight) {//�Է� �޸� �� ����
				if(cap->vcaps->fxType == trainable) {//adcg.����ġ�� �Ҵ�Ǵ� �޸� ���
					idone += (cap->vcaps->sizefx() * add_weight_multiple);
				} else idone += cap->vcaps->sizefx2(-1);//data + grad
			}//�Է� �޸𸮴� �̽������� ��� �׶��尡 �������� �ʱ� ������ ��� ����� ������ �ؿ���
			//�׶��尡 �ٸ� ��츸 ���� �Ҵ�ǹǷ� �޸� ����� ����.
			iground = availgpu(iGround, todo + idone, trc);
			if(iground == iGround) {//��� �׶��忡 �Ҵ�� �޸� �뷮�� ������ �Ҵ���� ���� �͸�
				for(FxAnchor *fxa = oper->lapOuput; fxa; fxa = fxa->ptrRight) {//�Ҵ�.
					freeGroundEqualShadow(fxa->fxPoint, iground);
					if(MATFX_INSTON(fxa->fxPoint) == 0) fxa->fxPoint->resizing5(-1, iground);//adges.
				}
			} else {//�޸� �뷮�� �ִ� �ٸ� ����̽��� �׶���� �Ͽ� �޸� �Ҵ�, ����� �׶���� ���
				iground = availgpu(iground, todo + done + idone, trc);//��ġ�ؾ��ϹǷ� �Ҵ�� �͵鵵
				for(FxAnchor *fxa = oper->lapOuput; fxa; fxa = fxa->ptrRight) {//�ٽ� ���Ҵ�
					freeGroundEqualShadow(fxa->fxPoint, iground);
					fxa->fxPoint->resizing5(-1, iground);//adges.
				}
			}
		} else iground = iGround;
		for(Capsule *cap = oper->lapInput; cap; cap = cap->ptrRight) {//��±׶���� �ٸ�
			//printf("%p aaa %p %p %d %d %d\n", this, oper, cap->vcaps, cap->vcaps->ofxArrange, DMATFX_GID(cap->vcaps), iground);
			if(DMATFX_GID(cap->vcaps) != iground || cap->vcaps->ofxArrange < trc->tcrArrange) {//��.
				sz = cap->vcaps->sizefx();
				if(cap->vcaps->fxType == trainable && cap->vcaps->ofxArrange < trc->tcrArrange) {
					if(DMATFX_GID(cap->vcaps) != iground) {//����ġ�� ���� ��ġȮ�� ���� ���ö�����
						cap->vcaps->instTens(true, iground);// ����� �ƴϸ� �� �׶��忡 ���Ҵ�.
					}
					cap->vcaps->ofxArrange = i_arrange;
					//printf("%p not resize arange %p %p %d %d %d %d\n", this, oper, cap->vcaps, DMATFX_GID(cap->vcaps), iground, cap->vcaps->scaleout, oper->apCode);
				} else {
					cap->allocShadow(sz, iground);//������ �޸� (��)�Ҵ�
				}
			}
		}
		iGround = iground;
	}
	void forwardDevArrange(Apply *oper, Trace *trc)//������ �������� ȣ��
	{//�Է°� ������ ����̽��� ����� �Ҵ��ϰ� �޸𸮰� ���߶� �Ҵ��Ҽ������� �ٸ� ����̽��� ����� �Ҵ��Ѵ�.
		intt n;
		longt mcalc;
		Capsule *cap;

		LOCK_MUT_(trc->mutArrange);//ffff
		lockArrange = 1;
		if(trc->tcrArrange > oper->lapOuput->fxPoint->ofxArrange) {//������ ó�� �����̸� ����
			for(cap = oper->lapInput; cap; cap = cap->ptrRight) {//�Է��߿��� �����Ϸ���(��ġȮ���)
				if(cap->vcaps->scaleout > 0) break;//�÷��� ����
			}
			if(cap) n = cap->vcaps->scaleout;//(Ȯ���)��ġ ����
			else n = -1;
			if(trc->gpugear == 0 && n < 0) {//gpugear�� true�̸� ��ġȮ����� �ʴ��� �׷��� �б�ɶ�
				//�ٸ� gpu���� ���ķ� ����Ǳ����� �׷��� �б⿡�� ���� gid�� �����Ǿ� �־��� ���� gid��
				//���Ҵ� �ؾ� �ϹǷ� �� ���� �������� �ʴ´�.
				//���ö����� �Է��� ����ġ�� ���� ��ġ Ȯ����� �ʴ� �͸� ����(split���� ���)
				//�Է��� initArrange�� ���ؼ��� �ʱ�ȭ ��� ofxArrange�� 1�̰� ���⼭ ����� arrange
				//�Ǿ� 1�� �Ǹ� ���� ���࿡���� ���� arrange������ ��ŵ�Ǿ� ���Ŀ� �� ���̽��� �ٽ� 
				//���۵Ǵ� �� ����. ��ġȮ��� ��������Ƿ� �Է��� ofxArrange�ٲ��� ���⶧����. �Ŀ�
				//�� ���۷��̼��� ������� ���Ҵ�Ǵ� ���� �Ʒ� ��ġȮ�� ���̽��� �����߿� arrange 
				//chain�� ���� ���� ������� �ٽ� �� arrange �Ǵ� ����̴�. �̶� �̹��� ��ġ�� 
				//ofxArrange�� ���� ��ŵ�Ǵ� �� ���� ���� arrange����ȴ�. ������ �Ʒ� ���̽� ������
				//���� feed������ ofxArrange�� 2���� ���۵Ǳ⶧���� ���⼭ 1�� �Ǿ �ε����� 
				//�ٸ��Ƿ� ��ŵ���� �ʰ� �ٽ� ��� �׶��忡 arrange�ȴ�.
				notResizeArrange(oper, trc, trc->tcrArrange);
			} else {//cap->vcaps->ofxArrange�� trc->tcrArrange�� ����
				//printf("aaa %p %d %d\n", oper->lapOuput->fxPoint, oper->apCode, oper->lapOuput->fxPoint->ofxArrange);
				mcalc = 0;//arrange�� memmory�� ��� ǥ��
				add_weight_multiple = 4 + (trc->mastTrc ? 0 : 1);//���ö����� �Է��� ����ġ�϶�
				//����ġ�� iGround �׶��忡 �ΰ������� �����Ǵ� ��Ʈ���� ���� ����, ����ġ�� data�� grad��
				//������ 2�� + �ƴ� ��Ƽ�������� ����Ͽ� ����ġ�� 2���� ��Ʈ���� �Ҵ� + ������� �����̰� 
				//�������̸� �����Ϳ� �����̺��� ������ �����Ϳ��� ���� ���� ���ؾ� �ϴµ� ���øӽų� �������
				//���� ����̸� �����Ϳ� �����̺��� �׶��尡 �ٸ��� �����Ϳ� �޸𸮸� �Ҵ� �غ��Ͽ� �����̺꿡
				//�����ϰ� �����̺��� ���� ������� ������ �������� devDock 1�� Ȥ�� �л������� �����̸�
				//�������� iGround�� ��Ʈ������ �����Ͽ� �����̺��� ���� ��Ʈ������ ������ �� 1��(�� �����̴�
				//�л��̴� ��������� ������ �����̸� 1�� �߰�)�Ͽ� 4 + 1(?)
				if(trc->gpugear == 0) {//���ö����� arrange�Ǿ��� ��°� ��� �Է��� �����츦
					iGround = oper->iidground();//�ϴ� �Է��� �׶��忡�� ���Ҵ� �Ҽ��ִ��� üũ
				}//else gpu���� ������ �����ϱ����� �־��� gid�� �Ҵ��Ѵ�.
				//oper ���ö��� ������ ���� ��� arrange�Ǵ� �ҿ� �޸� �뷮 ���
				applyDevArrange(oper, trc, nullptr, iGround, n, mcalc, trc->tcrArrange);
				iGround = availgpu(iGround, mcalc, trc);
				mcalc = -1;//arrange�� memmory �Ҵ� ǥ��
				applyDevArrange(oper, trc, nullptr, iGround, n, mcalc, trc->tcrArrange);
				//printf("%p 222 %p %d %d\n", this, oper, oper->apCode, iGround);
			}
		}
		lockArrange = 0;
		UNLOCK_MUT_(trc->mutArrange);
	}
	void focusDevice(Apply *ap, TCxtArray *tcxt_arr)
	{
		if(tcxt_arr->focusDevGThr != iGround) {//���ö��� �������� ����� �׶��� ��迡 ��Ŀ���Ѵ�.
			//printf("focus device %d %d %d %p\n", ap->apCode, tcxt_arr->focusDevGThr, iGround, tcxt_arr);
			tcxt_arr->focusDevGThr = iGround;
			CudaDevSet(iGround);//�� �Լ��� �׷��� ���� ������� ��� �����Ǵ� ���������� ȣ��ȴ�.
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
						iGround = ap->oidground();//��� �׶���
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
							continue;//trptn.�÷����� ������ apply�� �÷����� ������ Ʈ���̼��� �ƴ� �� �÷����� �Է������ϴ� 
						}			//Ʈ���̼����� ������ ���ö����̸� �� ���Ĵ� �������� �ʴ´�.
						if(ct->ptrRight) {//�÷��� ���� ���� �б� ����
							//checkTcr(ct->ptrRight, trctrk);
							//((Trace *)trctrk)->incFwFork();
							ttrk = (ThreadTrack *)rsc::brutra->trkGet2<ThreadTrack>();
							ttrk->mSetTrack(ct->ptrRight, true, trctrk->gpugear == 3 ? nextgpu(iGround, trctrk) : iGround, trctrk);//��.�б�ǹǷ� ���� �׶��忡�� ����
							ttrk->ontrack(srGate);
						}
						if(nextp) {//��)���� �°�, split�б� ����
							//checkTcr(nextp, trctrk);
							//((Trace *)trctrk)->incFwFork();
							ttrk = (ThreadTrack *)rsc::brutra->trkGet2<ThreadTrack>();
							ttrk->mSetTrack(nextp, true, trctrk->gpugear > 1 ? nextgpu(iGround, trctrk) : iGround, trctrk);//�б�ǹǷ� ���� �׶��忡�� ����
							ttrk->ontrack(srGate);
						}
						if(trctrk->devArrange && trctrk->cpmMode <= 0) {
							forwardDevArrange(ap, trctrk);
						}
						iGround = ap->oidground();//��� �׶���
						focusDevice(ap, tcxt_arr);
						ap->arrangeGround = iGround;
						ap_exec = 1;
						nextp = ap->forward(tcxt_arr->getgputxt(iGround, trctrk), mtx_bwmut);//��.
						ap_exec = 0;
						//checkTcr(nextp, trctrk);
						break;
					}
					break;
				case GEN_T_TACT:
					ct = (Contact *)nextp;//�ٸ� �������� ��)���� �°�
					//checkTcr(nextp, trctrk);
					nextp = nullx;
					fx = nullx;
					goto LB1;
				case GEN_T_CAP://split, ��)�� ����
					cap = (Capsule *)nextp;
					nextp = cap->ptrRight;//��.
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
				if(*(sytet *)nextp == GEN_T_CAP || ((Flux *)nextp)->checkInBwfx(1)) {//��)���� �°��̸�
					while(nextp) {								//��)��ó üũ�Ǿ��� ���� �������̸� ���⼭ üũ
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
							iGround = ap->oidground();//��� �׶���
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
								if(cap->vcaps->bwbreak || cap->vcaps->fxTcr != trctrk || cap->vcaps->checkInBwfx2() == false) continue;//��.
								//trptn.�÷����� �ٸ� Ʈ���̼����� �����Ǿ� �� Ʈ���̼����� �Է����� ����ϴ� ���̸� ������ ���� ��ŵ
								//tcfnr.bwbreak�̸� ������ ���� ���� �߰�
								if(cap->ptrRight) {
									//checkTcr(cap->ptrRight, trctrk);
									//((Trace *)trctrk)->incBwFork();
									ttrk = (ThreadTrack *)rsc::brutra->trkGet2<ThreadTrack>();
									ttrk->mSetTrack(cap->ptrRight, false, 0, trctrk);//��.������ �����忡�� ������ 
									ttrk->ontrack(srGate);					//�׶��忡�� �����ϹǷ� �׶��� �ƴϵ� �ǹ� ����
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
		gpuSetting(-1, nullptr);//set gpu device ���ϵ��� �� ����. rdtei)���� gpu focus����ȵǰ�
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
			Flux *slave_wfx = slavTrc->hbaper->getflux(cap->vcaps->bregid);//prelu�� ��� ����ġ���� ��ġ ����� 
			if(cap->vcaps->fxSize != slave_wfx->fxSize) slave_wfx->resizing2(cap->vcaps, "slav");//�����ϹǷ� ������ ����Ǹ� �ݿ��Ѵ�.
			DMATFX(slave_wfx)->inCopy(DMATFX(cap->vcaps), 0);
		}//������ ����ġ �����͸� �����̺�� ����
		try {
			if(tcxt_arr->focusDevGThr < 0) {//�����̺� �������� ��� ��Ŀ������ ���� �ʱ� �ѹ���
				//�̹� �����̺� ������ trc�� ���������̵�� �̹� �����̺�trc�� run�� �����̺� ��������
				slavTrc->focusgpudev(slavTrc->mastTrc->didTrace);//��Ŀ�� ��� ���̵� �����Ѵ�.
				tcxt_arr->focusDevGThr = slavTrc->didTrace;
			} else {//�����̺� �������� ��� ��Ŀ���� �Ŀ��� �Ź� ���ε��� �����̺�trc�� run�Լ� �����
				slavTrc->didTrace = tcxt_arr->focusDevGThr;//��Ŀ���� �����̺� �������� ��Ŀ���� 
			}											//�����Ͽ� �̹� �����̺� Ʈ���� �����Ѵ�.
			slavTrc->run(&target);
			LOCK_MUT_(slavTrc->mastTrc->mutMerge);
			if(slavTrc->trainOpt) {//�����̺��� ����� �������� ���⸦ ����.
				for(Capsule *cap = slavTrc->mastTrc->trainWeight;cap; cap = cap->ptrRight) {
					/*if(cap->vcaps->fxName[6] == 'c') {
						printf("------------------------------\n");
						cap->vcaps->printg();
					}*/
					Matrixr *gmst = GMATFX(cap->vcaps);
					Matrixr *gslv = GMATFX(slavTrc->hbaper->getflux(cap->vcaps->bregid));
					if(gmst->didground != gslv->didground) {//�����̺�� �������� �׷����Ʈ �׶��� 
						gmst->dockAlloc();//���̵� �ٸ��� �����Ϳ� ��ũ�� �����ϰ�(���� Ȯ���� adcg 
						gmst->devmcopy(gmst->devmpoint(3), gslv->devmpoint(0));//���� ��� ��) 
						gslv->devmsetting(0, gmst->devmpoint(3));//�����̺긦 ��ũ�� �������� 
					}							//�������� ��ũ�� �����̺��� ��Ŀ���� �����Ѵ�.
					if(tcxt_arr->focusDevGThr != gmst->didground) {
						tcxt_arr->focusDevGThr = gmst->didground;
						CudaDevSet(tcxt_arr->focusDevGThr);//�����̺� ���� ������� ��� �����ȴ�.
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
			for(auto tfx : *tgtfx) {//�����̺��� Ÿ�ٰ��� �������� Ÿ�ٰ��� ����.(�׶��� ���� Ÿ�� ��)
				Matrixr *tmst = DMATFX(tfx);
				Matrixr *tslv = DMATFX(slavTrc->hbaper->getflux(tfx->bregid));
				if(tmst->didground != tslv->didground) {//���� ���� ó��, Ÿ���� ����� �����Ƿ�
					tmst->dockAlloc();//adcg)���� ���� Ȯ�� ���� DEV_AVAIL_MARGIN������ ����ȴ�.
					tmst->devmcopy(tmst->devmpoint(3), tslv->devmpoint(0));
					tslv->devmsetting(0, tmst->devmpoint(3));
				}
				if(tcxt_arr->focusDevGThr != tmst->didground) {
					tcxt_arr->focusDevGThr = tmst->didground;
					CudaDevSet(tcxt_arr->focusDevGThr);//�����̺� ���� ������� ��� �����ȴ�.
				}
				tmst->marith(tcxt_arr->getgputxt(DMATFX_GID(tfx), slavTrc),
					tslv, tmst, nullx, 0, nullx, nullx, AOP_PLUS);
			}
			//for(auto tfx : *tgtfx) {//�����̺��� Ÿ�ٰ��� �����ͷ� ����
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
void Trace::infeed(Flux *feed_fx, void *pdat, sytet tsrc, intt sz) //sz�� ���� ����
{
	intt nbatch = sz / (feed_fx->fxSize / feed_fx->fshape[0]);
	intt bsz, rsz;

	nexeSlav = (nbatch % batchPart ? nbatch / batchPart + 1 : nbatch / batchPart) - 1;//�����̺� �����̹Ƿ� -1
	if(nslavTrc < nexeSlav) {
		for(intt i = nslavTrc;i < nexeSlav; i++) division();
	}
	ibatchSize = (nexeSlav ? nbatch : 0);
	bsz = (nbatch < batchPart ? nbatch : batchPart);

	if(feed_fx->fshape[0] < bsz && feed_fx->realwiden(bsz)) {
		tcrArrange = ++feed_fx->ofxArrange;//�޸� ����� Ȯ��ȴٸ� arrange����ǰ� ����.
	}
	if(tsrc < 0) rsz = feed_fx->copyf2(pdat, bsz);//rsz�� �̹��� ����� ���� ����Ʈ ��
	else rsz = feed_fx->copyt(pdat, tsrc, bsz);//rsz�� �̹��� ����� ���� ����Ʈ ��

	if(nexeSlav == 0) return;//run�Լ� ���� ���̰ų� ��ġ ������ ������ �ϳ� ������ ����ɰŸ� ��ŵ
	Flux *slav_fx;
	nbatch -= bsz;
	for(Trace *trc = lstSlav;trc && nbatch > 0; trc = trc->ptrRight, nbatch -= bsz) {//�����̺� �Է�
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

	for(Capsule *cap = lorigin;cap; cap = cap->ptrRight) {//����ġ�� ���� �׷����� ������ ������ �����Ƿ�
		//������ ����Ʈ�� ��� �����õ��ִ�. �̿� �ٸ� Ʈ���̼��� ��� �÷����� �Է����� �ϴ� ��� tcvy)���� ������ �����õǳ�
		if(cap->vcaps->fxType != memput::mp::trainable) continue;//����ġ�� �ƴϹǷ� ���⼭ ��ŵ�Ǿ� �������.
		acap = getCap();
		acap->vcaps = cap->vcaps;
		APPEND_LIST(trainWeight, acap); 
		//printf("%p %s\n", acap->vcaps, acap->vcaps->fxName);
	}
	//printf("\n");
}
void Trace::vSync(Flux *endp) //name, anonym ��� ��ũ�ȴ�.
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
		if(endp->fxTcr == this) endp->nbRefer++;//trptn.���� ����, Ʈ���̼����� �÷����� ���� �������� �ʴ´�.
		//msgdisp((bytet *)"222 %p %d\n", endp, endp->nbRefer);
		if(endp->quantum) TENSOR(endp->quantum)->mxGrad->nbackw = endp->nbRefer;
		//if(endp->quantum) msgdisp((bytet *)"bbb %p %d\n", TENSOR(endp->quantum)->mxGrad, TENSOR(endp->quantum)->mxGrad->nbackw);
		return;
	}
	while(1) {
		for(;ap; ap = (first && first->bwbreak == 0 ? (Apply *)first->bwLink : nullx)) {//ap�� ù��°�� �����������ʰ� ������ �ٷ� apŽ���Ѵ�.
			//printf("aaa: %d %p\n", ap->apCode, ap->apTcr);
			if(ap->apCode >= APC_ADMOPT && ap->apCode <= APC_SGDOPT) {
				trainOpt = ap;
				if(((ApOptimizer *)trainOpt)->optLoadStep == 1) ((ApOptimizer *)trainOpt)->optLoadStep++;//��Ƽ������  
			}	//�ڵ� �ε� �� ó�� �ѹ��� �ε� ������ ������Ų��. �ٽ� �� �ڵ尡 ����ɷ��� �ڵ�ε�(�׷��� ����)�� �ٽ� �Ǿ��Ѵ�.
			ap->nbfanOut++;
			//msgdisp((bytet *)"xxx %d %d\n", ap->apCode, ap->nbfanOut);
			//if(ap->apTcr->mastTrc != mastTrc) exit(1);
			if(ap->vbackW == v) break;
			apc = getApc();
			apc->vapply = ap;
			APPEND_LIST(lapply, apc);
			first = ap->lapInput->vcaps;//��� first�� ���� depth�� �����Ѵ�.
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
				//fnisqgt.first->instTens(false);//ap�� ù��°�� �Ʒ� �������� ������������Ƿ� ���⼭ �ʱ�ȭ.
				if(first->bwLink == nullx || first->fxTcr != this || first->bwbreak) {//trptn.�÷����� �� tcr�� �ƴ� �ٸ� tcr�� ��� 
					listOrigin(first);//tcvy.�� ���θ� �������ϱ����� ��üũ //�÷����� �Է����� �ϴ� ���̸� �����ľʰ� ���η� ������.
					//if(first->fxTcr != this) printf("bbb: %p %p\n", this, first->fxTcr);
				}
			} else first = nullx;//�߰� ���߿� ����.
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
					listOrigin(cap->vcaps);//tcvy.�� ���θ� �������ϱ����� ��üũ
					//if(cap->vcaps->fxTcr != this) printf("ccc: %p %p\n", this, cap->vcaps->fxTcr);
				}
				cap->vcaps->backwv = v;
				cap->vcaps->nbRefer = 1;
				//printf("666 %p %d\n", cap->vcaps, cap->vcaps->nbRefer);
				if(cap->vcaps->quantum) TENSOR(cap->vcaps->quantum)->mxGrad->nbackw = 1;
				//if(cap->vcaps->quantum) printf("fff %p %d\n", TENSOR(cap->vcaps->quantum)->mxGrad, TENSOR(cap->vcaps->quantum)->mxGrad->nbackw);
				//fnisqgt.cap->vcaps->instTens(false);
				cap2 = getCap();//���⼭ �Ҵ�Ǵ� cap�� ���⿡���� ��ȸ������ ���ǹǷ� �ؿ��� lrecyc��ȯ
				cap2->vcaps = cap->vcaps;
				APPEND_LIST(lcap, cap2);
			}
		}
		do {
			GET_LIST(lcap, cap);
		} while(cap && cap->vcaps->bwbreak);
		if(cap == nullx || cap->vcaps->fxTcr != this) break;//trptn.�÷����� �� tcr�� �ƴ� �ٸ� tcr�� ��� �÷����� �� tcr�� �� ������ ������Ѵ�.
		ap = (Apply *)cap->vcaps->bwLink;
		APPEND_LIST(lrecyc, cap);//�ؿ��� �����ϱ����� ������
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
//�Ʒ��� �ƴ� �� �ܰ� �����̸� ��ġ�� �����ϸ� ��ġ ���� ������ ��°��� ���ϹǷ� �Է¿� ��Ī�Ǵ� ��°��� ������ �����Ƿ�
//�򰡴ܰ迡�� ��ġ�� �����ϸ� �ȵȴ�.
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
	if(i < n || fp) {//Ÿ�� ����Ʈ�� ��������� �ʱ�ȭ ����.
		lstTarget = nullx;
		trainOpt = nullx;
		initVsync();
		for(auto fxp : *target) {
			APPEND_LIST(lstTarget, fxp);
			vSync(fxp);
		}
		listWeight();//�� v sync�Ŀ� �ؾ���
		if(mastTrc == nullx) {//������ trc������ ����.
			if(trainOpt) {
				if(((ApOptimizer *)trainOpt)->optLoadStep > 1) {//��Ƽ������ �ڵ尡 �ε���� ���� Ÿ�� �н��� 
					((ApOptimizer *)trainOpt)->optLoadStep = 0;//��Ƽ�������� ������ �Ʒ� ��Ȳ�̹Ƿ� �ڵ� �ε��� ó�� 
					init_train();								//�ѹ��� ����ġ�� �ʱ�ȭ ��Ų��.
				}
			} else loadWeight();//�н������� �ƴϸ� ����ġ �ε�
		}
		hbaper->baperEnding(1);
	}
	if(nexeSlav) {
		if(mastTrc == nullx) {//������ trc�̸�
			mastsr = rsc::prutra->srGet();
			Trace *trc = lstSlav;
			SlaveTrack *strk;
			for(i = 0;i < nexeSlav; i++, trc = trc->ptrRight) {
				strk = (SlaveTrack *)rsc::prutra->trkGet2<SlaveTrack>();
				strk->mSetTrack(trc, target);//��ġ ���ҵ� �Է� �� Ÿ������ �� ���� ��ġ �����̺� ������(run�Լ�) ����.
				strk->ontrack(mastsr);
			}
			LOCK_MUT_(mutMerge);//�������� ������ ������ ���� �����̺꿡�� ���� ���� add �ϴ� ���� ����
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
	if(trainOpt == nullx) {//��Ƽ�������� ���� �ȵ�.
		sr->srPut<ThreadTrack>();
		if(nexeSlav && mastTrc == nullx) {//��ġ���� �����̰� �������̸� �����̺갡 �����Ҷ����� ���
			UNLOCK_MUT_(mutMerge);
			mastsr->srWait();
			mastsr->srPut<SlaveTrack>();
		}
		goto LB1;//optimizer forward�� ������� �ʾ����� ����� ���� ���� �ʴ´�.
	}
	//rsc::brutra->onSingle(false);
	resetGrads();//���� ����� ����
	//if(lapType == 3) printf("run#3 lap: %lld\n", xucurrenttime() - lap);lap = xucurrenttime();
	for(auto fxp : *target) {
		//if(mastTrc == nullx) continue;
		ttrk = (ThreadTrack *)rsc::brutra->trkGet2<ThreadTrack>();
		ttrk->mSetTrack(fxp, false, 0, this);//����� ������ gid �ǹ� ����.
		ttrk->ontrack(sr);
	}
	sr->srWait();
	//chkBwFork(sr);
	//printf("%d: %d %d\n", idTrace, ibwFork, nbwFork);
	//if(lapType == 3) printf("run#4 lap: %f\n", (xucurrenttime() - lap) / 1000000.0);lap = xucurrenttime();
	sr->srPut<ThreadTrack>();//mmm
	if(nexeSlav) {//��ġ���� �����̰�
		if(mastTrc) return;//�����̺��̸� ���� ����ʰ� ����
		else {//�������̸� �����̺갡 �����Ҷ����� ���
			UNLOCK_MUT_(mutMerge);
			mastsr->srWait();
			mastsr->srPut<SlaveTrack>();
			/*floatt f = nexeSlav + 1;//����ġ ��� ���ϱ�
			for(Capsule *cap = trainWeight;cap; cap = cap->ptrRight) {
				TENSOR(cap->vcaps->quantum)->mxGrad->marith(trcCxt(), nullx, TENSOR(cap->vcaps->quantum)->mxGrad, 
					nullx, cap->vcaps->qType, &f, nullx, AOP_DIV);
			}*/
			//for(Capsule *cap = trainWeight;cap; cap = cap->ptrRight) {//Ŭ����
			//	TENSOR(cap->vcaps->quantum)->mxGrad->mclip(trcCxt(), TENSOR(cap->vcaps->quantum)->mxGrad, -5.0, 5.0);
			//}
		}
	}
	((ApOptimizer *)trainOpt)->update(tcxtarr, this);

	if(nexeSlav) {//��ġ���� �����̸� ��հ� ���� �÷������� ����� �����Ѵ�.
		for(auto fxp : *target) {//Ÿ�� ������ �����̺� Ʈ������ �� �����̺� ���� �������� sum�ǰ� 
			if(fxp->meanAfter) {//Ÿ�ٵ��� �׷������� mean�Լ� ������ �÷����� �ش��ϸ� �����̺� ������ ��� ����.
				TENSOR(fxp->quantum)->mxData->mmean(nexeSlav + 1);
			}
		}
	}
LB1:;
	//if(didTrace != sidTrace) {//�����̺�� ������� ������� �����Ƿ� �ش����.
	//	didTrace = sidTrace;
	//	CudaDevSet(didTrace);
	//}
	if(lapType == 3) printf("run#5 lap: %f\n", (xucurrenttime() - lap) / 1000000.0);
}

intt scoopeout_size(bool scoop_inner, intt seqy, intt seqx, intt slidey, intt slidex, intt stridey, intt stridex, intt &outy, intt &outx)
{
	if(scoop_inner) {
		outx = (((seqx - slidex) + stridex - 1) / stridex) + 1;//[((n - k) + d -1) / d] + 1 , d -1�� �������� ������� +1�ϱ�����
		if(seqy) outy = (((seqy - slidey) + stridey - 1) / stridey) + 1;
		else outy = 1;
	} else {
		outx = (seqx + stridex - 1) / stridex;//[(n + d -1) / d] , d -1�� �������� ������� +1�ϱ�����
		if(seqy) outy = (seqy + stridey - 1) / stridey;
		else outy = 1;
	}
	return outx * outy;//scoop�Ǵ� ���� ���� ����
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
	return n_derive;//scoop�Ǵ� ���� ���� ����
}