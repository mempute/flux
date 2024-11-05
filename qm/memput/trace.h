#pragma once

#include "rsc.h"

#define NONET_TP		0	//Ÿ���� ����
#define BYTET_TP		1
#define SHORTT_TP		2
#define FLOATT_TP		3
#define INTT_TP			4
#define LONGT_TP		5
#define DOUBLET_TP		6

typedef struct VectorTag_ {
	vector<Flux *> *pvector;
	struct VectorTag_ *ptrPrev;
} VectorTag;

typedef struct _ShadowCap {
	intt didshadow, shadowsz;
	void *devShadow;
	struct _ShadowCap *ptrLeft, *ptrRight;
} ShadowCap;

class Capsule : public Typer {
public:
	sytet Tgener = GEN_T_CAP;
	intt dbgIdShadow;
	Flux *vcaps;
	ShadowCap *dataShadow, *gradShadow;//didshadow, shadowsz�� ����
	Capsule *ptrLeft, *ptrRight;
	void allocShadow(intt sz, intt gid);
	void freeShadow(void);
	void setShadow(intt gid, bool grad = false);
};
class Apply;
class Matrixr;
typedef struct ApCap_ {
	Apply *vapply;
	struct ApCap_ *ptrLeft, *ptrRight;
} ApCap;
class Baper;
#define NAME_LENG	2048
#define TRACER(trc) ((Trace *)trc)
class Trace : public Tracer {
public:
	void *hmalc;
	intt colorVersion, bwVersion;
	sytet prompt;//��ý���, �Ź� ����ü ����, Trace�� delete �ܰ� api�� �����Ͽ� �Ϲ� ���α׷��� ���� ���� �����ֱ� �����ؾ� ��.
	sytet lapType;
	sytet cpmMode;
	bool loadedWeight, fastshape, devArrange;
	Apply *trainOpt;
	NameScope *nsTop, *nsFocus, *nsOrder;
	VectorTag *vtLast;
	Matrixr *mtxlist;
	ApCap *lapply, *poolApc;
	Capsule *lorigin, *poolCap;
	hmutex mutTrc, mutMerge, mutArrange;
	bytet *trcname;
	intt idTrace, seqNscope;
	Flux *lstTarget;
	Baper *hbaper;
	intt npAbrib, tcrArrange;
	intt didTrace, sidTrace;//Ʈ���̽��� �ʱ�ȭ �Ǵ� �����忡�� ������ ���� ��� ��Ŀ��
	TCxtArray *tcxtarr;//prompt�϶� �׷��� �����ϸ� �ٷ� �����Ҷ� ���.
	intt nslavTrc, nexeSlav, batchPart, ibatchSize;
	//intt nfwFork, nbwFork, ifwFork, ibwFork;
	Trace *mastTrc;
	SignalR *mastsr;
	Trace *lstSlav, *ptrLeft, *ptrRight;//lstSlav���� ��
	Capsule *trainWeight;
	bytet rPath[NAME_LENG];
	longt DEV_AVAIL_MARGIN;
	//anet ���� member
	floatt contigusRate, learningRate;
	intt bandwidth, spotlat;
	intt kernelSize, strideJump;
	intt convolving;//Ŀ�� ����� 8�̸� half(2), quard(4), octa(8) �������� ���ǰ�(octa�� 1/8�̹Ƿ� �� ��ҵɼ�����)
					//Ŀ�� ����� 16�̸� 16���� ������ 4������ ���ȴ�. Ŀ�� ������� ���� ������ �����ϹǷ� ������ ���ϼ��� ����.
	intt sourceInterv, compbot;
	intt nlearnStop;
	floatt elearnStop, lowbound, relualpha, rkernal2d;
	sytet gradby, gpugear, derivefeat;
	sytet dotType, zeroPadding, elasticHidden, ebatch_t, dot_tDec, actfType, actfAtt;
	sytet spotOpt, hidPrint, bpPrint, attsign, optType, completeConvolve, stratusOpt, stratusOpt2;
	sytet att_sDec, layer_norm, zstep, poskern, rderivet, dualloss, nblockChain, resiopt;
	sytet positional, dualChain, ehook, frozenwgt, eternity, enckern, samplingInterv, dualEmbed;
	bool dualStep, outback, tcrCublas, softfill, directMax;
	bool printBuilding, printLoad, endecode, trcpart, intergate, dualatten, resiFirst;
	bool directExec, pathPrint, ortho_single, reducea, inplus, softmdot, dual_lnorm, stracode, predlog;
	intt nbyout, slatent, sembedn, rseed, multihead, udreduce, icodefz, nbatch2pseq;
	Flux *bygate, *gotok, *endtok;//�ΰ� ��ū�� �߷� ���������� ���
	intt numdid, *didIndice;
	//���� dynagen����
	sytet overlapSlide, slTrainInterv, slPredInterv, concurtrain, szDualReduce, szkernDualNest;
	bool scoopinner, batchloss, printloss, autoRegress, outerMask, externFinal;
	intt dgenmgc, inzbound, tarzbound, szslide2d, szslide1d, dualdff;
	floatt rTrainBound;
	bytet trcmsg[16];//���� Ÿ�� ����� ����
	Trace(sytet stepw, const bytet *name)
	{
		bytet buf[NAME_LENG];
		
		trcmsg[0] = '\0';
		hmalc = rsc::mAllocator();
		idTrace = 0;
		cpmMode = -1;
		loadedWeight = 0;
		if(name) sprintf(buf, "%s", name);
		else sprintf(buf, "trc%d", idTrace);
		trcname = (bytet *)xalloc(strleng(buf) + 1);
		strcpy(trcname, buf);
		tcxtarr = (TCxtArray *)xalloc(sizeof(TCxtArray));
		tcxtarr->initTCxtArray(rsc::ngpudev);
		colorVersion = 0;
		idTrace = memput::glb::rsc::seqTraceId++;
		nsOrder = nsFocus = nullptr;
		seqNscope = 0;
		nsTop = namescope(nullx);
		vtLast = nullptr;
		mtxlist = nullptr;
		lapply = poolApc = nullx;
		lorigin = poolCap = nullx;
		trainWeight = nullx;
		prompt = (stepw > 0 ? 1 : 0);
		installBaper(stepw);
		intt rv; 
		CRE_MUT_(mutTrc, rv);
		if(rv < 0) throwFault(-1, (bytex *)"cre mut fail\n");
		CRE_MUT_(mutMerge, rv);
		if(rv < 0) throwFault(-1, (bytex *)"cre mut fail\n");
		CRE_MUT_(mutArrange, rv);
		if(rv < 0) throwFault(-1, (bytex *)"cre mut fail\n");
		bwVersion = 1;
		lstTarget = nullx;
		didTrace = 0;
		npAbrib = 7;
		lapType = 0;
		nslavTrc = nexeSlav = 0;
		mastTrc = nullx;
		lstSlav = nullx;
		batchPart = 10000;
		//nfwFork = nbwFork = POSITIVE_INT_MAX;
		strcpy(rPath, ".");
		ibatchSize = 0;
		gradby = 2;//�ټ��÷ο쿡�� 2�� ������.
		characterType = 0;
		characterName[0] = '\0';
		//anet ���� member
		bandwidth = kernelSize = 8;
		contigusRate = -1;
		strideJump = -1;
		dotType = zeroPadding = elasticHidden = -1;
		convolving = 1;
		completeConvolve = 0;
		printBuilding = 0;
		ebatch_t = 2;//�� ���̽��� �ƴ� ������ ���̽��� ����������� Ŀ�λ������� �������϶��� ����Ѵ�. ������� ������ ����� 56�̰�
					//Ŀ�λ���� 8�̸� ��Ʈ���̵尡 8�϶� part������ 7�̵ǰ�(56/8) convolving�� 4�϶� ��»������
					//16�� �Ǿ� 7�� ����� ���� �����Ƿ� ���� �������� ebatch_t�� 1 ���̽��� ����Ǿ� �߰��� �����İ�
					//���� �ʴ´�. �� ���̽��� 2�� �����ϸ� agzo)�� �������� ���� zpad 3�� ���̽��� �������� ���ϹǷ�
					//�̸� ����Ѵ�.
		printLoad = 0;
		dot_tDec = -1;
		endecode = 0;
		spotlat = -1;
		lowbound = 1;
		relualpha = -1;
		layer_norm = 0;
		spotOpt = -1;
		positional = 0;
		actfType = -1;
		hidPrint = -1;
		bpPrint = -2;
		directExec = 0;
		attsign = 0;
		optType = 0;
		learningRate = -1;
		pathPrint = 0;
		stratusOpt = 0;
		stratusOpt2 = 0;
		dbgStep = dbgStep2 = 0;
		sourceInterv = 1;
		trcpart = 0;
		intergate = 0;
		actfAtt = -1;
		nlearnStop = 0;
		elearnStop = 0.02;
		att_sDec = 0;
		ortho_single = 0;

		inzbound = tarzbound = 64;
		szslide2d = 8;//2d�� x*y(8*8)������, Ŀ�δ� 1���� ��µǾ��ϰ� �ִ� ������ Ŀ�� ����� 8�϶�
		//�����긦 8���ϸ� 1, �Է��� 16�̹Ƿ� ����� 2���Ǿ� ��¾����ڵ� ������� �������� �ʵǹǷ� lowbound��
		//0.5���Ͽ� 4(Ŀ�λ�����8 * 0.5)�� ���߰� 4(2*2) �μ� �������� �ǰ��Ѵ�, �����̵� ����� 64(8*8)��
		//�Ѵٸ� ������ 8, �ο�ٿ�� 2�� �Ͽ� ��¾����ڵ� ����� 16(4*4)�� �������� �ǰ� �Ѵ�.
		szslide1d = 8;//1d�� x ������, lowbound�� 0.25���ϸ� �����ڵ������� 2, 0.125�� 1, 0.5�� 4
		rkernal2d = 0.125;// 1/8
		overlapSlide = 1;
		slTrainInterv = 3;
		slPredInterv = 1;
		scoopinner = 0;
		batchloss = 0;
		printloss = 0;
		concurtrain = 0;
		rTrainBound = 0.007;
		nbyout = -1;
		autoRegress = 1;
		szDualReduce = 1;
		szkernDualNest = 0;
		zstep = -1;
		rderivet = 1;
		poskern = 3;
		derivefeat = 0;
		multihead = 0;
		//dualloss = 1;
		outerMask = 1;
		dgenmgc = 3;
		reducea = 0;
		rseed = -1;
		externFinal = 0;
		dualdff = 0;
		dualChain = 0;
		inplus = 0;
		nblockChain = 0;
		resiopt = 3;
		ehook = 0;
		softmdot = 0;
		udreduce = 0;
		dual_lnorm = 0;
		dualatten = resiFirst = 0;
		frozenwgt = 0;
		icodefz = 1024;
		eternity = 0;
		stracode = 0;
		predlog = 0;
		enckern = 100;//reset �ǹ�
		nbatch2pseq = 0;
		samplingInterv = 0;
		numdid = rsc::ngpudev;//default�� ��ü gpu id ����
		didIndice = (intt *)xalloc(numdid * sizeof(intt));
		for(intt i = 0; i < numdid; i++) *(didIndice + i) = i;
		fastshape = 0;
		devArrange = 1;
		outback = 0;
		dualStep = 1;
		tcrCublas = 1;
		DEV_AVAIL_MARGIN = 134217727; //128M //24696061952//24G
		gpugear = 3;
		softfill = true;
		dualEmbed = 3;//0 - �Է¸� �Ӻ���, 1 - ��¸� �Ӻ���, 2 - �Է� ��� ���� �Ӹ޵�, 3 - �Է� ��� �Բ� �Ӻ���
		compbot = -1;
		directMax = 0;
	}
	~Trace();
	void migopt(Tracer *_sor)
	{
		Trace *sor = (Trace *)_sor;

		npAbrib = sor->npAbrib;
		lapType = sor->lapType;
		cpmMode = sor->cpmMode;
		prompt = sor->prompt;
		directExec = sor->directExec;


		convolving = sor->convolving;
		printBuilding = sor->printBuilding;
		ebatch_t = sor->ebatch_t;
		strideJump = sor->strideJump;
		kernelSize = sor->kernelSize;
		contigusRate = sor->contigusRate;
		completeConvolve = sor->completeConvolve;
		printLoad = sor->printLoad;
		dotType = sor->dotType; 
		dot_tDec = sor->dot_tDec;
		endecode = sor->endecode;
		zeroPadding = sor->zeroPadding;
		spotlat = sor->spotlat; 
		lowbound = sor->lowbound;
		relualpha = sor->relualpha;
		layer_norm = sor->layer_norm;
		spotOpt = sor->spotOpt;
		positional = sor->positional;
		actfType = sor->actfType;
		hidPrint = sor->hidPrint;
		bpPrint = sor->bpPrint;
		attsign = sor->attsign;
		optType = sor->optType;
		learningRate = sor->learningRate;
		pathPrint = sor->pathPrint;
		dbgStep = sor->dbgStep;
		stratusOpt = sor->stratusOpt;
		stratusOpt2 = sor->stratusOpt2;
		sourceInterv = sor->sourceInterv;
		trcpart = sor->trcpart;
		intergate = sor->intergate;
		actfAtt = sor->actfAtt;
		nlearnStop = sor->nlearnStop;
		elearnStop = sor->elearnStop;
		att_sDec = sor->att_sDec;
		ortho_single = sor->ortho_single;
		inzbound = sor->inzbound;
		tarzbound = sor->tarzbound;
		szslide2d = sor->szslide2d;
		szslide1d = sor->szslide1d;
		overlapSlide = sor->overlapSlide;
		slTrainInterv = sor->slTrainInterv;
		slPredInterv = sor->slPredInterv;
		scoopinner = sor->scoopinner;
		batchloss = sor->batchloss;
		printloss = sor->printloss;
		concurtrain = sor->concurtrain; 
		rTrainBound = sor->rTrainBound; 
		bandwidth = sor->bandwidth; 
		szDualReduce = sor->szDualReduce; 
		szkernDualNest = sor->szkernDualNest; 
		zstep = sor->zstep;
		rderivet = sor->rderivet;
		poskern = sor->poskern; 
		derivefeat = sor->derivefeat; 
		multihead = sor->multihead; 
		//dualloss = sor->dualloss; 
		outerMask = sor->outerMask; 
		dgenmgc = sor->dgenmgc; 
		reducea = sor->reducea; 
		rseed = sor->rseed; 
		externFinal = sor->externFinal;
		dualdff = sor->dualdff;
		dualChain = sor->dualChain;
		inplus = sor->inplus; 
		nblockChain = sor->nblockChain; 
		resiopt = sor->resiopt; 
		ehook = sor->ehook; 
		softmdot = sor->softmdot;
		udreduce = sor->udreduce;
		dual_lnorm = sor->dual_lnorm; 
		dualatten = sor->dualatten;
		resiFirst = sor->resiFirst; 
		frozenwgt = sor->frozenwgt; 
		icodefz = sor->icodefz;
		eternity = sor->eternity; 
		stracode = sor->stracode; 
		predlog = sor->predlog; 
		enckern = sor->enckern;
		rkernal2d = sor->rkernal2d; 
		nbatch2pseq = sor->nbatch2pseq; 
		samplingInterv = sor->samplingInterv;
		numdid = sor->numdid;
		for(intt i = 0; i < numdid; i++) *(didIndice + i) = *(sor->didIndice + i);
		didTrace = sor->didTrace; 
		fastshape = sor->fastshape; 
		devArrange = sor->devArrange; 
		outback = sor->outback;
		dualStep = sor->dualStep; 
		gradby = sor->gradby; 
		tcrCublas = sor->tcrCublas; 
		DEV_AVAIL_MARGIN = sor->DEV_AVAIL_MARGIN;
		gpugear = sor->gpugear; 
		softfill = sor->softfill; 
		dualEmbed = sor->dualEmbed;
		compbot = sor->compbot;
		directMax = sor->directMax;
	}
	/*void incFwFork(void)
	{
		LOCK_MUT_(mutTrc);
		if(++ifwFork > nfwFork) {
			exit(1);
		}
		UNLOCK_MUT_(mutTrc);
	}
	void incBwFork(void)
	{
		LOCK_MUT_(mutTrc);
		if(++ibwFork > nbwFork) {
			exit(1);
		}
		UNLOCK_MUT_(mutTrc);
	}*/
	/*void chkFwFork(SignalR *sr)
	{
		if(nfwFork == POSITIVE_INT_MAX) {
			xsleep(1);
			nfwFork = ifwFork;
		}
	}
	void chkBwFork(SignalR *sr)
	{
		if(nbwFork == POSITIVE_INT_MAX) {
			xsleep(1);
			nbwFork = ibwFork;
		}
	}*/
	void setmsg(bytet *msg)
	{
		if(msg == nullx) trcmsg[0] = '\0';
		else strcpy(trcmsg, msg);
	}
	void directx(bool on);
	void sizeBatch(intt sz)
	{
		batchPart = sz;
	}
	Tracer *division(void);
	void infeed(Flux *feed_fx, void *pdat, sytet tsrc, intt sz);
	void *xalloc(intt rsz)
	{
		return rsc::ralloc(hmalc, rsz);
	}
	void *bxalloc(intt rsz)
	{
		LOCK_MUT_(mutTrc);
		void *p = rsc::ralloc(hmalc, rsz);
		UNLOCK_MUT_(mutTrc);

		return p;
	}
	void installBaper(sytet stepw);
	void freeBaper(void);
	void portingGraph(Tracer *target);
	Flux *getFlux(Flux *sfx);
	Capsule *getCap(void)
	{
		Capsule *cap;

		if(poolCap) {
			GET_LIST(poolCap, cap);
			return cap;
		}
		return (Capsule *)xalloc(sizeof(Capsule));
	}
	void putCap(Capsule *cap_list)
	{
		if(cap_list == nullx) return;
		CAT_LIST(Capsule, poolCap, cap_list);
	}
	ApCap *getApc(void)
	{
		ApCap *cap;

		if(poolApc) {
			GET_LIST(poolApc, cap);
			return cap;
		}
		return (ApCap *)xalloc(sizeof(ApCap));
	}
	void putApc(ApCap *cap_list)
	{
		if(cap_list == nullx) return;
		CAT_LIST(ApCap, poolApc, cap_list);
	}
	NameScope *namescope(const bytet *nsm, bool reuse = 0)
	{
		NameScope *nsc;// = nsFocus->ptrChild;
		bytet buf[NAME_LENG];
		//for(;nsc; nsc = nsc->ptrRight) {
		//	if(!strcmp(nsc->nsName, nsm)) {
		//		nsFocus = nsc;
		//		return;
		//	}
		//}
		if(nsFocus && nsFocus->reuseScope) reuse = 1;
		if(nsFocus && nsm) {
			for(nsc = nsFocus->ptrChild;nsc; nsc = nsc->ptrRight) {
				if(!strcmp(strchr(nsc->nsName, ';') + 1, nsm)) break;
			}
			if(nsc) {
				if(reuse == 0) throwFault(-1, "exist name %s\n", nsm);
				nsFocus = nsc;
				return nsc;
			}
			sprintf(buf, "%s;%s", trcname, nsm);
		} else sprintf(buf, "%s;%d", trcname, seqNscope);
		nsc = (NameScope *)xalloc(sizeof(NameScope));
		memset(nsc, 0x00, sizeof(NameScope));
		nsc->nsName = (bytet *)xalloc(strleng(buf) + 1);
		strcpy(nsc->nsName, buf);
		nsc->reuseScope = reuse;
		seqNscope++;
		nsc->nameWeights = nsc->anonymWeights = nullptr;
		nsc->ptrParent = nsFocus;
		nsc->ptrChild = nullptr;
		if(nsFocus) APPEND_LIST(nsFocus->ptrChild, nsc);
		APPEND_LIST2(nsOrder, nsc);
		nsFocus = nsc;
		return nsc;
	}
	NameScope *namescope(intt i, bool reuse = 0)
	{
		bytet i_name[3];
		sprintf(i_name, "%d", i);
		return namescope((const bytet *)i_name, reuse);
	}
	void endscope(void)
	{
		if(nsFocus->ptrParent == nullx) throwFault(-1, "top abscent\n");

		if(nsFocus->ptrParent) nsFocus = nsFocus->ptrParent;
	}
	bytet *namewrite(const bytet *fname)
	{
		bytet buf[NAME_LENG], *p;
		if(fname) sprintf(buf, "%s/%s", nsFocus->nsName, fname);
		else sprintf(buf, "%s/%d", nsFocus->nsName, nsFocus->idNameFx);
		p = (bytet *)xalloc(strleng(buf) + 1);
		strcpy(p, buf);
		nsFocus->idNameFx++;
		return p;
	}
	NameScope *findnsc_depth(NameScope *nsc, bytet *nsm)
	{
		for(;nsc; nsc = nsc->ptrRight) {
			if(nsc->ptrChild) {
				NameScope *fsc = findnsc_depth(nsc->ptrChild, nsm);
				if(fsc) return fsc;
			}
			if(!strcmp(strchr(nsc->nsName, ';') + 1, nsm)) break;
		}
		return nsc;
	}
	NameScope *findnsc(bytet *nsm, sytet root)
	{
		NameScope *nsc;

		if(root) {//Ʈ���̼��� ��� ���ӽ����� �˻�
			for(nsc = nsOrder;nsc; nsc = nsc->ptrRight2) {
				if(!strcmp(strchr(nsc->nsName, ';') + 1, nsm)) break;
			}
			return nsc;
		} else {//�� ��Ŀ�� ���ӽ������� �� ���� ���ӽ��������� �־��� �̸��� ���ӽ����� �˻�
			if(!strcmp(strchr(nsFocus->nsName, ';') + 1, nsm)) return nsFocus;
			return findnsc_depth(nsFocus->ptrChild, nsm);
		}
	}
	Flux *findfxns(bytet *fname) //�� ���ӽ��������� ���� ����ġ �˻�
	{
		WeightList *wl = nsFocus->nameWeights;
		for(;wl; wl = wl->ptrRight) {
			char *p = strchr(wl->weightFx->fxName, '/') + 1;
			if(!strcmp(strchr(wl->weightFx->fxName, '/') + 1, fname)) break;
		}
		if(wl) {
			return wl->weightFx;
		} else return nullx;
	}
	void get_trainvar(NameScope *nsc, vector<Flux *> *rsp)
	{
		for(WeightList *wl = nsc->nameWeights; wl; wl = wl->ptrRight) {
			if(wl->weightFx->fxType == trainable) rsp->push_back(wl->weightFx);//oiro.
		}
		if(nsc->ptrChild) get_trainvar(nsc->ptrChild, rsp);
		if(nsc->ptrRight) get_trainvar(nsc->ptrRight, rsp);
	}
	vector<Flux *> *trainvar(NameScope *nsc) //�־��� ���ӽ������� �� ���Ͽ��� train var(�̹��� ��þ��ص� trainable�̸�) ������
	{
		vector<Flux *> *rsp = new vector<Flux *>;

		listv(rsp);

		if(nsc == nullx) nsc = nsFocus;
		for(WeightList *wl = nsc->nameWeights; wl; wl = wl->ptrRight) {
			if(wl->weightFx->fxType == trainable) rsp->push_back(wl->weightFx);//oiro.
			rsp->push_back(wl->weightFx);
		}
		if(nsc->ptrChild) get_trainvar(nsc->ptrChild, rsp);

		return rsp;
	}
	void listw(Flux *fx, bool anonym)
	{
		WeightList *wl = (WeightList *)xalloc(sizeof(WeightList));

		wl->weightFx = fx;
		if(anonym) {
			APPEND_LIST(nsFocus->anonymWeights, wl);
		} else APPEND_LIST(nsFocus->nameWeights, wl);
	}
	Capsule *elistw(bool anonym, intt fx_type)
	{
		NameScope *nsc = nsOrder;
		WeightList *wl;
		Capsule *cap, *lcap = nullx;

		for(;nsc; nsc = nsc->ptrRight2) {
			for(wl = (anonym ? nsc->anonymWeights : nsc->nameWeights);wl; wl = wl->ptrRight) {
				//if(wl->weightFx->backwv != bwVersion) {//�����Լ��� �ԷµǴ� Ÿ�� fx�� �Է����� �����õ��� �ʾ�
				//	continue;//vsync���� ���� �������� �����Ƿ� ���� ���� üũ���� �ɷ� ��ŵ�ȴ�.(����)
				//}
				if(fx_type >= 0 && wl->weightFx->fxType != fx_type) continue;
				cap = (Capsule *)xalloc(sizeof(Capsule));//���⼭ �Ҵ�ȴ� cap�� divObj ������ �ʿ�����Ƿ� �ܼ� �Ҵ�.
				cap->vcaps = wl->weightFx;
				APPEND_LIST(lcap, cap);
			}
		}
		return lcap;
	}
	Capsule *persistw(void) //oiro.
	{
		NameScope *nsc = nsOrder;
		WeightList *wl;
		Capsule *cap, *lcap = nullx;

		for(; nsc; nsc = nsc->ptrRight2) {
			for(wl = nsc->nameWeights; wl; wl = wl->ptrRight) {
				if(wl->weightFx->fxType > trainable) continue;
				cap = (Capsule *)xalloc(sizeof(Capsule));//���⼭ �Ҵ�ȴ� cap�� divObj ������ �ʿ�����Ƿ� �ܼ� �Ҵ�.
				cap->vcaps = wl->weightFx;
				APPEND_LIST(lcap, cap);
			}
			for(wl = nsc->anonymWeights; wl; wl = wl->ptrRight) {
				if(wl->weightFx->fxType > trainable) continue;
				cap = (Capsule *)xalloc(sizeof(Capsule));//���⼭ �Ҵ�ȴ� cap�� divObj ������ �ʿ�����Ƿ� �ܼ� �Ҵ�.
				cap->vcaps = wl->weightFx;
				APPEND_LIST(lcap, cap);
			}
		}
		return lcap;
	}
	void reset_grads(bool anonym)
	{
		NameScope *nsc = nsOrder;
		WeightList *wl;
		for(;nsc; nsc = nsc->ptrRight2) {
			for(wl = (anonym ? nsc->anonymWeights : nsc->nameWeights);wl; wl = wl->ptrRight) {
				if(wl->weightFx->backwv != bwVersion) {//�����Լ��� �ԷµǴ� Ÿ�� fx�� ��Ƽ�������� ���fx�� ap�� �Է�����
					continue;//�����õ��� �ʾ� vsync���� ���� �������� �����Ƿ� ���� ���� üũ���� �ɷ� ��ŵ�ȴ�.(����)
				}
				if(didTrace != wl->weightFx->groundid(1)) {//backwardDevArrange������ ����ġ��
					didTrace = wl->weightFx->groundid(1); //��� �׶���� ����ȴ�.
					CudaDevSet(didTrace);//Ʈ���̽��� run�Լ� ���� ������� ��� �����ȴ�.
				}
				wl->weightFx->resetGrad();
				wl->weightFx->ibRefer = 0;
				//printf("2: %p: %s\n", wl->weightFx, wl->weightFx->fxName);
			}
		}
	}
	void resetGrads(void)
	{
		reset_grads(true);
		reset_grads(false);
	}
	void listv(vector<Flux *> *pvec)
	{
		VectorTag *vtag = (VectorTag *)xalloc(sizeof(VectorTag));

		vtag->pvector = pvec;
		vtag->ptrPrev = vtLast;
		vtLast = vtag;
	}
	void listm(Matrixr *mx);
	void npset(intt n)
	{
		npAbrib = n;
	}
	void lapset(sytet lap)
	{
		lapType = lap;
	}
	void gprset(floatt d)
	{
		memput::glb::rsc::srutra->gpuMultiple = d;
	}
	void modeset(sytet cpm)
	{
		cpmMode = cpm;
	}
	Matrixr *instMatrix(Matrixr *mtx, ubytet qt, intt ndim, intt *axid, bool o_mut, intt gid = -1, Matrixr *mast = nullptr);
	void promptMode(bool on)
	{
		prompt = on;
	}
	void listOrigin(Flux *ori);
	void initArrange(void);
	void initVsync(void);
	void listWeight(void);
	void vSync(Flux *endp);
	void resetApply(void);
	void init_train(void);
	void run(Flux *target);
	void run(vector<Flux *> target);
	void run(vector<Flux *> *target);
	void reposet(bytet *rpath)
	{
		strcpy(rPath, rpath);
	}
	void setgpudev(intt gid)//Ư�� ����̽��� ��õ��� �ʰ� ���Ÿ� �Ҵ�ɶ��� ����̽� ����
	{
		if(didTrace != gid) {
			didTrace = gid;
			CudaDevSet(gid);
		}
	}
	void focusgpudev(intt gid = -1)
	{
		if(gid < 0) gid = didTrace;

		setgpudev(gid);
	}
	void multiDevice(vector<intt> axid)
	{
		intt i = 0, gid[256];

		for(auto iter : axid) gid[i++] = iter;

		if(i > numdid) throwFault(-1, "gpu over\n");
		if(gid[0] < 0) return;//��ü ����̽� ����, default�� �����������Ƿ� ��ŵ

		numdid = i;
		for(i = 0; i < numdid; i++) *(didIndice + i) = gid[i];

		if(*(didIndice + 0) != didTrace) {//��õ� ����̽� ����Ʈ�� ù��° ���̵� ����.
			didTrace = *(didIndice + 0);//setgpudev�� �����Ȱ��� �̰����� ��ä��
			CudaDevSet(didTrace);
		}
	}
	TContext *trcCxt(intt gid)
	{
		return tcxtarr->getgputxt(gid, this);
	}
	//��ġ����� �����ҷ��� gate, go, end�� feed�Լ��� �Է��ϰų� resizing5�� ��ġ����� �����Ѵ�.
	//dual encoder������ chatbot�������� <s> ������ū�� �����Ϳ��� �����ϰ� �����Ѵ�. ������ū�� 0��
	//���μ� auto regression���� ���� ��ū���� ���Ǳ� ������, ���̳������� ����� ��� embedim��
	//��� ���ڴ��� �Ӻ��� ������� �ݵ�� �����ϰ� latent_sz, indiscret�� ����� ������ ���̳���
	//�����ڿ��� ����� �Ͱ� �ٸ��� �� ��� �����Ѵ�. outdiscret�� ���̳��� �����ڿ��� ��� ���ڴ���
	//���� ��õǹǷ� ���� ���� �ʿ����.
	void setbygate(Flux *gate, intt nout, Flux *go, Flux *end, intt embedim = -1, intt latent_sz = -1, intt indiscret = -1)
	{
		if(nout > gate->fshape[1]) throwFault(-1, "out size(%d) is longer than by gate(%d) error \n", nout, gate->fshape[1]);
		bygate = gate;//�Է� + �߷���� pair Ȥ�� �߷����, �߷� ����� <s> + Ÿ��
		nbyout = nout;//�߷� ��� ����, �� ���� ���̰���Ʈ ���������̸� ���⿡ �Է��� ����
		//�Է��� ���ʸ��� ingate�� �־��� ���ʸ��� �Է� ���� ������ ����Ǿ� pcode�� ������ڴ��� �Էµ�.
		gotok = go;//�н��� �� ���� �ʿ����.
		endtok = end;//�߷��� ��ġ ������ �� ���� �ʿ����.
		sembedn = embedim;
		slatent = latent_sz;
	}
	Flux *migbygate(Trace *src)
	{
		bygate = flux(this, src->bygate, variable);
		nbyout = src->nbyout;
		gotok = src->gotok;
		endtok = src->endtok;
		sembedn = src->sembedn;
		slatent = src->slatent;
		
		return bygate;
	}
	void traceopt(intt i, doublet v)
	{
		switch(i) {
		case 0:
			convolving = (intt)v;
			break;
		case 1:
			printBuilding = (intt)v;
			break;
		case 2:
			ebatch_t = (intt)v;
			break;
		case 3:
			strideJump = (intt)v;
			break;
		case 4:
			kernelSize = (intt)v;
			break;
		case 5:
			contigusRate = v;
			break;
		case 6:
			completeConvolve = (intt)v;//1�Ƹ� ���ڵ������� splot lat����, 2�̸� ���ڵ�/���ڵ����� splot lat����
			break;
		case 7:
			printLoad = (intt)v;
			break;
		case 8:
			if((intt)v == 1) BLOCK_SIZE = SMALL_BLOCK;
			else BLOCK_SIZE = 2 * SMALL_BLOCK;
			break;
		case 9:
			dotType = (intt)v;
			break;
		case 10:
			dot_tDec = (intt)v;
			break;
		case 11:
			endecode = (intt)v;
			break;
		case 12:
			spotlat = (intt)v;
			break;
		case 13:
			lowbound = v;
			break;
		case 14:
			CPU_CORE_N = (intt)v;
			break;
		case 15:
			layer_norm = (intt)v;
			break;
		case 16:
			spotOpt = (intt)v;
			break;
		case 17:
			positional = (intt)v;
			break;
		case 18:
			actfType = (intt)v;
			break;
		case 19:
			hidPrint = (intt)v;
			break;
		case 20:
			bpPrint = (intt)v;
			break;
		case 21:
			attsign = (intt)v;//spot opt�� 3�� ���ǰ� �Բ� 2�̸� ��� ���ټ� ���� 
			break;// 1�̸� ���ڴ�-���ڴ� ���� ��ũ��ũ(�ڱⱸ���н�)������ ���� �ʰ� ���ڴ��� ���� ��ũ��ũ(�ҽ������н�)������ ����.
		case 22:
			optType = (intt)v;
			break;
		case 23:
			learningRate = v;
			break;
		case 24:
			pathPrint = (intt)v;
			break;
		case 25:
			dbgStep = (sytet)v;
			break;
		case 26:
			stratusOpt = (intt)v;
			break;
		case 27:
			stratusOpt2 = (intt)v;
			break;
		case 28:
			sourceInterv = (intt)v;
			break;
		case 29:
			trcpart = (intt)v;
			break;
		case 30:
			intergate = (intt)v;
			break;
		case 31:
			actfAtt = (intt)v;
			break;
		case 32:
			nlearnStop = (intt)v;
			break;
		case 33:
			elearnStop = v;
			break;
		case 34:
			att_sDec = (intt)v;
			break;
		case 35:
			ortho_single = (intt)v;
			break;
		case 36:
			break;
		case 37:
			break;
		case 38:
			break;
		case 39:
			break;
		case 40:
			break;
		case 41:
			break;
		case 42:
			printloss = (intt)v;
			break;
		case 43:
			inzbound = (intt)v;
			break;
		case 44:
			tarzbound = (intt)v;
			break;
		case 45:
			szslide2d = (intt)v;
			break;
		case 46:
			szslide1d = (intt)v;
			break;
		case 47:
			overlapSlide = (intt)v;
			break;
		case 48:
			slTrainInterv = (intt)v;
			break;
		case 49:
			slPredInterv = (intt)v;
			break;
		case 50:
			scoopinner = (intt)v;
			break;
		case 51:
			batchloss = (intt)v;
			break;
		case 52:
			concurtrain = (intt)v;
			break;
		case 53://dynagen default ����
			convolving = 8;//Ŀ�� ����� 8�϶� �ϴ� �ִ� ���������μ� Ŀ�δ� 1�� �ǰ��ϰ� �ο�ٿ��� �����ϰ��Ѵ�.
			if((intt)v == 1) lowbound = 0.125;//1d, 0.25 ��� �����ڵ������� 2, 0.125�� �����Ͽ� 1�� ���� ����
			else if((intt)v == 2) lowbound = 0.5;//2d
			else throwFault(-1, "non def\n");
			dot_tDec = 1;//TENSOR_DOT.���⼭ �� �ΰ� �ɼǰ��� �����Ҷ� ���ڴ� 2��(����) �н�(translate�϶�)�� 
			dotType = 4;//ORTHO_DOT.//�ߵȴ�. �ٸ����� �⺻���϶��� �ξ� �н��� �ߵ����� ���ڴ� 2���н��� ���� 
			//�̰��� �����Ѵ�. ���� ���ڵ��� �Ұ�쿡�� �⺻��(�Ѵ� 0[stride dot])���� �����Ͽ� �����Ѵ�.
			derivefeat = 0;//������ڴ��� ����ɶ� �ǹ�, �̰��� 0�� �ϸ� feature�� �������� 
			//�Ǿ� �� ������ ��ġ�ȴ�. ORTHO_DOT�����̸� dynagen�� ���ڵ������� feature�� ���������
			//�� �ǰ� �ɼ����� derivefeat�� 1�� �����Ͽ��� ������ڴ����� pcode�� feature���� 
			break; //����� derive������� �ڵ���ȯ�ǹǷ� �Ŀ� 1�� �����ص� ��������.
		case 54:
			rTrainBound = v;
			break; 
		case 55:
			autoRegress = v;
			break; 
		case 56:
			bandwidth = v; 
			break;
		case 57:
			szDualReduce = v;
			break;
		case 58:
			relualpha = v; 
			break;
		case 59:
			szkernDualNest = v;
			break;
		case 60:
			zstep = (intt)v;
			break;
		case 61:
			rderivet = (intt)v;
			break;
		case 62:
			poskern = (intt)v;
			break; 
		case 63:
			derivefeat = (intt)v;
			break; 
		case 64:
			multihead = (intt)v; 
			break;
		case 65:
			//dualloss = (intt)v;
			break;
		case 66:
			outerMask = (intt)v; 
			break; 
		case 67:
			dgenmgc = (intt)v;
			break;
		case 68:
			reducea = (intt)v;
			break; 
		case 69:
			rseed = (intt)v;
			break;
		case 70:
			externFinal = (intt)v;
			break;
		case 71:
			zeroPadding = (intt)v; 
			break;
		case 72:
			dualdff = (intt)v;
			break;
		case 73:
			dualChain = (intt)v;
			break;
		case 74:
			inplus = (intt)v;
			break;
		case 75:
			nblockChain = (intt)v;
			break; 
		case 76:
			resiopt = (intt)v; 
			break; 
		case 77:
			ehook = (intt)v;
			break;
		case 78:
			softmdot = (intt)v;
			break; 
		case 79:
			udreduce = (intt)v;
			break;
		case 80:
			dual_lnorm = (intt)v;
			break;
		case 81:
			dualatten = (intt)v; 
			break;
		case 82:
			resiFirst = (intt)v;
			break; 
		case 83:
			frozenwgt = (intt)v;
			break;
		case 84:
			icodefz = (intt)v; 
			break; 
		case 85:
			eternity = (intt)v;
			break;
		case 86:
			stracode = (intt)v;
			break; 
		case 87:
			predlog = (intt)v;
			break; 
		case 88:
			enckern = (intt)v;
			break;
		case 89:
			rkernal2d = v;
			break;
		case 90:
			nbatch2pseq = (intt)v;
			break;
		case 91:
			samplingInterv = (intt)v;
			break;
		case 92:
			fastshape = (intt)v;
			break;
		case 93:
			devArrange = (intt)v;
			break; 
		case 94:
			outback = (intt)v;
			break;
		case 95:
			dualStep = (intt)v; 
			break;
		case 96:
			gradby = (intt)v;
			break; 
		case 97:
			tcrCublas = (intt)v; 
			break;
		case 98:
			DEV_AVAIL_MARGIN = (longt)v; 
			break;
		case 99://0 : �ʱ� ������ gpu������ ����, 1 : ���� gpu�� �޸𸮰� ���ڸ��� ���� gpu���� ����
			gpugear = (intt)v;//2 : split �б⿡���� ���� gpu ��Ƽ ����. 3 : ��� �б⸶�� ���� gpu ��Ƽ ����
			break; 
		case 100:
			softfill = (intt)v;
			break;
		case 101:
			dualEmbed = (intt)v; 
			break;
		case 102:
			compbot = (intt)v;
			break;
		case 103:
			directMax = (intt)v;
			break;
		default:
			throwFault(-1, "non def trace option\n");
		}
	}
	void saveWeight(void);
	intt loadWeight(void);
	void truncWeight(void);
	void printWeight(void);
	Flux *tcr_combination(Flux *in, intx width, intx stride, doublex exc_contig_r, sytet zero_pading, bool one_batch);
	Flux *tcr_combination2(Flux *in, intx width, intx stride, doublex exc_contig_r, sytet zero_pading, bool one_batch);
};
//���� 30�� ���� �׽�Ʈ
/* stratus - 50% ��Ȯ��, ���� ������ 128, Ÿ�� �������� ���ڵ�1����(32->8->32) �ڱⱸ���н���Ų��  
			Ÿ�ٽ����� ���ڴ� ���������ڵ带 �ҽ��� ���ڵ�2����(8->32) �н��ϰ� Ÿ������ �ҽ��������(32->8->8) �н�, 
			���ڵ�2���� ������ 0.02%�� ������ �� ������ �ҽ� ����� ������ ���ؼ� ��ä ��Ȯ���� 50%������ ���´�.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);//���ڴ����� ����Ǵ� dotŸ��
TRACER(tcr)->traceopt(6, 1);//�������� �ɼ�, 1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����(��������), 2�̸� ���ڴ�-���ڴ� ��ũ��ũ���� splot lat����
TRACER(tcr)->traceopt(9, 4);//�� ��ü�� ������� dotŸ��
TRACER(tcr)->traceopt(26, 3);//�ҽ����ڴ� �����, Ÿ�Խ����� ���ڴ�-���ڴ� �ڱⱸ���н����� ���Ͽ� ���ڵ�2 ���� �ϳ� �� �����Ͽ� �н�. stratus �� ���� �ɼ�
TRACER(tcr)->traceopt(27, 1);//�ҽ��� Ÿ���� �����ڵ� ���� �����Լ��� 0�̸� �������ƽ�ũ�ν���Ʈ����, 1�̸� mse
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 10~20% ��Ȯ��, ���� ������ 128, �ҽ������ ���ڴ��� �������� ���ټ� �����ϰ�(32->8->8[attention]) Ÿ�ٽ������� 
			�ڱⱸ���н����� ���ڴ�-���ڴ�2��(32->8->32) �����Ѵ�. ���ڴ�-���ڴ�2 �ڱⱸ���н��� ������ ���� �ʴ´�.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(11, 1);//���ڴ�-���ڴ� ������ ���ڴ��� ���ڴ�2�� ����
TRACER(tcr)->traceopt(6, 1);//1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����, 2�̸� ���ڴ�-���ڴ� ��ũ��ũ�� ���ڴ����� splot lat����
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(16, 3);//���ڴ����̴� ���ڴ�-���ڴ����̴� Ư���ɼ�(���ټ�, �����ڵ�ũ���)���� ���� ���࿡���� ��ȿ�ϰ�
TRACER(tcr)->traceopt(21, 1);//���ټ��� �� 6�� �Բ� �̰��� 1�Ƹ� ���ڴ���������, 2�̸� ���ڴ�-���ڴ����� ���ڴ����� ����
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 10~% ��Ȯ��, ���� ������ 64, �ҽ������ ���ڴ��� �������� ���ټ� �����ϰ�(32->8->8[attention]) Ÿ�ٽ������� 
			�ڱⱸ���н����� ���ڴ�-���ڴ�2��(32->8->32[attention]) �����Ѵ�. ���ڴ�-���ڴ�2 �ڱⱸ���н��� ������ ���� �ʴ´�.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(11, 1);
TRACER(tcr)->traceopt(6, 1);//1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����, 2�̸� ���ڴ�-���ڴ� ��ũ��ũ���� splot lat����
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(16, 3);
TRACER(tcr)->traceopt(21, 1);//���ټ��� �� 6�� ���� ���� ���� ����.
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);
TRACER(tcr)->traceopt(34, 1);//���ڴ����� ���ټ� ���� �ɼ�, 1�̸� ������ ù��° ���ڴ������� 2�̸� ���ڴ� ��ü���� ���ټ� ����

// stratus - 90% ��Ȯ��, ���� ������ 64, �ҽ������(32->8->8) ���ڴ��� �������� �����ϰ� Ÿ�ٽ������� �ڱⱸ���н�����
			���ڴ�-���ڴ���(32->8->32) �����Ѵ�.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 1);//1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����, 2�̸� ���ڴ�-���ڴ� ��ũ��ũ�� ���ڴ����� splot lat����
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 70% ��Ȯ��, ���� ������ 64, �ҽ������(32->8->8) ���ڴ��� �������� �����ϰ� Ÿ�ٽ������� �ڱⱸ���н�����
			���ڴ�-���ڴ���(32->8->8->32) �����Ѵ�.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 2);//1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����, 2�̸� ���ڴ�-���ڴ� ��ũ��ũ���� splot lat����
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 80~90% ��Ȯ��, ���� ������ 64, �ҽ������(32->8->8[attention]) ���ڴ��� �������� �����ϰ� Ÿ�ٽ������� �ڱⱸ���н�����
			���ڴ�-���ڴ���(32->8->8[attention]->32) �����Ѵ�.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 2);//1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����, 2�̸� ���ڴ�-���ڴ� ��ũ��ũ���� splot lat����
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(16, 3);
TRACER(tcr)->traceopt(21, 2);//���ټ��� �� 6�� ���� ���� ���� ����.
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 70 ~ 80% ��Ȯ��, ���� ������ 64, �ҽ������ ���ڴ���(32->8->8) �������� �����ϰ� Ÿ�ٽ������� �ڱⱸ���н�����
			���ڴ�-���ټǵ��ڴ���(32->8->32[attention]) �����Ѵ�. ���ڴ�-���ټǵ��ڴ� �н��� �ұ�Ģ�Ͽ� 0.005�� 100�� �����Ͽ�
			�������� ����	�ڱⱸ���н����� ���ڴ� ���������ڵ尡 �������� �ʾƼ� �ҽ� ������� ��Ȯ���� ��������.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 1);//1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����, 2�̸� ���ڴ�-���ڴ� ��ũ��ũ���� splot lat����
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.005);
TRACER(tcr)->traceopt(34, 1);

//stratus - ���� ������ 64, Ÿ�ٽ������� �ڱⱸ���н����� ���ڴ�-���ڴ�2��(32->8->32[attention]) �����ϰ�
			�򰡶� �Է½������� �����ϰ��� �ϴ� �������� ���Ѿ����ڵ带 �ְ� ���� ����� �����Ѵ�.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(11, 1);
TRACER(tcr)->traceopt(6, 1);//1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����, 2�̸� ���ڴ�-���ڴ� ��ũ��ũ���� splot lat����
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(16, 3);
TRACER(tcr)->traceopt(21, 1);//���ټ��� �� 6�� ���� ���� ���� ����.
TRACER(tcr)->traceopt(26, -1);
TRACER(tcr)->traceopt(34, 1);



// generic - 95% ��Ȯ��, ���� ������ 64, (32 -> 8 -> 32)
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
//TRACER(tcr)->traceopt(6, 1);//���ڴ� only ��ũ��ũ�� �����Ƿ� �ǹ̾���
TRACER(tcr)->traceopt(9, 4);

// generic - 20% ��Ȯ��, ���� ������ 64, 32 -> 8 -> 8 -> 32)
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 2);//1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����, 2�̸� ���ڴ�-���ڴ� ��ũ��ũ���� splot lat����
TRACER(tcr)->traceopt(9, 4);

// generic - 80% ��Ȯ��, ���� ������ 64, (32 -> 8 -> 8[attention] -> 32) ����
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 2);//1�Ƹ� ���ڴ� ��Ʈ��ũ������ splot lat����, 2�̸� ���ڴ�-���ڴ� ��ũ��ũ���� splot lat����
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(16, 3);
TRACER(tcr)->traceopt(21, 2);//���ټ��� �� 6�� ���� ���� ���� ����.

*/