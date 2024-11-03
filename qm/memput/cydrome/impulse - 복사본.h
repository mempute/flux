#pragma once
//injection, ejection, organail, isleloop, telepulse
#include "islet.h"
#include "misc/fio.h"

class Isleloop;
class Impulse;
class Dynagen;

class SyncInvoke : public Typer {
public:
	sytet sockid;
	Isleloop *invokeisle;
	SyncInvoke *ptrLeft, *ptrRight;
	SyncInvoke(Isleloop *isle, sytet socid)
	{
		invokeisle = isle;
		sockid = socid;
	}
};
class Socketisle : public Typer {
public:
	sytet sockid;
	Isleloop *destisle;
	Socketisle *ptrLeft, *ptrRight;
};
class Impulse : public Cell {//�̹� �н��� ���� ���� ����� �ٸ� ���� �н� �Է����� �Ҷ� impulse�� �и��Ͽ� �����Ѵ�.
public:
	Trace *impTcr;
	QueryContext impAllocator, *qcimp;
	intt ntrainStep;//, ntrainBatch;
	SyncInvoke *synchor;
	intt sample_size, batch_size, ntrainisle, ndecisle, itrainisle, idonetrain;
	longx syncIndice;//�Է��� �ΰ� �� �͵��� �� �������� �����ؾ� �۵��Ѵ�.
	bool tadapt, predwithloss, finalSync, runPredict;
	sytet trainstep, init_trainstep;
	Isleloop *islelist;
	Dynagen *dgenlist;
	hmutex mutimp;
	lsemx semimp;
	CydChain *cydimp;
	bytet impname[128], imppath[128];
	intt cntdgen;
	Generic accucell;// Cell *accucell;
	Tracer *accutrc;
	Impulse(Tracer *tcr, const bytet *name = nullx)
	{
		impTcr = (Trace *)tcr;//�޸� �Ҵ�� �����ؼ��� �����Ҷ��� �����Ǿ�� �Ѵ�.
		qcimp = &impAllocator;
		InitSelPage(qcimp);
		synchor = nullx;
		syncIndice = 0;
		islelist = nullx;
		dgenlist = nullx;
		tadapt = 0;
		ntrainisle = ndecisle = idonetrain = 0;
		init_trainstep = 1;
		runPredict = 0;
		intt rv;
		CRE_MUT_(mutimp, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre mut fail");
		}
		cydimp = rsc::cydrome->getcydid();
		CRE_LSEM_(semimp, 0, cydimp->sem_iden, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre sem fail\n");
		}
		predwithloss = 0;
		if(strcmp(impTcr->rPath, ".")) strcpy(imppath, impTcr->rPath);
		else strcpy(imppath, ".");
		if(name) {
			if(strlen(name) >= 128) throwFault(-1, (bytex *)"name over fail\n");
			strcpy(impname, name);
		} else strcpy(impname, "_impluse");
		cntdgen = 0;
		finalSync = 0;
		accucell.makeAnet(impTcr, 0, -1);
	}
	~Impulse();
	void *impalloc(size_t size)
	{
		bytex *rp;

		SelAlloc(qcimp, size, rp);

		return rp;
	}
	void trainStep(longx v)
	{
		ntrainStep = v;
	}
	void countDoneTrain(void)
	{
		LOCK_MUT_(mutimp);
		idonetrain++;
		UNLOCK_MUT_(mutimp);
	}
	void sigtraining(void)
	{
		if(finalSync) {//�����н� 2�ܰ��̸� ���������� �н� ����� isle�� �̹� ���� �н� �Ϸ� �ñ׳�
			LOCK_MUT_(mutimp);
			if(++itrainisle == ntrainisle) {
				SIG_LSEM_(semimp);
			}
			UNLOCK_MUT_(mutimp);
		}
	}
	void waittraining(void)
	{
		if(finalSync) WAIT_LSEM_(semimp);
	}
	void regifeed(Isleloop *begin_isle, sytet socid)
	{
		SyncInvoke *s = new(impTcr) SyncInvoke(begin_isle, socid);

		APPEND_LIST(synchor, s);
	}
	void recording(void);
	void retrain(void);
	void regiisle(Isleloop *isle);
	void regidgen(Dynagen *dgen);
	void coupleisleThr(void);
	Flux *train(intt *n_train = 0);
	Flux *predict(Flux **ploss = nullx);
	Flux *loss2(void) { return nullx; }
	void accuracy(Flux *&predicts, Flux *&targets, intt discrete_out) //��Ȯ�� ���� �׶��� ����
	{//�н��Ҷ��� �Է°� Ÿ�� ����Ʈ�� ������� �ʰ� ���� ������� �δ� ���� ���� �ڵ� ��Ȯ���� ����Ҷ��� 
		//�н� ����¶��� �����Ϳ� �����ڵ��� ���°� Ʋ���� �����̴�.
		accucell.accuracy(predicts, targets, discrete_out);//ipare.�ʿ��ϸ� �Ʒ������� �Ѵ�.
		//predicts = flux(accutrc, predicts, variable);
		//targets = flux(accutrc, targets, variable);
		//accucell->accuracy(predicts, targets, discrete_out);
	}
	Flux *measureAccuracy(void) //��Ȯ�� ���� �׶��� ����.
	{
		return accucell.measureAccuracy();//accucell->measureAccuracy();
	}
};
#define NHNDI	0 //not head, not discrete, input
#define NHNDT	1 //not head, not discrete, target
#define NHNDC	2 //not head, not discrete, couple
#define NHOC	3 //not head, output couple
#define HID		4 //head input discrete
#define HOD		5 //head output discrete
#define HAD		6 //head input, output all discrete


#define ENCODE_ISLE	0
#define DECODE_ISLE	1
#define COUPLE_ISLE	2
class Dynagen;
class Isleloop : public Typer {
public:
	sytet isleType;
	sytet headisle, actcode, noperand, cntoper, cntwait, islebusy;
	sytet lossIndice, nprotect;
	bytex isleName[128];
	Trace *isletrc;
	Cell *islenet;//stratus: �ҽ� - Ÿ�� ����, generic: �������ڴ�, decoder: �ڱ⺹�� ��
	Impulse *isleimp;
	Dynagen *isledgen;
	Socketisle *isleRoute;
	longx syncurIndice, rcvsynIndice;
	intt cntTStep, nTStep, szlatent, isindisc, isoutdisc, szembede, icntTrain, icumTrain;
	floatt rLeaning, sumloss, minloss;
	intt istableMean, istableMean2;
	Flux *predloss, *islein[2], *isleout, *scoopout, *extingate, *exttargate;
	bool fintrain, sockIndice[2], tadapt, alignoper, finalisle, fowardit, neverTrained;
	Cydrome *cydisle;
	CydChain *cydcijec, *cydcpeek;
	hmutex mutisle;
	lsemx semijec, sempeek;
	Isleloop *ptrLeft, *ptrRight;

	Isleloop(Impulse *imp, Dynagen *dgen, sytet head, bytet *name);

	~Isleloop()
	{
		rmloop();
	}
	virtual void rmloop()
	{
		inject(nullx, nullx, -1, 0);//isle ���� ���� �� ������ ���ᶧ���� ���
		CLOSE_MUT_(mutisle);
		CLOSE_LSEM_(semijec);
		cydisle->putcydid(cydcijec);
		CLOSE_LSEM_(sempeek);
		cydisle->putcydid(cydcpeek);
		delete isletrc;
	}
	//���߿� buildislet, trainislet, predislet�� ���� ���μ����� �����ϰ� �Ǹ� in, tar�� �����Ͽ� ������ �Ѵ�.
	//�̵� �Լ��� ���ÿ� �����ϸ� in, tar�� �Է� �÷���(islein[2])�� �����������Ƿ� �ٽ� ������ �ʿ����.
	virtual Flux *trainislet(Flux *in, Flux *tar)
	{
		return ((Generic *)islenet)->train();
	}
	virtual Flux *predislet(Flux *in, Flux **loss)
	{
		return ((Generic *)islenet)->predict(loss);
	}
	virtual void trainloop(longx sync_indice)
	{
		if(fintrain) {//�� �ܰ� �н��� ��������
			predloop(sync_indice);//���� ���� ������ �н��� ���� �� ������ ���� �� ������ ������
			return;
		}
		Flux *r = trainislet(islein[0], islein[1]);
		finaltrain(r);
		lossout(r, "_train");
		if(isletrc->concurtrain && fowardit) {//�����н��̸� ���� ������ ������(���ڴ��� �ƴ�)
			predloop(sync_indice);//�� �� �� ���� ������
		}
	}
	virtual void predloop(longx sync_indice);
	virtual Flux *restoreOrigin(Flux *dout)
	{
		return dout;
	}
	void printisle(Flux *pf, const bytet *msg)
	{
		LOCK_MUT_(isleimp->mutimp);
		printf("%s [%p] %s\n", isleName, this, msg);
		pf->printo(2, 10);
		UNLOCK_MUT_(isleimp->mutimp);
	}
	void setFinal(bool on)
	{
		if(isletrc->externFinal) neverTrained = on;
		finalisle = on;
	}
	void decreaseProtect(void);
	void finaltrain(Flux *loss);
	void lossout(Flux *loss, const bytet *action);
	void setdgencoup(Dynagen *dgen);
	void regiRouteTo(Isleloop *next, sytet socid)
	{
		Socketisle *s = new(isleimp->impTcr) Socketisle;
		s->destisle = next;
		s->sockid = socid;
		APPEND_LIST(isleRoute, s);
		if(next->isleType != DECODE_ISLE) fowardit = 1;
	}
	void eject(Flux *preds, longx sync_indice)
	{
		for(Socketisle *s = isleRoute;s; s = s->ptrRight) {
			//�н� 1�ܰ迡���� ���ڴ����� ���ڴ��� eedp)���� inject�� ���� �������ϹǷ� ���⼭�� ��ŵ�ϰ�
			//2�ܰ� �н� Ȥ�� �򰡴ܰ迡���� ���ڴ����� ������ ���� Ŀ�� isle->���� ���ڴ�, ���ڴ�->���ڴ����� �ȴ�.
			if(isleimp->trainstep > 0 && s->destisle->isleType == DECODE_ISLE) continue;
			//�߷� �������� ���� ȹ������ �ʴ� �ɼ��̸� Ÿ�ٰ��� ���Ե��� �����Ƿ� �Է¸����� ����ǰ� -1�� �Ѵ�.
			s->destisle->inject(preds, nullx, sync_indice, isleimp->predwithloss || s->sockid == -2 ? s->sockid : -1);
		}
	}
#define CHKSIG_ISLE(snd_peek) {\
	if(snd_peek) SIG_LSEM_(sempeek);\
	if(cntwait) {\
		SIG_USEM_(semijec, cntwait);\
		cntwait = 0;\
	}\
}
	//�� isle�� ���� �ڽ��� tracer�� ������ ���������İ� ���������� ����ǹǷ� �Է� �� Ÿ�� �÷����� feedf�� ���� �����ؾ��Ѵ�.
	//�ȱ׷��� Ʈ���̽����� �׶����� ����ȴ�.
	void inject(Flux *in, Flux *tar, longx sync_indice, sytet i_socket) //�� isle�� tar�� ���̸� ���ڴ� �ƴϸ� ���ڴ�
	{
		bool rcv_wait = 0;
		while(1) {
			LOCK_MUT_(mutisle);
			if(syncurIndice < 0 || (isleimp->trainstep == 0 && isleimp->syncIndice != sync_indice)) {
				UNLOCK_MUT_(mutisle);//ieie.�����̰ų� �� �ܰ�����̸� ���isle���� ���� ��ũ�� ����ǹǷ� �� ����  
				return;//�򰡽����Ҷ� ���� Ʋ���� ���� �н��ܰ��� �÷����� ���� �߰��� �����ִ� �͵��̹Ƿ� �����Ѵ�.
			}
			if(islebusy || rcv_wait) {//isle�� �������̸� ���
				cntwait++;
				UNLOCK_MUT_(mutisle);
				//printf("111-1: %p %d\n", this, cntwait);
				WAIT_USEM_(semijec);
				//printf("111-2: %p\n", this);
				if(rcv_wait == 0 && islebusy) {
					printf("in ject conflict\n");
					exit(1);
				}
				rcv_wait = 0;
			} else {
				if(sync_indice < 0) {//����
					islebusy = 1;
					syncurIndice = sync_indice;
					CHKSIG_ISLE(1);
					UNLOCK_MUT_(mutisle);
				} else if(i_socket >= 0) {//�Է°� Ÿ�� �����н����̰� ���δٸ��ҽ�(isle)2������ �޴� ���
					if(cntoper) {//�ι�° �÷���(����) ����
						if(sockIndice[i_socket]) {//������ ���� ���Ե����� ��ø�Ǿ� �� ���Ե�����
							if(syncurIndice < sync_indice) {//�̹��� ������ ������ �� �ֱ��̸�
								if(alignoper) {//ieij.���� ���� �շ� �����̸� ��� ������ ���ŵǾ� ��)�̳� ��)���� 
									rcv_wait = 1;//�ñ۳� �Ҷ����� ����ϰ� ��)�̸� �������� �������� ���̰� 
									UNLOCK_MUT_(mutisle); //��)�̸� �̹����� ����Ȱ�
									continue;//��.�ٸ� ���� ������ ���ŵɶ����� ����Ͽ� ���� �Է´ܱ��� ���ǰ� �Ѵ�.(�߰� isle���� ��� ������̶��)
								} else {//���� ���� �շ� �����̸� ������ ������ ������� ����
									islein[i_socket]->feedf(in);//�������� ������ �̹������� ��ü
									syncurIndice = sync_indice;
								}
							}//else �Ϲ������δ� ���� ���Ͽ� �ڿ� �����ϴ� �÷����̹Ƿ� �̹��� ������ ������ ����  
							//���ŵ� ���Ϻ��� �� ������ ���̽��� ������(�ִ��ص� �������� ���� ����) ��)�� ��� �ü��ְ� �������� ��������
							UNLOCK_MUT_(mutisle);
						} else {//�Է�/Ÿ�� ���۷����� ���� ������ ������ ����� ����
							if(alignoper == 0) alignoper = 1;//ó������ ������ �շ����� ǥ��, ���������� ���
							//������ �߰� isle�� �н����̾ ������� ���ɵɼ����� �����ϼ������Ƿ�(�׷�ġ�ʰ� 
							//���� �����ʺ��� ������ ������ �ʰ� ������ ��쿡�� ó�� ����� ������ �ߺ��ؼ� ������
							//��� �������� ��) ������ �շ� ���ĺ��ʹ� ��� ������� �����ؾ� �ϹǷ� ��� ������ 
							//������ ���� ������ ���� ��� ����Ͽ� �����Ѵ�.(����ϸ� �Է´ܱ��� �� �ɸ����ְ�(
							//���� ���� �н��� ���isle�� ������̶��) �̷����Ͽ� �÷ο� ��Ʈ�� �ȴ�.
							if(syncurIndice == sync_indice) {//�������� ���ེ��. �ΰ� ������ �ι�°�� ���ŵ�
								islein[i_socket]->feedf(in);//������ ���������� ���Ͽ� ������ �ε��ϰ� 
								islebusy = 1;				//�н� �����Ѵ�.
								cntoper = 0;//�����Ͽ� ���� �Է�/Ÿ�� �� ���� �غ�
								sockIndice[0] = sockIndice[1] = 0;
								CHKSIG_ISLE(1);//��.
							} else {//������ �ٸ� ������, ���Ź��� �ΰ� ������ ��ũ�������� Ʋ����
								if(syncurIndice < sync_indice) {//��.�̹��� ������ ������ �� �ֱ��̸�
									islein[i_socket]->feedf(in);//�������� ������ �̹������� ��ü
									syncurIndice = sync_indice;//�ٸ� ���� ������ ������̸�
									sockIndice[i_socket] = 1;//���� ������ �̹� �������� �ٲٰ�
									sockIndice[!i_socket] = 0;//������ ��������� �ٲ�� �������
									CHKSIG_ISLE(0);//��.�������� ������ Ż���Ͽ� ������ �����ϵ��� �ñ׳�
								}// else cntoper = 0;//�̹��� ������ ������ ���� ���ŵ� ���Ϻ��� �� ������ 
							}						//���̸� �̹����� ������ ���� ��ġ������� ����
							UNLOCK_MUT_(mutisle);
						}
					} else {//���� �н��� ������� �ʱ���·μ� �з�/Ÿ�� �ΰ��� ��� �ϳ� ó������ receive�̸�  
						islein[i_socket]->feedf(in);//�÷��� ������ ��� ���� ��ġ�� �����ϵ��� ����
						syncurIndice = sync_indice;
						cntoper = 1;
						sockIndice[i_socket] = 1;//0�� �Է°�, 1�� Ÿ�ٰ� ��� ���� �̹��� ���ŵƴ��� ǥ��
						UNLOCK_MUT_(mutisle);
					}
				} else {//ijed.�������� ���ེ��. i_socket�� �����̸� ���۷��� �����μ� -2�� 2�� ��� ���� �ҽ�(isle)
					//���� ���ԵǴ� ���̹Ƿ� ���� ���о��� ����(�ڱⱸ�� ���ڴ� �н�, �Է°� Ÿ�� ������ �ڱⱸ���н�
					//���� �ٷ� ���԰��� �����ϴ� �н�), ������̸� ���� �����϶��μ� Ÿ�ٰ��� �ʿ����� �ʾ� ������
					//���ϰ��̾��� ���� ������ �ٷ� ����
					islein[0]->feedf(in);
					if(i_socket == -2) islein[1]->feedf(tar);
					syncurIndice = sync_indice;//Ŀ�� isle �Ѱ����� ���ŵǹǷ� ������ 1���� ���ǹǷ�
					islebusy = 1;				//sockIndiceǥ�� �ʿ����.
					CHKSIG_ISLE(1);
					UNLOCK_MUT_(mutisle);
				}
				return;//��)�� ��츦 ���� ������ ��� �����Ͽ� ������ ����.
			}
		}
	}
	bool peekevent(void)
	{
		LOCK_MUT_(mutisle);
		islebusy = 0;
		if(cntwait) {
			//printf("222- sig: %p %d\n", this, cntwait);
			SIG_USEM_(semijec, cntwait);
			cntwait = 0;
		}
		UNLOCK_MUT_(mutisle);
		//printf("222-1: %p\n", this);
		WAIT_USEM_(sempeek);
		//printf("222-2: %p\n", this);
		if(syncurIndice < 0) return 1;//�Ʒ� ó�� �ѹ��� async �޼��� ��� �ǰ� �̴� �����̴�
		if(rcvsynIndice + 1 != syncurIndice) printf("async %d\n", (intt)(syncurIndice - rcvsynIndice));
		rcvsynIndice = syncurIndice;

		return 0;
	}
	void runisle(void)
	{
		while(1) {
			if(peekevent()) break;
			if(isleimp->trainstep > 0 || (isleimp->trainstep < 0 && fintrain == 0)) {//iril.1�ܰ� 
				trainloop(rcvsynIndice);//�н��̰ų� 2�ܰ� �н��̸� ���ڴ��� �н��Ǵµ� 2�ܰ� �н����� 
				isleimp->sigtraining();//���ڴ�(���� 2�ܰ� �н� �Ϸ���� ���� ���ڴ�)�� �н� ����.
			} else predloop(rcvsynIndice);
		}
	}
};
class Encodeisle;
extern void convert_2d_to_1d(Flux *sor, intt dims[]);
//�׸��� ������ ������ �� �׸��带 ���δ� 8�� �����ڵ� �׸��带 �Է� �������η� �ϰ� ������ �Ѱ� ���� �׸��带
//Ÿ�� �������� �Ͽ� �����н��Ѵ�
class Deocdeisle : public Isleloop {
public:
	bool finalDenseDec;//���� ���ڴ��� ���ڴ�
	Flux *originDout;//���� ���ڴ��� ���ڴ�
	Encodeisle *encisle;
	void buildislet(Flux *in, Flux *tar)
	{
		intt dims[MX_DIM];

		if(isletrc->printBuilding) {
			printf("decord isle in: ");
			in->shape();
			printf("decord isle tar: ");
			tar->shape();
		}
		finalDenseDec = 0;
		originDout = nullx;

		convert_2d_to_1d(in, dims);
		islein[0] = flux(isletrc, 3, dims, in->qType, variable);

		convert_2d_to_1d(tar, dims);
		islein[1] = flux(isletrc, 3, dims, tar->qType, variable);

		islenet = new(isletrc)Generic;
		Generic *conet = (Generic *)islenet;

		conet->makeAnet(isletrc, szlatent, actcode);
		conet->setFrontendFactor2(islein[0]);
		//encmhead();
		conet->decompose(islein[1]->fshape[1]);//������ ���̰� ũ�Ƿ� decompose2�� �Ҽ� ����. ���ټ��� ����ɼ�����
		conet->connect(islein[1], headisle == HOD ? isoutdisc : isindisc, 0, rLeaning, isletrc->optType);//�����δ� ���ڴ��� Ÿ�� ���ڵ������� �����ǹǷ� outdiscrt�� �ǹ������Ƿ� ���� �ʿ������ �׳� üũ
		isleout = conet->cypred;
	}
	void encmhead(void);
	Deocdeisle(Impulse *imp, Dynagen *dgen, Encodeisle *encnet, Flux *in_gate, Flux *tar_gate, sytet head, bytet *name) : Isleloop(imp, dgen, head, name)
	{
		isleType = DECODE_ISLE;
		if(isletrc->frozenwgt < 2) fintrain = 0;//frozenwgt�� 2�̻��̸� ���ڴ� ����ġ �����ϵ��� �н��� �����ʰ��Ѵ�.
		imp->ndecisle++;
		encisle = encnet;
		//���ڴ��� �ܺο��� �ٷ� ������� �����Ƿ� extingate ���� �ʿ����.
		buildislet(in_gate, tar_gate);
		noperand = 2;
	}
	void adaptOrigin(Flux *src)
	{
		//anet�� �������� 2d�� ���� 1�������θ� �����ϹǷ� �Է��� 2d�ϰ��
		//�Է°� ����� ���� ����ϱ����� �Է� ����� �÷����� �غ��ϰ� ����� ���� �����Ѵ�.
		if(src->fdim > 3) originDout = flux(isletrc, src, variable);
		else originDout = nullx;
	}
	Flux *restoreOrigin(Flux *dout)
	{
		if(originDout) {
			originDout->copyf(dout);
			return originDout;
		} else return dout;
	}
	void trainloop(longx sync_indice)
	{
		if(fintrain) return;
		//if(isleimp->trainstep < 0) printisle(islein[0], "decoder train in");
		Flux *r = trainislet(islein[0], islein[1]);//�н��Ⱓ�߿� �н������� ������ �ʿ������ �����Ƿ� �н����Ѵ�.
		lossout(r, "_train");//���ڴ��� �н��� ���� üũ�� ���ڴ������ϰ� �н��� ������ eedp)���� ���̻� 
					//���ڴ� �н����� �����������Ƿ� ���⼭ �н� ���� üũ�� �ν��ƿ� ���θ� üũ�� �ʿ����.
		finaltrain(r);
		if(isleimp->trainstep < 0 && isletrc->concurtrain) predloop(sync_indice);//2�ܰ� �н�
		//�����̰� �����н��̸� �� �ܰ� �н��� �������� �ܰ� ���ڴ� �н����� �� �ܰ� �� �� ������
	}
	//void predloop(longx sync_indice) //����� �뵵
	//{
	//	Isleloop::predloop(sync_indice);
	//}
};

class QSortAllocator : public SortAllocator {
public:
	Trace *tracealt;

	QSortAllocator(Trace *trc)
	{
		tracealt = trc;
	}
	void *sortalloc(size_t size)
	{
		return tracealt->bxalloc(size);
	}
};
class Deconflict : public Typer {
public:
	intt xDiff, yDiff, xGrid, yGrid, xnGrid, ynGrid, xStride, yStride, xnStride, ynStride;
	intt xszZcode, ynZcode, bszGrid, szGrid, nScoop;//��ġ�� Ŀ�� ��Ʈ���̵�� ���� �Ļ��Ǵ� ������ ����
	intt szZcode, szFeat, xszReduce, szReduce;
	bool deconf2d;
	intt *mGrid;//reduced map
	virtual void checkinDeconflict(intt iorderive) = 0;
	virtual void resizem(void *zcodem, void *reductm, void *indicem) = 0;
	virtual void resizcode(void *zcodem) = 0;
};

template<typename DT>
class Deconflict2d : public Deconflict {
public:
	DT *mZcode;
	DT *mReduout;//indice map�� xy code size�� ������ ��
	Deconflict2d(void *zcodem, void *reductm, void *indicem, intt ydiff, intt xdiff, intt gridy, intt gridx,
		intt n_grid_y, intt n_grid_x, intt stridey, intt stridex, intt n_pred_scoop_y, 
		intt n_pred_scoop_x, intt ylen_zcode, intt xlen_zcode, intt feat_sz)
	{
		mZcode = (DT *)zcodem; mReduout = (DT *)reductm; mGrid = (intt *)indicem;
		szFeat = feat_sz;
		yDiff = ydiff; xDiff = xdiff;
		yGrid = gridy; xGrid = gridx;
		ynGrid = n_grid_y; xnGrid = n_grid_x;
		yStride = stridey; xStride = stridex;
		ynStride = n_pred_scoop_y;
		xnStride = n_pred_scoop_x;
		ynZcode = ylen_zcode; 
		xszZcode = xlen_zcode * szFeat;
		nScoop = ynStride * xnStride;
		szGrid = ynGrid * xnGrid * 2;//y, x��ǥ
		szZcode = ynZcode * xszZcode;
		xszReduce = xnGrid * xszZcode;
		szReduce = ynGrid * ynZcode * xszReduce;//�׸��� �� ĭ�� �ڵ尡 ����Ǵ� ������
		deconf2d = 1;
	}
	void resizem(void *zcodem, void *reductm, void *indicem)
	{
		mZcode = (DT *)zcodem; mReduout = (DT *)reductm; mGrid = (intt *)indicem;
	}
	void resizcode(void *zcodem)
	{
		mZcode = (DT *)zcodem;
	}
#define PGRID2d(deconf, pgrid, gy, gx) (pgrid + (gy * deconf->xnGrid + gx) * 2) //y, x��ǥ �ΰ���
#define PZCODE(deconf, mzcode, ips) (mzcode + ips * deconf->szZcode)
#define PREDUCE2d(deconf, predu, gy, gx) (predu + gy * deconf->ynZcode * deconf->xszReduce + (gx * deconf->xszZcode))
	void checkinDeconflict(intt ipred_scoop)
	{
		intt ibatch = ipred_scoop / nScoop;
		intt ipred_rest = ipred_scoop - (ibatch * nScoop);
		intt y = (ipred_rest / xnStride) * yStride, x = (ipred_rest % xnStride) * xStride;//�Է� �ۻ��� scoop�������� ������ ��ǥx,y
		intt gy = y / yGrid, gx = x / xGrid;//grid matrix x,y��ǥ
		intt *pgrid = mGrid + ibatch * szGrid;//�� ��ġ�� �׸��� ���� ������
		DT *pr = mReduout + ibatch * szReduce;//�� ��ġ�� ��� ��� ��Ʈ�m�� ���� ������

		if(*PGRID2d(this, pgrid, gy, gx) >= 0) return;//�̹� �׸��尡 ���� �׸��� ������ �ִ� �͵��� �̹� ����  
										//��Ȯ���� ���� ��Ʈ���̵� ������ �����ڵ尡 ��������� �̹����� ������.
		intt *gp;
		if(gy > 0) {//�� ��Ʈ���̵� ������ ������ �ٷ��� �� ������ְ� ������ �������� �����Ǹ� �̹��͵� ������.
			gp = PGRID2d(this, pgrid, (gy - 1), gx);
			if(*gp >= 0 && y - *gp < yDiff) return;
		}
		if(gy < ynGrid - 1) {//�� ���� ����üũ
			gp = PGRID2d(this, pgrid, (gy + 1), gx);
			if(*gp >= 0 && *gp - y < yDiff) return;
		}
		if(gx > 0) {//�� ���� ����üũ
			gp = PGRID2d(this, pgrid, gy, (gx - 1)) +1;//x��ǥ�� �ι�°�̹Ƿ� +1
			if(*gp >= 0 && x - *gp < xDiff) return;
		}
		if(gx < xnGrid - 1) {//�� ���� ����üũ
			gp = PGRID2d(this, pgrid, gy, (gx + 1)) +1;//x��ǥ�� �ι�°�̹Ƿ� +1
			if(*gp >= 0 && *gp - x < xDiff) return;
		}
		gp = PGRID2d(this, pgrid, gy, gx);//�� �׸��� ��ǥ�� �׸��� �ʿ� ����
		*gp = y; *(gp + 1) = x;
		//��� ��� ��Ʈ������ �̹� �׸��� ���� �ڵ� ����
		DT *rp = PREDUCE2d(this, pr, gy, gx), *zp = PZCODE(this, mZcode, ipred_scoop);
		for(intt i = 0;i < ynZcode; i++) {
			for(intt j = 0;j < xszZcode; j++) *(rp + j) = *zp++;
			rp += xszReduce;
		}
	}
};
template<typename DT>
class Deconflict1d : public Deconflict {
public:
	DT *mZcode;
	DT *mReduout;
	Deconflict1d(void *zcodem, void *reductm, void *indicem, intt xdiff, intt gridx, intt n_grid_x,
		intt stridex, intt n_pred_scoop_x, intt xlen_zcode, intt feat_sz)
	{
		mZcode = (DT *)zcodem; mReduout = (DT *)reductm; mGrid = (intt *)indicem;
		szFeat = feat_sz;
		xDiff = xdiff; xGrid = gridx; szGrid = xnGrid = n_grid_x;
		xStride = stridex; xnStride = nScoop = n_pred_scoop_x;
		szZcode = xszZcode = xlen_zcode * szFeat;
		szReduce = xnGrid * xszZcode;
		deconf2d = 0;
	}
	void resizem(void *zcodem, void *reductm, void *indicem)
	{
		mZcode = (DT *)zcodem; mReduout = (DT *)reductm; mGrid = (intt *)indicem;
	}
	void resizcode(void *zcodem) 
	{
		mZcode = (DT *)zcodem;
	}
#define PGRID1d(deconf, pgrid, gx) (pgrid + gx)
#define PREDUCE1d(deconf, predu, gx) (predu + gx * deconf->xszZcode)
	void checkinDeconflict(intt ipred_scoop)
	{
		intt ibatch = ipred_scoop / xnStride;
		intt ipred_rest = ipred_scoop - (ibatch * xnStride);
		intt x = ipred_rest * xStride;//�Է���ǥx
		intt gx = x / xGrid;//grid matrix x��ǥ
		intt *pgrid = mGrid + ibatch * xnGrid;//�� ��ġ�� �׸��� ���� ������
		DT *pr = mReduout + ibatch * szReduce;//�� ��ġ�� ��� ��� ��Ʈ�m�� ���� ������

		if(*PGRID1d(this, pgrid, gx) >= 0) return;//�̹� �׸��尡 ���� �׸��� ������ �ִ� �͵��� �̹� ���� 
										//��Ȯ���� ����  ��Ʈ���̵� ������ �����ڵ尡 ��������� �̹����� ������.
		intt *gp;
		if(gx > 0) {//�� ���� ����üũ
			gp = PGRID1d(this, pgrid, gx - 1);
			if(*gp >= 0 && x - *gp < xDiff) return;
		}
		if(gx < xnGrid - 1) {//�� ���� ����üũ
			gp = PGRID1d(this, pgrid, gx + 1);
			if(*gp >= 0 && *gp - x < xDiff) return;
		}
		*PGRID1d(this, pgrid, gx) = x;//�� �׸��� ��ǥ�� �׸��� �ʿ� ����
		//��� ��� ��Ʈ������ �̹� �׸��� ���� �ڵ� ����
		DT *rp = PREDUCE1d(this, pr, gx), *zp = PZCODE(this, mZcode, ipred_scoop);
		for(intt j = 0;j < xszZcode; j++) *(rp + j) = *zp++;
	}
};
template<typename DT> class TeleReduction : public TeleSort {
public:
	Deconflict *inChecker;
	DT *ordermap;
	intx nSortRange, ibatchSort;

	TeleReduction(ParallelSort *ps, BaseSort *bs, Deconflict *deconf, intt nsort) : TeleSort(ps, bs)
	{
		inChecker = deconf;
		nSortRange = nsort;
	}
	void settings(intx s, intx e, void *dc, void *sc = nullx)
	{
		ibatchSort = s; 
		ordermap = (DT *)dc;
	}
	void sorting(void)
	{
		DT *porder = ordermap + ibatchSort * nSortRange * 2 + 1;//2(�սǿ����� ���� �ڽ� ���� �ε���) + 1(�ε��� ���� ����)
		for(intt i = 0;i < nSortRange; i++) {
			inChecker->checkinDeconflict((intt)*(porder + i * 2));
		}
	}
};
template<typename DT> class BaseReduction : public BaseSort {
public:
	Deconflict *inChecker;
	intx nSortRange;

	BaseReduction(ParallelSort *psort, SortAllocator *alc, Deconflict *deconf, intt nsort) : BaseSort(psort, alc)
	{
		inChecker = deconf;
		nSortRange = nsort;
	}
	TeleSort *newTeles(void)
	{
		TeleSort *ts = new(allocators, sortProsBs)TeleReduction<DT>(sortProsBs, this, inChecker, nSortRange);
		return ts;
	}
};
extern intt scoopeout_size(bool scoop_inner, intt seqy, intt seqx, intt slidey, intt slidex, intt stridey, intt stridex, intt &outy, intt &outx);
class Encodeisle : public Isleloop {
public:
	bool findectrain;
	intt zcompsize, cntDecTStep, szsortm, szindice, nzcDerive;
	void *outindice, *sortmap;
	Generic *conet;
	Deocdeisle *decisle;//predict�϶��� ���� islepair�� decodisle�� eject�Ѵ�.
	Deconflict *deconflict;
	QSortAllocator *qsrtalc;
	BaseSort *qsrtbase;
	BaseSort *redubase;
	void buildislet(Flux *in, intt &bound_sz)
	{
		isletrc->batchloss = 1;
		Flux *isle_in = islein[0] = in->duplicate(isletrc);//�� isle�� ���� �ڽ��� tracer�� ������ 
		//���������İ� ���������� ����ǹǷ� ����� �÷������� �����ؾ� �Ѵ�. �ȱ׷��� Ʈ���̽����� �׶����� ����ȴ�.
		bool d2 = isle_in->fdim > 3 ? 1 : 0;//�����̵� Ŀ���� �������� �����Ѵ�.
		intt seqy = d2 ? isle_in->fshape[1] : 0, seqx = d2 ? isle_in->fshape[2] : isle_in->fshape[1];
		intt sz_slide = (d2 ? std::sqrt(isletrc->szslide2d) : isletrc->szslide1d);//szslide2d�� 1���� �����̹Ƿ� 
		intt sz_outrow, sz_outcol, out_grid = sz_slide;//��°���
		intt diff = sz_slide - isletrc->overlapSlide;//���ı� ��� ����
		intt out_shape[MX_DIM];//, ind_shape[MX_DIM];
		//�����̵� ������, �н� ��Ʈ���̵�� ������ �߷�
		scoopout = isle_in->scoopup(sz_slide, sz_slide, isletrc->slTrainInterv, isletrc->slTrainInterv);//����
		intt discrete = (headisle == HID ? isindisc : isoutdisc);//�� isle�� �Է�/��� ���ڵ��� ���� isle�϶� ����
		islenet = generic(isletrc, scoopout, scoopout, szlatent, discrete, discrete,
					szembede, actcode, rLeaning);//Ŀ�� �ڱⱸ���н� ��
		conet = (Generic *)islenet;
		intt zcode_len = conet->zcodec->fshape[1];//����붧 ������ üũ
		intt feat_sz = conet->zcodec->fshape[2];
		intt ylen_grid, xlen_grid, ind_sz;
		//������ ��� �������� ��Ʈ���̵� ���� ��� ��ҵǴ� �׸��� ��Ʈ������ y,x ������ ���
		scoopeout_size(1, seqy, seqx, sz_slide, sz_slide, out_grid, out_grid, ylen_grid, xlen_grid);
		if(d2) {
			zcode_len = std::sqrt(zcode_len) < 1 ? 1 : std::sqrt(zcode_len);//������ ���������� �Ѵ�.
			//ind_shape[1] = ylen_grid;//reduce y��Ҹ�Ʈ���� ä������� ����üũ�� �ʿ��� �ε��� �� ������ ����.
			//ind_shape[2] = xlen_grid;//reduce x
			//ind_shape[3] = 2;//y,x �ε��� �ΰ�
			szindice = ylen_grid * xlen_grid * 2 * sizeof(intt);//x,y ��ǥ �ΰ�
			//Ŀ�� ���� �ڵ带 ���������� ���� ���̹Ƿ� x�� y�� ���� �ڵ� ���̸� ���Ѵ�. ��)Ŀ�� ������ 8�� �Ͽ�
			//scoop�� �������� 16�̸� 16 / 8 == 2�� �ڵ忡 �ڵ�� ���� 2, 2 * 2 == 4
			//scoop�� �������� 64�̸� 64 / 8 == 8�� �ڵ忡 �ڵ�� ���� 2, 8 * 2 == 16 -> y,x�� ���� 4�� ��
			out_shape[1] = ylen_grid * zcode_len;//y,���������� �ϱ⶧���� y�൵ ��½����� �ڵ� ����� ���Ѵ�
			out_shape[2] = xlen_grid * zcode_len;//x Ŀ�δ� ��½�����(1����) �ڵ� ����� ���Ѵ�.
			out_shape[3] = feat_sz;//���� ����
			if(out_shape[1] * out_shape[2] < bound_sz) bound_sz = -1;//��� ����� ���� ���ϸ� �������� fully ����
		} else {
			//ind_shape[1] = xlen_grid;//reduce x �ε���
			//ind_shape[2] = 1;//x �ε��� �Ѱ�
			szindice = xlen_grid * sizeof(intt);//x��ǥ �Ѱ�
			out_shape[1] = xlen_grid * zcode_len;//Ŀ�δ� ��½�����(1����) ����� ���Ѵ�. ��±��̰� 2��
			//�̻��̶� �������� 1���϶��� �����ϰ� ������ 1�� �Է°� ���������� ���������� ó���Ѵ�. ��� ���̰� 
			//2���̻� ��µɶ� �� ��� ���� ���� ����ġ�� �������� ���̹Ƿ� 2�� ���� �Ϲ����� �ڸ��� �����ϵ� 
			//�ΰ��� �Բ���޵ɶ��� �ǹ� �ִ� ���� �ƴϹǷ�
			out_shape[2] = feat_sz;//���� ����
			if(out_shape[1] < bound_sz) bound_sz = -1;
		}
		out_shape[0] = isle_in->fshape[0];//��ġ ���� ����.
		isleout = flux(isletrc, isle_in->fdim, out_shape, isle_in->qType, variable);//��� �ڵ�� ��ҵ� ��
		//outindice = flux(isletrc, isle_in->fdim, ind_shape, tint, variable);//�ε��� ��
		intt n_pred_scoop_y, n_pred_scoop_x;//������ �������� ��Ʈ���̵� ���� ���� y,x ������ ��� ����.
		nzcDerive = scoopeout_size(0, seqy, seqx, sz_slide, sz_slide, isletrc->slPredInterv, isletrc->slPredInterv, n_pred_scoop_y, n_pred_scoop_x);
		qsrtalc =  new QSortAllocator(isletrc);//���� �Ҵ��� ����.
		switch(isle_in->qType) {
			case NONET_TP:
				break;
			case BYTET_TP:
				break;
			case tshort:
				break;
			case tfloat:
				szsortm = sizeof(floatt) * 2;//������ ���� �ڵ� ���� �ε��� pair ������
				outindice = malloc(szindice * out_shape[0]);//�ε��� �׸��� �� �Ҵ�.
				sortmap = malloc(szsortm * nzcDerive);//������ ���� �ڵ� ���� �ε��� ���� ����
				if(d2) deconflict = new(isleimp->impTcr)Deconflict2d<floatt>(conet->zcodec->begin_p(),
					isleout->begin_p(), outindice, diff, diff, out_grid, out_grid,
					ylen_grid, xlen_grid, isletrc->slPredInterv, isletrc->slPredInterv,
					n_pred_scoop_y, n_pred_scoop_x, zcode_len, zcode_len, feat_sz);
				else deconflict = new(isleimp->impTcr)Deconflict1d<floatt>(conet->zcodec->begin_p(),
					isleout->begin_p(), outindice, diff, out_grid, xlen_grid,
					isletrc->slPredInterv, n_pred_scoop_x, zcode_len, feat_sz);
				qsrtbase = new BaseQSort<floatt>(rsc::cydrome->parallels, qsrtalc);
				redubase = new BaseReduction<floatt>(rsc::cydrome->parallels, qsrtalc, deconflict, nzcDerive);
				break;
			case tint:
				break;
			case tlong:
				break;
			case tdouble:
				break;
		}
	}
	Encodeisle(Impulse *imp, Dynagen *dgen, Flux *in_gate, intt &bound_sz, bool pair, sytet head, bytet *name, floatt lr = -1) : Isleloop(imp, dgen, head, name)
	{
		if(isletrc->printBuilding) {
			printf("encord isle: ");
			in_gate->shape();
		}
		isleType = ENCODE_ISLE;
		extingate = in_gate;//�ܺ� ����Ʈ ����
		cntDecTStep = 0;
		noperand = 1;
		if(lr > 0) rLeaning = lr;
		buildislet(in_gate, bound_sz);
		if(isletrc->printBuilding) {
			printf("encord isle out: ");
			isleout->shape();
		}
		if(pair) {
			strcat(name, "_decoder");
			decisle = new(imp->impTcr)Deocdeisle(imp, dgen, this, isleout, in_gate, head, name);
			//regiRouteTo(decisle, -2);eepl)���� ���� �����ϹǷ� ����� ��� �ʿ����.
			findectrain = 0;
		} else {
			decisle = nullx;
			findectrain = 1;
		}
	}
	void rmloop()
	{
		free(outindice);
		free(sortmap);
		delete qsrtalc;
		Isleloop::rmloop();
	}
	void trainloop(longx sync_indice)
	{
		if(fintrain) {
			predloop(sync_indice);
			return;
		}
		scoopout->scoopup(isletrc->slTrainInterv, isletrc->slTrainInterv);
		Flux *r = trainislet(scoopout, scoopout);//scoopout�� �����Ҷ� �Է����� �����������Ƿ� �Է��� �ǹ̾��� �׳��Ѵ�.
		finaltrain(r);
		lossout(r, "_train");
		if(isletrc->concurtrain) predloop(sync_indice);//�����н��̸� �� �ܰ� �н��� ���� �ܰ� �н����� �� �ܰ� �� �� ������
	}
	void predloop(longx sync_indice)
	{
		Flux *in = islein[0];
		scoopout->scoopup(isletrc->slPredInterv, isletrc->slPredInterv);//scoopout�� scoop�� ��� �����
		predislet(scoopout, &predloss);//scoopout�� �����Ҷ� �Է����� �����������Ƿ� �Է��� �ǹ̾��� �׳��Ѵ�.
		//1.inner scoop - ��Ʈ���̵�� ������ ��� ������ ������ �����Ѵ�. ��Ʈ���̵�� �� �������� ���ĸ��� �����ϴ� ���̰�
		//�������� �̷��� ������ ���ĸʵ�� ��� �ʸ� �����Ҷ� Ŀ�λ������ ��� ���� ���ĵ鰣�� ��ġ�� ����� ���̳� �̴�
		//���� ��� ���� ������� Ŀ�λ������ ������ ����� �� ������� �Է� ���� ���� ��[(n + d -1) / d]�� �ȴ�.
		//���� Ŀ�λ����� 4*4, �Է� ������ 32*32��� ������ ������ 32 + (4-1) / 4 = 8, 8*8�� �ǰ� 1ĭ ������ ����ϸ�
		//32 + (3-1) / 3 = 11, 11*11�� �ȴ�. �� ��� �ʿ� ��Ʈ���̵��Ͽ� ������ 4*4 ���ĸ��� �� ���� �𼭸��� ����������
		//�Ͽ� �Է¸ʿ����� ��ǥ�� ���ڴ� 4�� ������, ����(������1)�� 3���� ���� ���� ��ǥ�� �׷���(�������� ���� �͵�)�Ͽ� 
		//�׷�� ���ĵ� ���� ���� ���� ������ ��¸ʿ� �ε����ϰ� ���� ������ �ļ����� �������� �̹� ��ǥ�������־� �ߺ��� ����
		//��¸ʻ󿡼� �̹��� �ε����� ��ǥ�� �����¿� �װ��� ���� �ε��̵� �͵��� ������ ���� �Է¸ɻ��� ��ǥ�� ���Ͽ� 
		//��� �������� �Ѿ�� �̹� ���ĸ��� ������. ��Ʈ���̵尡 ��ĭ�� �ƴ� ��ĭ, ��ĭ �����̷��� �� �������� ���ĸ��� 
		//�����Ͽ� �� ����� �����ϰ� �����Ѵ�. ��Ʈ���̵� ��ī�̻��̸� �� ��¸� �������� ����� ����� �̴� ��¿������. 
		//Ŀ�λ���� 32���ϸ� 4�� �������� �򸦼��ְ� ORTHO_DOT�� ���������� 4�� �������� �� ����� ���������Ƿ� �߰��������ʹ� 
		//��Ʈ���̵�2ĭ�� ������ ���� 2ĭ���� �Ͽ� 4�� �������� �ѵ���� ����Ҽ��ִ�. ORTHO_DOT�� ortho_single��
		//�����Ѵٸ� 4�������� ������ ������ �ֱ��Ͽ� ��Ʈ���̵峪 �������Ҷ� 1ĭ�� �ص� �ȴ�.(Ȯ������ ���� �ƴ�)
		//�����Ҷ��� 1ĭ �������� ��Ʈ���̵��ϰ� �н� �Ҷ��� ���� �������� ��Ʈ���̵� �Ѵ�.
		//�з�,�˻����� �ۿ����� ���� ���� �̻� ���ĵ��� �����ϰ�(������� �ƽ�Ǯ���� ���� �ǹ�) ORTHO_DOT�� ortho_single��
		//�����Ͽ� �������� �������μ� ���͸� �� ���� ������ �����Ѵ�. ������ ������ ������ �ִ� ������ �̹����� �н��� ����
		//���� ������ �̹����� ū������ �̹������� ���� ���� �������� �񱳸� �����Ͽ� ������ �����ϴ� �� ����, 
		//�������ڵ� ����̹Ƿ� ������ �����Ͽ� ������ ������ ã���� ���� ��

		//2.outer scoop - ������ Ŀ�� ������ �������� ������(��Ʈ���̵�)�� ���� �Ļ��Ǵ� ���ĵ��� ��ü ������ �����
		//�Ѵ� ���� �����е��ǹǷ� �׻� �� ���� ���İ� ���� ��Ȯ���� ���� �ǹǷ� �� ����� ���ϴ� ���� 
		//[((n - k) + d -1) / d] + 1 �� ���� �ణ �����Ѵ�. k�� Ŀ�� ������
		//��) n: 8, k: 4, (1) d: 4 -> [(4 + 3) / 4] +1 => 2, (2) d: 3 -> [(4 + 2) / 3] +1 => 3, 
		//(3) d: 2 -> [(4 + 1) / 2] +1 => 3, (4) d: 1 -> [(4 + 0) / 1] +1 => 5
		//(3)�� ���� ��� ���� ������ [0,2,4,6]������ 4�� 6�� ������ Ŀ�α��� [4 - 8]���� [4,5,6,7]�� [6,7,0,0]�� ����
		//�ΰ� ���İ� �Ļ��ǰ� �ι�° ���Ŀ��� ���� �����е� 2���� �߰��Ǿ� �����е��� �ѻ� �����Ƿ� �ظ��ϸ� �ι�° ���İ� 
		//��Ȯ���� ���� ���ð��̹Ƿ� �ְ� �ɼ������Ƿ� �ι�° ������ �Ѵ�.

		//3. 2.���� ���� �Ұ�� ��Ʈ���̵带 1�� �������̵带 1�� ������ �������̵� �׸���� [0,1,2,3] [3,4,5,6] [6,7,0,0]��
		//���̵ǰ� ��Ʈ���̵� �׸���� [0,1,2,3] ~ [4,5,6,7]�� �ǰ� �������̵� �׸��� [6,7,0,0]�� ���ϴ� �Ļ� ��������
		//��Ʈ���̵� �׸��忡�� ������� �ʴ´�. ���� 2.���� ���� [(n + d -1) / d]�� �Ѵ�. 2.�� �������� [6,7]�� �׻�
		//�����ͼ� �װ��� ��Ȯ���� ���ٸ� �̰��� ���õǴ� ���� �°� �� ������ [3,4,5,6]�� ���õɼ������Ƿ� ������.
		//�Ǿ��ʿ� �ƿ��� ������ ���ϴ� ���� �� �ڿ��� ���� ������ [3,4,5,6]�ε� �̰��� ���õȴ� �ص� [0,1,2,3]�� ���õǸ�
		//�ϳ��� ���ǵǴ� ���� �����Ƿ� �ս￡�� �ƿ��� ������ �� �ʿ���� [0,1,2,3] �ȿ��� [0,1]�� �� ��Ȯ�ϴ� [2,3]�� �� 
		//��Ȯ�ϴ� �ϴ� [0,1,2,3]�� �������� �ԷµǸ� ���� �������� �н��ǹǷ� �������.
		intt n_batch = in->fshape[0];
		if(isleout->fshape[0] != n_batch) {
			isleout->resizing2(in);
			free(outindice);
			free(sortmap);
			outindice = malloc(szindice * n_batch);//�ε��� �׸��� �� �� �Ҵ�.
			sortmap = malloc(szsortm * conet->zcodec->fshape[0]);//fshape[0]�� ��ġ�� �Ļ������� ������ ������ ������, ������ ���� �ڵ� ���� �ε��� ���� ���� �� �Ҵ�
			deconflict->resizem(conet->zcodec->begin_p(), isleout->begin_p(), outindice);
		} else deconflict->resizcode(conet->zcodec->begin_p());//gpu��������� ������ ȣ��Ʈ �޸𸮷� �����Ű���, 
																//�缳��, isleout�� ��������Ƿ� pass
		memset(outindice, -1, szindice * n_batch);
		isleout->resetData(-1);//ȣ��Ʈ �޸𸮸� ���� ����
		//conet->zcodec->printo(2, 10);
		//dbg_check_mark(conet->zcodec->begin_p(), predloss->begin_p(), n_batch, conet->zcodec->fshape[0], predloss->fshape[0]);//dbg_check2�� �Ҹ�� ����.
		TeleSort *sort_head = nullx, *reduct_head = nullx, *sts, *rts;
		void *loss_m = predloss->begin_p();
		intt i_beg = 0;
		//printf("aaa-1: %p\n", this);
		for(intt i = 0;i < n_batch; i++, i_beg += nzcDerive) {
			sts = qsrtbase->getTeles();
			APPEND_LIST2(sort_head, sts);
			sts->settings(i_beg, i_beg + nzcDerive, sortmap, loss_m);
			sts->rsort();//���� ����.
		}
		sts = sort_head;
		for(intt i = 0;i < n_batch; i++, sts = sts->ptrRight2) {
			sts->wsort();
			//dbg_print_sort(sortmap, loss_m, conet->zcodec->begin_p(), nzcDerive);
			//reductionOut(i);//����ó�� ���ҷ��� �Ʒ� ��ſ� Ÿ�Կ� ���� ȣ�� 
			rts = redubase->getTeles();
			APPEND_LIST2(reduct_head, rts);
			rts->settings(i, -1, sortmap, nullx);
			rts->rsort();//reduction ����
		}
		//printf("aaa-2: %p\n", this);
		for(rts = reduct_head;rts; rts = rts->ptrRight2) {
			rts->wsort();
		}
		//printf("aaa-3: %p\n", this);
		//printisle(isleout, "encode pred");
		//dbg_check2(conet->zcodec->begin_p(), isleout->begin_p(), sortmap, predloss->begin_p(), n_batch, 1);//dbg_check_mark�� �����ϸ� ���� 1��, �ƴϸ� 0�� ȣ��
		//doublet mincode, maxcode;
		//isleout->minmax(mincode, maxcode);
		//isleout->minmax_normal(mincode, maxcode);
		isleout->stdnormal();
		//printisle(isleout, "encode pred2");
		lossout(predloss, "_pred");
		//isleout eject�� inject���� �޸� ������ �����ϴµ��� ���ǹǷ� ȣ��Ʈ �޸� ���� �ʿ����.
		if(isleimp->trainstep < 1 || findectrain) {//���������̰ų� 2�ܰ� �н� Ȥ�� 1�ܰ� �н��ε�
			Isleloop::eject(isleout, sync_indice);//���ڴ� �н����� �������� �������� �н����� ������
			if(decisle) {
				if(isleimp->trainstep > 0) isleimp->sigtraining();//1�ܰ��н��ε� ���ڴ� �н��� ��������
				//���ڴ��� ������ϹǷ� �����н� 2�ܰ� �����̸� ���ڴ��� ����Ϸ� ī���ø� �Ѵ�.
				if(isleimp->predwithloss) decisle->islein[1]->feedf(in);//�򰡶� Ÿ���� �־��� �ν�����̸� ���ڴ� Ÿ�� ���� ����
			}
		} else {//eedp.���ڴ��� �н��� �������� ���ڴ� �н� ����.(���ڴ��� ������ ����� ���� �ʴ´�)
			decisle->inject(isleout, in, sync_indice, -2);//eepl.���ڴ��� 0�� ���Ͽ� preds�� �Է�������, in�� Ÿ�ٰ����� ���� ����
			Isleloop::eject(isleout, sync_indice);//������ �������Ͽ� ���� �н�, ���ڴ��� ���İ��谡 �����Ƿ� ���� �н���.
		}
		//printf("aaa-3: %p\n", this);
	}
	template<typename DT> void reductionOut(intt ibatch)
	{
		DT *porder = (DT *)sortmap + ibatch * nzcDerive * 2 + 1;//2(�սǿ����� ���� �ڽ� ���� �ε���) + 1(�ε��� ���� ����)
		for(intt i = 0;i < nzcDerive; i++) {
			deconflict->checkinDeconflict((intt)*(porder + i * 2));
		}
	}
	void dbg_print_sort(void *_smap, void *_lmap, void *_z_map, intt ns)
	{
		Deconflict *df = deconflict;
		floatt *smap = (floatt *)_smap, *lmap = (floatt *)_lmap, *z_map = (floatt *)_z_map;
		floatt v = -100;//��������
		//floatt v = 100;//��������
		intt x, y, ips;

		LOCK_MUT_(isleimp->mutimp);
		for(intt i = 0;i < ns; i++) {
			ips = (intt)*(smap + i * 2 + 1);
			if(*(smap + i * 2) < v) {//��������
			//if(*(smap + i * 2) > v) {//��������
				printf("inverse: %f %d\n", v, ips);
				break;
			}
			y = (ips / df->xnStride) * df->yStride, x = (ips % df->xnStride) * df->xStride;
			if(*(smap + i * 2) == v) printf("dup: %f %d (%d, %d)\n", v, ips, y, x);
			else printf("%f %d (%d, %d)\n", *(smap + i * 2), ips, y, x);
			v = *(smap + i * 2);
			if(v != *(lmap + ips)) {
				printf("error %f\n", *(lmap + ips));
				break;
			}
			for(intt j = 0;j < df->szZcode; j++) printf("%f ", *(z_map + ips * df->szZcode + j));
			printf("\n");
		}
		UNLOCK_MUT_(isleimp->mutimp);
	}
	void dbg_check_mark(void *z_map, void *b_loss, intt n_batch, intt zc_batch, intt l_batch)
	{
		Deconflict *df = deconflict;
		intt x, y;

		if(n_batch * nzcDerive != zc_batch) exit(1);
		if(zc_batch != l_batch) exit(1);
		if(nzcDerive != deconflict->nScoop) exit(1);

		floatt *zp = (floatt *)z_map, *lp = (floatt *)b_loss, *ep;
		intt sz_feat = deconflict->szFeat;
		//Ÿ���� floatt, �Ѱ� z_map�� 4�� ���̷� ����, ��µ� �����ڵ� �ʿ� �� �ڵ��� ������ ������ ����
		for(intt i = 0;i < n_batch; i++) {				//[index, 0
			for(intt j = 0;j < nzcDerive; j++, lp++) {	// 0, loss]�� ���� ���Ŀ� �����ϰ� ���������� ����.
				ep = zp + df->szZcode;
				if(df->deconf2d) {
					y = (j / df->xnStride) * df->yStride, x = (j % df->xnStride) * df->xStride;
					for(intt k = 0;k < sz_feat; k++) *zp++ = (floatt)y;
					for(intt k = 0;k < sz_feat; k++) *zp++ = (floatt)x;
					for(intt k = 0;k < sz_feat; k++) *zp++ = (floatt)j;
					for(intt k = 0;k < sz_feat; k++) *zp++ = *lp;
				} else {
					for(intt k = 0;k < sz_feat; k++) *zp++ = (floatt)j;
					if(df->xszZcode > 1) {
						for(intt k = 0;k < sz_feat; k++) *zp++ = *lp;
					}
				}
				for(;zp < ep; zp++) *zp = -1;
			}
		}
	}
	void dbg_check2(void *z_map, void *r_map, void *s_map, void *b_loss, intt n_batch, bool dgb_chk)
	{//�� �Լ��� ���������� dgb_chk�� 1�� �Ѵ�.
		floatt *sp = (floatt *)s_map, *lp = (floatt *)b_loss;
		floatt *rp = (floatt *)r_map, *rp3, *zp = (floatt *)z_map, *zp3;
		Deconflict *df = deconflict;

		intt sz_redu = df->szReduce, sz_feat = df->szFeat;
		intt xsz_redu = df->xszReduce, n_scoop = df->nScoop;
		intt x, y, gx, gy, *pg, *gp, ipred_scoop;

		for(intt i = 0;i < n_batch; i++, rp += sz_redu) {
			pg = df->mGrid + i * df->szGrid;//�� ��ġ�� �׸��� ���� ������
			//rp - �� ��ġ�� ��� ��� ��Ʈ�m�� ���� ������
			if(df->deconf2d) {
				for(intt j = 0;j < n_scoop; j++) {//�Ѱ� ��ġ�� �Ļ� �������� ������ ������ ���������� üũ
					y = (j / df->xnStride) * df->yStride;
					x = (j % df->xnStride) * df->xStride;//�Է� �ۻ��� scoop�������� ������ ��ǥx,y
					gy = y / df->yGrid, gx = x / df->xGrid;//grid matrix x,y��ǥ
					gp = PGRID2d(df, pg, gy, gx);//�� x,y��ǥ�� align�� �׸��� ��ǥ�� ����� scoop�������� ������
					if(*gp != y || *(gp + 1) != x) continue;//��ǥx,y�� �� x,y��ǥ�� ����������(���õ������� �Ļ� ��������) ��ŵ
					rp3 = PREDUCE2d(df, rp, gy, gx);//�� �Ļ������������ڵ� j�� �����Ҹʻ��� ���������� 			
					if(dgb_chk) {//dbg_check_mark���� �� �Ļ��������� sort�ʿ� ����� ������ ������ ��Ҹʿ� ���� ����ƴ��� Ȯ��
						rp3 += xsz_redu;//[2,2]���·� ����ư� ���� �ο쿡 ������ ����.
						for(intt b = 0;b < sz_feat; b++) {//���ļ���ŭ �ߺ� ���������.
							if(*(rp3 + b) != j) {
								exit(1);//����°�� ������ ����� ���� ��ġ�ϴ°� üũ
							}
							if(*(rp3 + sz_feat + b) != *(lp + i * n_scoop + j)) {
								exit(1);//������(�׹�°)�� �սǿ��� ����Ȱ� ��ġ���� üũ
							}
						}
					} else {//�Ļ������� �����ڵ� ���� �� �ڵ尡 ������ �ʿ� ����� ����Ƴ� �˻�
						ipred_scoop = i * n_scoop + ((y * df->xnStride) / df->yStride) + (x / df->xStride);
						if(ipred_scoop != i * n_scoop + j) {
							exit(1);//�Ļ������� ����
						}
						zp3 = PZCODE(df, zp, ipred_scoop);
						for(intt a = 0;a < df->ynZcode; a++) {//[2,2]���·� ���������Ƿ� 2���ο� ����
							for(intt b = 0;b < df->xszZcode; b++) {
								if(*(rp3 + b) != *zp3++) {
									exit(1);
								}
							}
							rp3 += xsz_redu;
						}
					}
				}
			} else {
				for(intt j = 0;j < n_scoop; j++) {//�Ѱ� ��ġ�� �Ļ� �������� ������ ������ ���������� üũ
					x = j * df->xStride;//�Է� �ۻ��� scoop�������� ������ ��ǥx
					gx = x / df->xGrid;//grid matrix x��ǥ
					gp = PGRID1d(df, pg, gx);//�� x��ǥ�� align�� �׸��� ��ǥ�� ����� scoop�������� ������
					if(*(gp + 1) != x) continue;//��ǥx�� �� x��ǥ�� ����������(���õ������� �Ļ� ��������) ��ŵ
					rp3 = PREDUCE1d(df, rp, gx);//�� �Ļ������������ڵ� j�� �����Ҹʻ��� ���������� 			
					if(dgb_chk) {//dbg_check_mark���� �� �Ļ��������� sort�ʿ� ����� ������ ������ ��Ҹʿ� ���� ����ƴ��� Ȯ��
						for(intt b = 0;b < sz_feat; b++) {
							if(*(rp3 + b) != j) exit(1);//ù��°�� ������ ����� ���� ��ġ�ϴ°� üũ
							if(df->xszZcode > 1) {
								if(*(rp3 + sz_feat + b) != *(lp + i * n_scoop + j)) {
									exit(1);//�ι�°�� �սǿ��� ����Ȱ� ��ġ���� üũ
								}
							}
						}
					} else {
						zp3 = PZCODE(df, zp, j);
						for(intt a = 0;a < df->xszZcode; a++) {
							if(*(rp3 + a) == *zp3++) exit(1);
						}
					}
				}
			}
		}
	}
};
class CoupleIsle : public Isleloop {
public:
	CoupleIsle(Impulse *imp, Dynagen *dgen, sytet head, bytet *name) : Isleloop(imp, dgen, head, name)
	{
		isleType = COUPLE_ISLE;
		if(isletrc->frozenwgt < 3) fintrain = 0;//frozenwgt�� 3�̻��̸� Ŀ��isle ����ġ �����ϵ��� �н��� �����ʰ��Ѵ�.
		nprotect = 2;//�������尡 �Է� �� Ÿ�� �ΰ��̹Ƿ� 
	}
	void trainloop(longx sync_indice)
	{
		if(fintrain) {//Ŀ��isle�� �н��� �������� ������ �� �͵��߿� ���ڴ��� �ƴ� ���� �ϳ��� ������ ������
			bool forward_train = 0;
			for(Socketisle *s = isleRoute;s; s = s->ptrRight) {
				if(s->destisle->isleType != DECODE_ISLE) forward_train = 1;
			}
			if(forward_train) predloop(sync_indice);//�� �Լ��� e ject���� ���ڴ����� ������� 2�ܰ� �н� Ȥ�� �򰡴ܰ迡���� �ȴ�.
		} else {//Ŀ��isle �н� ����, �� �Լ��ȿ��� �����н� ������ ��� predȣ��Ǹ� ������ ������ �ȴ�.
			//printf("=====================================\n");
			//islein[0]->printo();
			Isleloop::trainloop(sync_indice);//�� �Լ� e ject���� �н� 1�ܰ迡���� ���ڴ��� �ƴ� �͵鸸 ������ȴ�.
		}
	}
	/*void predloop(longx sync_indice)
	{
		printf("=====================================\n");
		islein[0]->printo();
		Isleloop::predloop(sync_indice);
	}*/
};
class Coaxisle : public CoupleIsle {
public:
	Flux *dualbgate;

	void buildislet(Flux *in, Flux *tar)
	{
		intt dims[MX_DIM];

		convert_2d_to_1d(in, dims);
		islein[0] = flux(isletrc, 3, dims, in->qType, variable);

		convert_2d_to_1d(tar, dims);
		islein[1] = flux(isletrc, 3, dims, tar->qType, variable);
		
		if(isleimp->impTcr->bygate && szembede < 0) {//���̰���Ʈ�� �����ư� classification����
			//ȣ��Ǿ����� ��� ���ڴ��� �����Ѵ�. ��������Ʈ�� �ܺο��� ���� �����Ͱ� �ε�Ǳ⶧���� 
			//�Էµ����Ϳ� ��ũ�� ����������Ƿ� �н��� ��������� �����Ѵ�.
			dualbgate = isletrc->migbygate(isleimp->impTcr);
			//setdgendual();
		}
		islenet = generic(islein[0], islein[1], szlatent, isindisc, isoutdisc, szembede, actcode, rLeaning);//csnr.
		isleout = ((Generic *)islenet)->cypred;
		//isleimp->accucell = islenet;//ipare.
		//isleimp->accutrc = isletrc;
	}
	void setdgendual(void);
	Coaxisle(Impulse *imp, Dynagen *dgen, Flux *in_gate, Flux *tar_gate, sytet head, bytet *name) : CoupleIsle(imp, dgen, head, name)
	{
		dualbgate = nullx;
		if(isletrc->printBuilding) {
			printf("coax isle in: ");
			in_gate->shape();
			printf("coax isle tar: ");
			tar_gate->shape();
		}
		extingate = in_gate;
		exttargate = tar_gate;
		setdgencoup(dgen);
		buildislet(in_gate, tar_gate);
		noperand = 2;
	}
};
class Stratusisle : public CoupleIsle {
public:
	void buildislet(Flux *in, Flux *tar)
	{
		intt dims[MX_DIM];

		convert_2d_to_1d(in, dims);//��Ҹ� �ٲ�� ������� ����, inject���� ���� ��.
		islein[0] = flux(isletrc, 3, dims, in->qType, variable);

		convert_2d_to_1d(tar, dims);
		islein[1] = flux(isletrc, 3, dims, tar->qType, variable);

		islenet = stratus(islein[0], islein[1], szlatent, isindisc, isoutdisc, szembede, actcode, rLeaning, nullx);
		//isleimp->accucell = islenet;//ipare.
		//isleimp->accutrc = isletrc;
	}
	Flux *trainislet(Flux *in, Flux *tar)
	{
		return ((Stratus *)islenet)->train();
	}
	Flux *predislet(Flux *in, Flux **loss)
	{
		return ((Stratus *)islenet)->predict(loss);
	}
	Stratusisle(Impulse *imp, Dynagen *dgen, Flux *in_gate, Flux *tar_gate, sytet head, bytet *name) : CoupleIsle(imp, dgen, head, name)
	{
		if(isletrc->printBuilding) {
			printf("stratus isle in: ");
			in_gate->shape();
			printf("stratus isle tar: ");
			tar_gate->shape();
		}
		setdgencoup(dgen);
		buildislet(in_gate, tar_gate);
		noperand = 2;
	}
};
#define T_DG_TRANSLATE	0
#define T_DG_RECALL		1
#define T_DG_CALSS		2
#define T_DG_CONNECT	3
#define T_DG_ENCODE		4

//translate, recall, classification, connect
class Dynagen : public Typer {
public:
	Impulse *dgenimp;
	Isleloop *inCodeisle, *coupleisle;
	Encodeisle *tarCodeisle;
	intt inz_bound, tarz_bound, coupConvolv;
	intt szlatent, dgindisc, dgoutdisc, szembede;
	sytet actcode, lossIndice;
	floatt rLeaning, couplowbo;
	Flux *dgenloss[2], *dgenpred, *dgenbgate, *dumloss;
	lsemx semdgen;
	CydChain *cydgen;
	Dynagen *ptrLeft, *ptrRight;
	bytet dgenname[128];
	void dgeninst(Impulse *imp, const bytet *name)
	{
		intt rv;

		dgenimp = imp;
		imp->regidgen(this);

		cydgen = rsc::cydrome->getcydid();
		CRE_LSEM_(semdgen, 0, cydgen->sem_iden, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre sem fail\n");
		}
		dgenpred = nullx;

		if(name) {
			if(strlen(name) >= 128) throwFault(-1, (bytex *)"name over fail\n");
			sprintf(dgenname, "%s_%d", imp->cntdgen++, name);
		} else sprintf(dgenname, "_dynagen_%d", imp->cntdgen++);
	}
	void dgenremove(void)
	{
		CLOSE_LSEM_(semdgen);
		rsc::cydrome->putcydid(cydgen);
	}
	void sigprec(Flux *pred)
	{
		dgenpred = pred;
		SIG_LSEM_(semdgen);
	}
	void waitprec(void)
	{
		WAIT_LSEM_(semdgen);
	}
	Flux *predictv(Flux **loss)
	{
		//printf("return loss: %p pred: %p\n", dgenloss[0], dgenpred);
		if(loss) *loss = dgenloss[0];//���� Ŀ�� isle�� �ν�
		dgenloss[0] = dumloss;//������ dgenloss�� ���ο� �ν��� �����Ǳ����� ȣ��Ǹ� ���̷ν��� 
		return dgenpred;		//�ٽ� �����ǵ��� 
	}
	void dumyloss(void)
	{
		intx dim[1];
		dim[0] = 1;
		dgenloss[0] = dumloss = flux(dgenimp->impTcr, 1, dim, tfloat, variable);
	}
	Flux *getbygate(void) { return dgenbgate; }
	Dynagen(sytet t_dynagen, Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret,
		intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)
	{
		Isleloop *isl, *next;
		Isleloop *in_head_isle, *tar_head_isle;
		sytet tar_sid = -1;
		intt rsz, i;
		bytet _name[128];

		dgeninst(imp, name);

		szlatent = latent_sz;
		dgindisc = indiscret;
		dgoutdisc = outdiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;

		inz_bound = imp->impTcr->inzbound;
		tarz_bound = imp->impTcr->tarzbound;
		in_head_isle = tar_head_isle = nullx;
		if(t_dynagen < T_DG_CALSS) {//translate, recall, 8���̱��� �����Ѵ�. Ÿ�� ������ ���ڵ�/���ڵ� �н�
			if(t_dynagen == T_DG_RECALL) {//�����̸� �Է�����Ʈ�� �Ű��ǳ� Ÿ�� ������ ���Ͻ��� ó��, �����Ҷ� ingate��
				targate = ingate;			//�Է��ϰ� ������Ʈ�� ���ڴ��� ����� �޴´�.
				tarz_bound = inz_bound;
			}
			rsz = tarz_bound;
			if(t_dynagen == T_DG_TRANSLATE) lossIndice = 1;//�����̸� Ÿ�ٽ����� �ν��� 1���� ����
			else lossIndice = 0;//������ �ƴϸ� Ÿ�ٽ����� �н��� �����Ƿ� �ν��� 0������ �����ϸ� �ȴ�.
			i = 0; sprintf(_name, "tar_encode_isle_%d", i++);
			for(tar_head_isle = isl = next = new(dgenimp->impTcr) Encodeisle(dgenimp, this, targate, rsz, 1, HOD, _name); rsz > 0; isl = next) {
				sprintf(_name, "tar_encode_isle_%d", i++);
				next = new(dgenimp->impTcr) Encodeisle(dgenimp, this, next->isleout, rsz, 1, NHNDT, _name);//���ڴ��� �԰� ����
				isl->regiRouteTo(next, -1);
				((Encodeisle *)next)->decisle->regiRouteTo(((Encodeisle *)isl)->decisle, -1);//decoder�� <-- ���� ����
			}
			tarCodeisle = (Encodeisle *)next;//last encoder
			tarCodeisle->decisle->finalDenseDec = 1;
			((Encodeisle *)tar_head_isle)->decisle->setFinal(1);
			((Encodeisle *)tar_head_isle)->decisle->adaptOrigin(ingate);
			if(t_dynagen == T_DG_RECALL) {//���� ���ڴ��� ����� ���ڴ� �Է����� ����(������ ���� ����)
				tarCodeisle->regiRouteTo(tarCodeisle->decisle, -1);
			}
		}
		lossIndice = 0;//�Է� ������(ingate) isle�� Ŀ�� isle �ν��� 0���� ����
		if(t_dynagen == T_DG_TRANSLATE || t_dynagen == T_DG_CALSS || t_dynagen == T_DG_ENCODE) {
			rsz = inz_bound;
			i = 0; sprintf(_name, "in_encode_isle_%d", i++);
			for(in_head_isle = isl = next = new(dgenimp->impTcr) Encodeisle(dgenimp, this, ingate, rsz, 0, HID, _name); rsz > 0; isl = next) {
				sprintf(_name, "in_encode_isle_%d", i++);
				next = new(dgenimp->impTcr) Encodeisle(dgenimp, this, next->isleout, rsz, 0, NHNDI, _name);//�Է����� ������ �ʿ�����Ƿ� ���ڴ� ���� ���Ѵ�.
				isl->regiRouteTo(next, -1);
			}
			inCodeisle = next;//last encoder
			coupConvolv = 4;
			couplowbo = 1;
			strcpy(_name, "couple_isle");
			if(t_dynagen == T_DG_TRANSLATE) {
				if(imp->impTcr->stratusOpt == 0) {
					coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, inCodeisle->isleout, tarCodeisle->isleout, NHNDC, _name);
				} else {
					coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, inCodeisle->isleout, tarCodeisle->isleout, NHNDC, _name);
				}
				inCodeisle->regiRouteTo(coupleisle, 0);//�������� �Է� ����ø����� ����
				tarCodeisle->regiRouteTo(coupleisle, 1);//�Ʒö��� �Է°� Ÿ�� �ΰ��� �������� ��ġ�ؾ� ����.
				coupleisle->regiRouteTo(tarCodeisle->decisle, -1);//Ŀ�� isle�� �����ٷ��� Ÿ�� ����isle���ڴ� �Է����� ����(������ ���� ����)
			} else if(t_dynagen == T_DG_CALSS) {//classification
				if(imp->impTcr->stracode == 0) {//Coaxisle�� �Է��� �����ڵ��̹Ƿ� -1 ����.������ڴ� �����϶� �ǹ�.
					szembede = -1;//���� �� ������ dgen������ ���̻� ���� �����Ƿ� ������ �ʿ����.
				}//else dncs.stracode�̸� ���̳������� ���� ���ڵ��ϰ� �Ʒ� ���ʸ����� �ѹ��� �߰� ���ڵ��� 
				//���� ������ڴ��� �Է��ϰ� �ϵ��� szembede�� �״�� �д�.
				coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, inCodeisle->isleout, targate, NHOC, _name);
				dgenbgate = ((Coaxisle *)coupleisle)->dualbgate;
				inCodeisle->regiRouteTo(coupleisle, 0);//�Է��� ����isle�� ����� Ŀ���� �Է����� ����
				tar_head_isle = coupleisle;//��ǥ���� Ŀ��isle�� ��ǥ������ �ٷ� ����
				tar_sid = 1;//��ǥ������ ���ԵǾ��ϹǷ� 1.
				coupleisle->setFinal(1);
			} else inCodeisle->setFinal(1);//T_DG_ENCODE, ����� ��������, �ľ� ����Ǹ� ���µ�.
		}
		if(in_head_isle) dgenimp->regifeed(in_head_isle, -1);
		if(tar_head_isle) dgenimp->regifeed(tar_head_isle, tar_sid);//recall�̸� �Է� �Ű������� Ÿ�� ������ ����Ǿ� ��ϵȴ�.
	}
	//���� �Է����� Isleloop�� ������ �̴�.
	Dynagen(Impulse *imp, Isleloop *in_zc_isle, Isleloop *tar_zc_isle, bool pred_link = 1, 
		sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)//T_DG_CONNECT
	{
		dgeninst(imp, name);

		in_zc_isle->setFinal(0);//encoder�� ��� �̹��� ������ ����ǹǷ� ������ �ƴϰ� �Ǿ� �����Ѵ�.
		tar_zc_isle->setFinal(0);
		actcode = af;
		rLeaning = lr;
		lossIndice = 0;
		if(imp->impTcr->stratusOpt != 0) {
			coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, in_zc_isle->isleout, tar_zc_isle->isleout, NHNDC, (bytet *)"copule_isle");
		} else {
			coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, in_zc_isle->isleout, tar_zc_isle->isleout, NHNDC, (bytet *)"copule_isle");
		}
		in_zc_isle->regiRouteTo(coupleisle, 0);//�����ڵ� �Է°��� Ŀ��isle�� �Է°����� ����
		tar_zc_isle->regiRouteTo(coupleisle, 1);//�����ڵ� Ÿ�ٰ��� Ŀ��isle�� Ÿ�ٰ����� ����
		if(pred_link) {//������ ���� ����, ĿǮ�� ���� ��°��� Ÿ��isle�� ���ڴ��� �Է°����� ����,�̶� Ÿ��isle��
			coupleisle->regiRouteTo(((Encodeisle *)tar_zc_isle)->decisle, -1);//T_DG_RECALL�� ����Ǿ� �Ѵ�.
		} else coupleisle->setFinal(1);
	}
	Dynagen(Impulse *imp, Flux *ingate, intt latent_sz, intt indiscret, intt embedim, Isleloop *tar_zc_isle,
		bool pred_link = 1, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)
	{
		dgeninst(imp, name);

		tar_zc_isle->setFinal(0);//encoder�� ��� �̹��� ������ ����ǹǷ� ������ �ƴϰ� �Ǿ� �����Ѵ�.
		szlatent = latent_sz;
		dgindisc = indiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;
		lossIndice = 0;
		if(imp->impTcr->stratusOpt != 0) {
			coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, ingate, tar_zc_isle->isleout, HID, (bytet *)"copule_isle");
		} else {
			coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, ingate, tar_zc_isle->isleout, HID, (bytet *)"copule_isle");
		}
		tar_zc_isle->regiRouteTo(coupleisle, 1);//�����ڵ� Ÿ�ٰ��� Ŀ��isle�� Ÿ�ٰ����� ����
		if(pred_link) {//������ ���� ����, ĿǮ�� ���� ��°��� Ÿ��isle�� ���ڴ��� �Է°����� ����,�̶� Ÿ��isle��
			coupleisle->regiRouteTo(((Encodeisle *)tar_zc_isle)->decisle, -1);//T_DG_RECALL�� ����Ǿ� �Ѵ�.
		} else coupleisle->setFinal(1);
		dgenimp->regifeed(coupleisle, 0);//�Է°��� Ŀ��isle�� �Է°����� ����
	}
	Dynagen(Impulse *imp, Isleloop *in_zc_isle, Flux *targate, intt latent_sz, intt outdiscret,
		intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx) //���� �ڱⱸ�� �н����� �ٷ� Ŀ��(����)isle�� ����
	{
		dgeninst(imp, name);

		in_zc_isle->setFinal(0);//encoder�� ��� �̹��� ������ ����ǹǷ� ������ �ƴϰ� �Ǿ� �����Ѵ�.
		szlatent = latent_sz;
		dgoutdisc = outdiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;
		lossIndice = 0;
		if(imp->impTcr->stratusOpt != 0) {
			coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, in_zc_isle->isleout, targate, HOD, (bytet *)"copule_isle");
		} else {
			coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, in_zc_isle->isleout, targate, HOD, (bytet *)"copule_isle");
		}
		in_zc_isle->regiRouteTo(coupleisle, 0);//�����ڵ� �Է°��� Ŀ��isle�� �Է°����� ����
		dgenimp->regifeed(coupleisle, 1);//��ǥ���� Ŀ��isle�� ��ǥ������ ����
		coupleisle->setFinal(1);
	}
	Dynagen(Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret,
		intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)
	{//���� �ڱⱸ�� �н����� �ٷ� Ŀ��(����)isle�� ����, ���� Coaxisle�̳� Stratusisle�� ���̳��� ������ ����Ҷ�.
		dgeninst(imp, name);

		szlatent = latent_sz;
		dgindisc = indiscret;
		dgoutdisc = outdiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;
		lossIndice = 0;
		if(imp->impTcr->stratusOpt != 0) {
			coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, ingate, targate, HAD, (bytet *)"copule_isle");
		} else {
			coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, ingate, targate, HAD, (bytet *)"copule_isle");
		}
		dgenimp->regifeed(coupleisle, -2);//�Է°��� Ÿ�ٰ��� Ŀ��isle�� �Է°� Ÿ�ٰ����� ����
		coupleisle->setFinal(1);
	}
	/*Dynagen(Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret,
		intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)
	{//����� �׽�Ʈ �ܵ� ����� ������.
		Isleloop *isl, *next;
		Isleloop *in_head_isle, *tar_head_isle;
		intt rsz;
		sytet tar_sid = 0;

		dgeninst(imp, name);

		szlatent = latent_sz;
		dgindisc = indiscret;
		dgoutdisc = outdiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;

		inz_bound = imp->impTcr->inzbound;
		tarz_bound = imp->impTcr->tarzbound;
		in_head_isle = tar_head_isle = nullx;
		rsz = inz_bound;
		lossIndice = 0;
		Encodeisle *ei = new(dgenimp->impTcr) Encodeisle(dgenimp, this, ingate, rsz, 1, HID, (bytet *)"copule_isle");

		intt dim[MX_DIM];
		memcpy(dim, ingate->fshape, 4 * sizeof(intt));
		dim[0] = 10;

		Flux *tf = flux(imp->impTcr, ingate->fdim, dim, tfloat, variable);
		tf->randn(0, 1);
		ei->islein[0]->feedf(tf);
		ei->findectrain = 1;//�ؿ��� ���ڴ��� ���� ������ ���̹Ƿ�
		while(1) {
			ei->trainloop(1);
			ei->predloop(1);
			ei->decisle->islein[0]->feedf(ei->isleout);
			ei->decisle->islein[1]->feedf(ei->islein[0]);
			//ei->decisle->islein[0]->printo(2, 10);
			//ei->decisle->islein[1]->printo(2, 10);
			ei->decisle->trainloop(1);
			dgenloss[lossIndice]->printo();
		}
	}*/
};
Impulse *impluse(Tracer *tcr, const bytet *name = nullx);
Dynagen *dynagen(sytet t_dynagen, Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret,
	intt outdiscret, intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);
Dynagen *dynagen(Impulse *imp, Isleloop *in_zc_isle, Isleloop *tar_zc_isle, bool pred_link = 1,
	sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);
Dynagen *dynagen(Impulse *imp, Flux *ingate, intt latent_sz, intt indiscret, intt embedim, Isleloop *tar_zc_isle,
	bool pred_link = 1, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);
Dynagen *dynagen(Impulse *imp, Isleloop *in_zc_isle, Flux *targate, intt latent_sz, intt outdiscret,
	intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);
Dynagen *dynagen(Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret,
	intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);